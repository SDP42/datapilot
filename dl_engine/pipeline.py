"""Phase 14.14 — deep learning pipeline orchestration.

:func:`run_mlp_pipeline` is the **first place an MLP is reachable end to
end from a raw DataFrame + objective** — every Phase-8 building block
(`architectures.py`, `execution.py`, `contracts.py`) existed already, but
nothing composed them the way
:func:`data_engine.modeling.pipeline.run_modeling_pipeline` composes the
classical path. This module does exactly that, for the one architecture
(MLP) that supports both regression and classification.

A composition layer only: task/feature resolution reuses the same
public Phase-5/6/7 functions `run_modeling_pipeline` itself calls
(`build_problem_spec`, `build_feature_engineering_spec`,
`assess_model_readiness`). The one piece of real logic here is turning
a DataFrame into the numeric `(X, y)` arrays `run_mlp_modeling` requires
— median-impute + standard-scale numeric columns, one-hot encode
categoricals, label-encode a classification target — since this module
deliberately does not reach into `data_engine.modeling.training`'s own
*private* preprocessor (`dl_engine` only ever imports `data_engine`'s
public API; see every other module in this package).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.modeling import (
    ModelingRequest,
    ModelingStatus,
    TrainingRunStatus,
    assess_model_readiness,
    build_feature_engineering_spec,
    build_problem_spec,
)
from data_engine.problem_understanding import TaskType
from datapilot.contracts import ColumnType

from .architectures import MLPArchitectureConfig
from .contracts import DLLoss, DLModelingResult, DLTrainingConfig
from .execution import run_mlp_modeling

_TASK_TO_CATEGORY: dict[TaskType, str] = {
    TaskType.REGRESSION: "regression",
    TaskType.BINARY_CLASSIFICATION: "classification",
    TaskType.MULTICLASS_CLASSIFICATION: "classification",
}
_DEFAULT_HIDDEN_LAYERS = [64, 32]
_TEST_FRACTION = 0.2
_MLP_RANDOM_SEED = 42


def _unavailable(reason: str, task_type: TaskType = TaskType.OTHER) -> DLModelingResult:
    return DLModelingResult(
        status=TrainingRunStatus.UNAVAILABLE, task_type=task_type, reason=reason
    )


def _prepare_numeric_arrays(
    df: pd.DataFrame, feature_cols: list[str], target_column: str, category: str
) -> tuple[np.ndarray, np.ndarray, list[str]] | str:
    """Returns `(X, y, encoded_feature_names)` or an error string.

    Drops rows with a missing target first (the same convention the
    classical expanded search uses), then imputes/scales/encodes the
    feature columns and, for classification, label-encodes the target
    into `0..n_classes-1`.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    work = df[[*feature_cols, target_column]].dropna(subset=[target_column]).reset_index(drop=True)
    if work.empty:
        return "every row has a missing target value; nothing is left to train on"

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False, max_categories=30
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        return "no usable numeric / categorical feature columns are available"

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    try:
        X = np.asarray(preprocessor.fit_transform(work[feature_cols]), dtype=np.float64)
    except ValueError as exc:
        return f"could not build a numeric feature matrix: {exc}"

    if category == "regression":
        y = work[target_column].to_numpy(dtype=np.float64)
    else:
        y = LabelEncoder().fit_transform(work[target_column]).astype(np.float64)

    return X, y, list(preprocessor.get_feature_names_out())


def run_mlp_pipeline(
    df: pd.DataFrame,
    request: ModelingRequest,
    *,
    hidden_layer_sizes: list[int] | None = None,
    epochs: int = 100,
) -> DLModelingResult:
    """Build, train, and evaluate one MLP end to end for `df` + `request`.

    Regression and classification only (the Phase-8.3 MLP's own
    supported set) — clustering, forecasting, and anything task-type
    inference couldn't resolve report `status=unavailable`.
    """
    problem = build_problem_spec(df, request)
    task_inference = problem.task_type
    if task_inference.status.value != "completed" or task_inference.task_type is None:
        return _unavailable("task-type inference is not completed")
    task = task_inference.task_type
    category = _TASK_TO_CATEGORY.get(task)
    if category is None:
        return _unavailable(
            f"the MLP pipeline supports regression and classification only (got '{task.value}')",
            task_type=task,
        )

    feature_engineering = build_feature_engineering_spec(df, request, problem)
    readiness = assess_model_readiness(
        df, problem, feature_engineering, objective=request.objective
    )
    if readiness.status is not ModelingStatus.COMPLETED or readiness.ready is False:
        first = (
            readiness.blocking_issues[0]
            if readiness.blocking_issues
            else (readiness.reason or "the data is not ready for modeling")
        )
        return _unavailable(
            f"training is blocked by model-readiness issues: {first}", task_type=task
        )

    target_column = problem.target.target_column
    if target_column is None:
        return _unavailable("no target column was identified", task_type=task)

    selection = feature_engineering.selection
    col_type = {c.column: c.column_type for c in feature_engineering.inventory.candidates}
    candidates = (
        sorted(set(selection.selected_features) | set(selection.review_features))
        if selection.status.value == "completed"
        else sorted(feature_engineering.inventory.candidate_features)
    )
    feature_cols = sorted(
        c
        for c in candidates
        if c != target_column
        and col_type.get(c) in (ColumnType.NUMERIC, ColumnType.BOOLEAN, ColumnType.CATEGORICAL)
    )
    if not feature_cols:
        return _unavailable(
            "no usable numeric / categorical feature columns are available", task_type=task
        )

    prepared = _prepare_numeric_arrays(df, feature_cols, target_column, category)
    if isinstance(prepared, str):
        return _unavailable(prepared, task_type=task)
    X, y, _encoded_names = prepared

    n_classes = len(np.unique(y)) if category == "classification" else 0
    if category == "classification" and n_classes < 2:
        return _unavailable("the target has fewer than 2 distinct classes", task_type=task)

    from sklearn.model_selection import train_test_split

    try:
        stratify = y if category == "classification" else None
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=_TEST_FRACTION, random_state=_MLP_RANDOM_SEED, stratify=stratify
        )
    except ValueError as exc:
        return _unavailable(
            f"could not split the data for training/evaluation: {exc}", task_type=task
        )

    output_dim = (
        1
        if category == "regression"
        else (2 if task is TaskType.BINARY_CLASSIFICATION else n_classes)
    )
    architecture = MLPArchitectureConfig(
        task_type=task,
        input_features=X.shape[1],
        output_dim=output_dim,
        hidden_layer_sizes=hidden_layer_sizes or _DEFAULT_HIDDEN_LAYERS,
    )
    training_config = DLTrainingConfig(
        architecture_name="mlp",
        task_type=task,
        seed=_MLP_RANDOM_SEED,
        epochs=epochs,
        loss=DLLoss.MSE if category == "regression" else DLLoss.CROSS_ENTROPY,
    )

    return run_mlp_modeling(X_train, y_train, X_eval, y_eval, architecture, training_config)
