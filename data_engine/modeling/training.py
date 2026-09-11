"""Phase 7.4 — deterministic baseline model training & evaluation.

:func:`train_and_evaluate_models` is the **first** DataPilot component
allowed to fit estimators and compute evaluation metrics. It:

* consumes the Phase-5 :class:`ProblemSpec`, the Phase-6
  :class:`FeatureEngineeringSpec`, and the Phase-7.2 / 7.3
  :class:`ModelReadiness` / :class:`DataSplitPlan` / :class:`ModelCandidates`;
* performs the actual train / validation / test split **exactly** as the
  :class:`DataSplitPlan` specifies, with a fixed random seed;
* executes only the Phase-6.5 preprocessing requirements, fitted **only**
  on the training partition (leakage-safe within this pipeline);
* fits one conservative baseline scikit-learn estimator per Phase-7.3
  candidate family;
* computes task-appropriate deterministic metrics on the test partition;
* returns a structured, JSON-primitive-only :class:`TrainingOutcome`.

It does **not** select, rank, or recommend a model; tune hyperparameters;
cross-validate; do model-based feature selection, SHAP, feature
importance, or leakage detection; generate features; re-infer the target
or task; use target encoding / SMOTE / PCA; persist any artifact; or
modify the input DataFrame or any upstream model.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from data_engine.feature_engineering import FeatureEngineeringSpec, FeatureEngineeringStatus
from data_engine.problem_understanding import ProblemSpec, ProblemUnderstandingStatus, TaskType
from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    DataSplitPlan,
    DataSplitStrategy,
    ModelCandidates,
    ModelFamily,
    ModelingStatus,
    ModelReadiness,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
)
from .temporal_execution import (
    build_calendar_features,
    build_temporal_features,
    temporal_feature_spec,
)

# --- fixed, documented tunables --------------------------------------------

# The one random seed used everywhere randomisation is required. A named
# module constant — never generated from the clock, environment, or system
# state.
MODEL_TRAINING_RANDOM_SEED = 42

MODEL_TRAINING_TREE_MAX_DEPTH = 8  # conservative depth cap for a baseline tree
MODEL_TRAINING_FOREST_N_ESTIMATORS = 100  # sklearn default; explicit for reproducibility
MODEL_TRAINING_KNN_N_NEIGHBORS = 5  # sklearn default; explicit
MODEL_TRAINING_N_CLUSTERS = (
    3  # fixed baseline cluster count (cluster-count selection is out of scope)
)
MODEL_TRAINING_LOGREG_MAX_ITER = 1000  # allow convergence on scaled features
MODEL_TRAINING_MLP_MAX_ITER = 200  # modest cap for the optional neural baseline
MODEL_TRAINING_METRIC_ROUND = 6  # decimal places for every reported metric
MODEL_TRAINING_MIN_TRAIN_ROWS = 5  # fewer -> the run is unavailable
MODEL_TRAINING_MIN_TEST_ROWS = 1  # fewer -> the run is unavailable
# Forecasting-execution: after building lag / rolling features and dropping the
# warm-up rows, fewer modelable rows than this -> the whole outcome is unavailable.
MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS = 20

_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})
_TASK_CATEGORY: dict[TaskType, str] = {
    TaskType.REGRESSION: "regression",
    TaskType.TIME_SERIES_FORECASTING: "regression",
    TaskType.BINARY_CLASSIFICATION: "classification",
    TaskType.MULTICLASS_CLASSIFICATION: "classification",
    TaskType.CLUSTERING: "clustering",
}

_OP_IMPUTATION = "missing-value imputation"
_OP_ENCODING = "categorical encoding"
_OP_SCALING = "numerical scaling"

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

try:  # pragma: no cover - environment dependent
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# --- helpers -------------------------------------------------------------


def _normalise_error(message: str) -> str:
    """Strip nondeterministic detail (memory addresses) from an exception message."""
    return _ADDR_RE.sub("0x...", message).strip()


def _verify_chronological_order(df: pd.DataFrame, time_column: str | None) -> tuple[bool, str]:
    """Deterministically check that a forecasting frame's rows are chronological.

    For ``time_ordered_holdout`` the **row order is the time axis** — Phase
    7.4 slices it positionally and never sorts or infers a time column.

    * ``time_column`` given (the Phase-5.3 resolved axis) — that exact column
      must be present and non-decreasing.
    * ``time_column`` ``None`` (a caller not going through Phase 5) — fall back
      to the heuristic: the frame must be non-decreasing on one of its own
      datetime columns.

    Returns ``(ok, detail)`` where ``detail`` is a note on success or an
    explicit reason on failure.
    """
    column_names = [str(c) for c in df.columns]

    if time_column is not None:
        if time_column not in column_names:
            return False, (f"the declared time column '{time_column}' is not in the DataFrame")
        parsed = pd.to_datetime(
            df.iloc[:, column_names.index(time_column)], errors="coerce"
        ).dropna()
        if len(parsed) < 2:
            return False, (
                f"the declared time column '{time_column}' has fewer than 2 usable timestamps"
            )
        if parsed.is_monotonic_increasing:
            return True, (
                f"chronological-order precondition satisfied: the declared time column "
                f"'{time_column}' is non-decreasing across the supplied rows"
            )
        return False, (
            f"the rows are not in chronological order on the declared time column "
            f"'{time_column}'; sort the DataFrame chronologically before modeling — Phase 7.4 "
            "does not reorder rows"
        )

    datetime_cols = sorted(
        name
        for i, name in enumerate(column_names)
        if infer_column_type(df.iloc[:, i]) is ColumnType.DATETIME
    )
    if not datetime_cols:
        return False, (
            "time_series_forecasting uses row order as the time axis, but the DataFrame has no "
            "datetime column to verify that the rows are in chronological order; sort the rows "
            "chronologically and keep the timestamp column in the frame"
        )
    for name in datetime_cols:
        parsed = pd.to_datetime(df.iloc[:, column_names.index(name)], errors="coerce").dropna()
        if len(parsed) >= 2 and parsed.is_monotonic_increasing:
            return True, (
                f"chronological-order precondition satisfied: datetime column '{name}' is "
                "non-decreasing across the supplied rows"
            )
    return False, (
        "time_series_forecasting uses row order as the time axis, but none of the datetime "
        f"column(s) ({', '.join(datetime_cols)}) is non-decreasing across the supplied rows; "
        "sort the DataFrame chronologically before modeling — Phase 7.4 does not reorder rows"
    )


def _unavailable(reason: str, *, objective_used: bool) -> TrainingOutcome:
    return TrainingOutcome(
        status=ModelingStatus.UNAVAILABLE,
        reason=reason,
        runs=[],
        successful_runs=[],
        failed_runs=[],
        objective_used=objective_used,
        notes=[],
    )


def _eligible_features(feature_engineering: FeatureEngineeringSpec) -> list[str]:
    selection = feature_engineering.selection
    if selection.status is _FE_COMPLETED:
        return sorted(set(selection.selected_features) | set(selection.review_features))
    return sorted(feature_engineering.inventory.candidate_features)


def _round(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(float(value), MODEL_TRAINING_METRIC_ROUND)


def _build_estimator(family: ModelFamily, category: str) -> tuple[str, Any] | None:
    """The fixed, documented family -> concrete baseline estimator mapping."""
    seed = MODEL_TRAINING_RANDOM_SEED
    if family is ModelFamily.LINEAR:
        if category == "regression":
            from sklearn.linear_model import LinearRegression

            return "LinearRegression", LinearRegression()
        if category == "classification":
            from sklearn.linear_model import LogisticRegression

            return "LogisticRegression", LogisticRegression(
                max_iter=MODEL_TRAINING_LOGREG_MAX_ITER, random_state=seed
            )
    elif family is ModelFamily.TREE_BASED:
        if category == "regression":
            from sklearn.tree import DecisionTreeRegressor

            return "DecisionTreeRegressor", DecisionTreeRegressor(
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH, random_state=seed
            )
        if category == "classification":
            from sklearn.tree import DecisionTreeClassifier

            return "DecisionTreeClassifier", DecisionTreeClassifier(
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH, random_state=seed
            )
    elif family is ModelFamily.ENSEMBLE:
        if category == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return "RandomForestRegressor", RandomForestRegressor(
                n_estimators=MODEL_TRAINING_FOREST_N_ESTIMATORS,
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH,
                random_state=seed,
                n_jobs=1,
            )
        if category == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return "RandomForestClassifier", RandomForestClassifier(
                n_estimators=MODEL_TRAINING_FOREST_N_ESTIMATORS,
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH,
                random_state=seed,
                n_jobs=1,
            )
    elif family is ModelFamily.PROBABILISTIC:
        if category == "classification":
            from sklearn.naive_bayes import GaussianNB

            return "GaussianNB", GaussianNB()
        if category == "clustering":
            from sklearn.mixture import GaussianMixture

            return "GaussianMixture", GaussianMixture(
                n_components=MODEL_TRAINING_N_CLUSTERS, random_state=seed
            )
    elif family is ModelFamily.DISTANCE_BASED:
        if category == "regression":
            from sklearn.neighbors import KNeighborsRegressor

            return "KNeighborsRegressor", KNeighborsRegressor(
                n_neighbors=MODEL_TRAINING_KNN_N_NEIGHBORS, n_jobs=1
            )
        if category == "classification":
            from sklearn.neighbors import KNeighborsClassifier

            return "KNeighborsClassifier", KNeighborsClassifier(
                n_neighbors=MODEL_TRAINING_KNN_N_NEIGHBORS, n_jobs=1
            )
        if category == "clustering":
            from sklearn.cluster import KMeans

            return "KMeans", KMeans(
                n_clusters=MODEL_TRAINING_N_CLUSTERS, random_state=seed, n_init=10
            )
    elif family is ModelFamily.NEURAL:
        if category == "regression":
            from sklearn.neural_network import MLPRegressor

            return "MLPRegressor", MLPRegressor(
                max_iter=MODEL_TRAINING_MLP_MAX_ITER, random_state=seed
            )
        if category == "classification":
            from sklearn.neural_network import MLPClassifier

            return "MLPClassifier", MLPClassifier(
                max_iter=MODEL_TRAINING_MLP_MAX_ITER, random_state=seed
            )
    return None


def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    req_by_col: dict[str, set[str]],
) -> tuple[Any, str | None]:
    """A leakage-safe ColumnTransformer built strictly from Phase-6.5 requirements."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        numeric_ops = {op for col in numeric_cols for op in req_by_col.get(col, set())}
        steps: list[tuple[str, Any]] = []
        if _OP_IMPUTATION in numeric_ops:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        if _OP_SCALING in numeric_ops:
            steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps) if steps else "passthrough"
        transformers.append(("numeric", numeric_pipeline, sorted(numeric_cols)))

    if categorical_cols:
        categorical_ops = {op for col in categorical_cols for op in req_by_col.get(col, set())}
        if _OP_ENCODING not in categorical_ops:
            return None, (
                "categorical feature(s) are present but Phase 6.5 identified no encoding "
                "requirement; Phase 7.4 does not invent an encoder"
            )
        steps = []
        if _OP_IMPUTATION in categorical_ops:
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformers.append(("categorical", Pipeline(steps), sorted(categorical_cols)))

    if not transformers:
        return "passthrough", None
    return ColumnTransformer(transformers, remainder="drop"), None


def _split_indices(
    n: int, plan: DataSplitPlan, y: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Deterministic train / validation / test index partitions (positions into df_work)."""
    train_f = plan.train_fraction or 0.0
    test_f = plan.test_fraction or 0.0
    val_f = plan.validation_fraction
    notes: list[str] = []

    all_idx = np.arange(n)

    if plan.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT:
        n_train = round(n * train_f)
        n_val = round(n * val_f) if val_f else 0
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        return (
            all_idx[:n_train],
            all_idx[n_train : n_train + n_val],
            all_idx[n_train + n_val :],
            notes,
        )

    stratified = plan.strategy is DataSplitStrategy.STRATIFIED_HOLDOUT and y is not None
    if stratified and y is not None:
        from sklearn.model_selection import train_test_split

        try:
            rest, test_idx = train_test_split(
                all_idx,
                test_size=test_f,
                stratify=y,
                random_state=MODEL_TRAINING_RANDOM_SEED,
                shuffle=True,
            )
            if val_f:
                rel_val = val_f / max(1e-9, (1.0 - test_f))
                train_idx, val_idx = train_test_split(
                    rest,
                    test_size=rel_val,
                    stratify=y[rest],
                    random_state=MODEL_TRAINING_RANDOM_SEED,
                    shuffle=True,
                )
            else:
                train_idx, val_idx = rest, np.empty(0, dtype=int)
            return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), notes
        except ValueError:
            notes.append(
                "stratified split was not possible (a class has too few members); falling "
                "back to a shuffled random holdout"
            )

    rng = np.random.default_rng(MODEL_TRAINING_RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = round(n * train_f)
    n_val = round(n * val_f) if val_f else 0
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    return (
        np.sort(perm[:n_train]),
        np.sort(perm[n_train : n_train + n_val]),
        np.sort(perm[n_train + n_val :]),
        notes,
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics: dict[str, float] = {
        "rmse": _round(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": _round(mean_absolute_error(y_true, y_pred)),
    }
    if float(np.var(y_true)) > 0.0:
        metrics["r2"] = _round(r2_score(y_true, y_pred))
    return metrics


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": _round(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None and n_classes == 2 and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = _round(roc_auc_score(y_true, proba))
        except ValueError:
            pass
    return metrics


def _recursive_horizon_metrics(
    pipeline: Any,
    x_all: pd.DataFrame,
    y_all: np.ndarray,
    feature_cols: list[str],
    target_spec: dict[str, tuple[str, str, int]],
    test_start: int,
    n_rows: int,
    horizon: int,
) -> dict[str, float]:
    """Rolling-origin **recursive** multi-step RMSE, one value per horizon step.

    From every origin ``o`` in ``[max(test_start, max_lookback), n_rows - horizon]``
    forecast ``horizon`` steps: a target lag / rolling feature is rebuilt from the
    growing ``actuals[:o] + predictions[o:]`` history; every other feature
    (exogenous, calendar) uses its **actual** row value (the standard assumption
    that regressors and the calendar are known over the forecast window).
    Deterministic; no metric here ever feeds ``select_model`` — the selection
    metric stays the one-step ``rmse``.
    """
    y = np.asarray(y_all, dtype=float)
    actual_features = {c: x_all[c].to_numpy() for c in feature_cols}
    target_kinds = {c: target_spec[c] for c in feature_cols if c in target_spec}
    max_lookback = max((n for _, _, n in target_kinds.values()), default=0)
    start = max(test_start, max_lookback)
    per_h_sq: list[list[float]] = [[] for _ in range(horizon)]

    for origin in range(start, n_rows - horizon + 1):
        preds: list[float] = []
        for step in range(horizon):
            pos = origin + step
            history = np.concatenate([y[:origin], np.asarray(preds, dtype=float)])
            row: dict[str, float] = {}
            for col in feature_cols:
                spec = target_kinds.get(col)
                if spec is None:
                    row[col] = float(actual_features[col][pos])
                    continue
                _, kind, k = spec
                if kind == "lag":
                    row[col] = float(history[pos - k]) if pos - k >= 0 else float("nan")
                else:
                    window = history[pos - k : pos]
                    if window.size == k:
                        row[col] = float(
                            window.mean() if kind == "rollmean" else window.std(ddof=1)
                        )
                    else:
                        row[col] = float("nan")
            pred = float(np.asarray(pipeline.predict(pd.DataFrame([row], columns=feature_cols)))[0])
            preds.append(pred)
            actual = y[pos]
            if math.isfinite(actual) and math.isfinite(pred):
                per_h_sq[step].append((pred - actual) ** 2)

    metrics: dict[str, float] = {}
    for h in range(horizon):
        if per_h_sq[h]:
            metrics[f"rmse_h{h + 1}"] = _round(math.sqrt(sum(per_h_sq[h]) / len(per_h_sq[h])))
    return metrics


def _clustering_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return {}
    return {
        "silhouette_score": _round(silhouette_score(features, labels)),
        "calinski_harabasz_score": _round(calinski_harabasz_score(features, labels)),
        "davies_bouldin_score": _round(davies_bouldin_score(features, labels)),
    }


# --- public API --------------------------------------------------------


def train_and_evaluate_models(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    candidates: ModelCandidates,
    *,
    objective: str | None = None,
    forecast_horizon: int = 1,
) -> TrainingOutcome:
    """Deterministically train & evaluate one baseline estimator per candidate.

    See the module docstring for the full boundary. ``status = unavailable``
    when an upstream contract does not permit execution; ``status =
    completed`` once execution was permitted (even if every individual
    candidate failed). Phase 7.4 never selects a model.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"train_and_evaluate_models expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(problem, ProblemSpec):
        raise TypeError(
            f"train_and_evaluate_models expects a ProblemSpec, got {type(problem).__name__}"
        )
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            "train_and_evaluate_models expects a FeatureEngineeringSpec, "
            f"got {type(feature_engineering).__name__}"
        )
    if not isinstance(readiness, ModelReadiness):
        raise TypeError(
            f"train_and_evaluate_models expects a ModelReadiness, got {type(readiness).__name__}"
        )
    if not isinstance(split, DataSplitPlan):
        raise TypeError(
            f"train_and_evaluate_models expects a DataSplitPlan, got {type(split).__name__}"
        )
    if not isinstance(candidates, ModelCandidates):
        raise TypeError(
            f"train_and_evaluate_models expects a ModelCandidates, got {type(candidates).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    # --- deterministic upstream precedence ---------------------------
    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED:
        return _unavailable(
            f"task-type inference is not completed (status = {task_inference.status.value})",
            objective_used=objective_used,
        )
    task = task_inference.task_type
    if task is None:
        return _unavailable(
            "task-type inference completed without a task type", objective_used=objective_used
        )
    if task in _UNSUPPORTED_TASKS:
        return _unavailable(
            f"model training does not support task type '{task.value}'",
            objective_used=objective_used,
        )
    if readiness.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"model readiness is not completed (status = {readiness.status.value})",
            objective_used=objective_used,
        )
    if readiness.ready is False:
        first = (
            readiness.blocking_issues[0]
            if readiness.blocking_issues
            else (readiness.reason or "no reason given")
        )
        return _unavailable(
            f"training is blocked by model-readiness issues: {first}",
            objective_used=objective_used,
        )
    if split.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"the data-split plan is not completed (status = {split.status.value})",
            objective_used=objective_used,
        )
    if candidates.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"model candidates are not available (status = {candidates.status.value})",
            objective_used=objective_used,
        )
    if feature_engineering.assessment.status is not _FE_COMPLETED:
        return _unavailable(
            "feature-engineering assessment is not completed "
            f"(status = {feature_engineering.assessment.status.value})",
            objective_used=objective_used,
        )
    if not _SKLEARN_AVAILABLE:
        return _unavailable(
            "scikit-learn is not available in this environment; Phase 7.4 cannot train "
            "baseline models",
            objective_used=objective_used,
        )

    # --- forecasting chronological-order precondition (audit H1) --------
    # A time-ordered holdout slices row order positionally as the time
    # axis. Phase 7.4 never infers a time column, sorts, or reorders — so
    # the rows must already be non-decreasing on one of the frame's own
    # datetime columns, otherwise the split (and every metric) would be
    # silently meaningless.
    chronological_note: str | None = None
    if split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT:
        ordered_ok, ordered_detail = _verify_chronological_order(df, problem.task_type.time_column)
        if not ordered_ok:
            return _unavailable(ordered_detail, objective_used=objective_used)
        chronological_note = ordered_detail

    category = _TASK_CATEGORY[task]
    is_supervised = category in {"regression", "classification"}

    # --- resolve features / target (from the Phase-5 / Phase-6 contracts) ---
    df_columns = [str(c) for c in df.columns]
    column_set = set(df_columns)
    target_column = problem.target.target_column if is_supervised else None

    col_type = {c.column: c.column_type for c in feature_engineering.inventory.candidates}
    eligible = [
        c for c in _eligible_features(feature_engineering) if c in column_set and c != target_column
    ]

    numeric_cols = sorted(
        c for c in eligible if col_type.get(c) in (ColumnType.NUMERIC, ColumnType.BOOLEAN)
    )
    categorical_cols = sorted(c for c in eligible if col_type.get(c) is ColumnType.CATEGORICAL)
    excluded_cols = sorted(
        c for c in eligible if col_type.get(c) in (ColumnType.DATETIME, ColumnType.UNKNOWN, None)
    )
    feature_cols = sorted(numeric_cols + categorical_cols)

    req_by_col: dict[str, set[str]] = {}
    for requirement in feature_engineering.preprocessing.requirements:
        req_by_col.setdefault(requirement.column, set()).add(requirement.description)

    notes: list[str] = [
        (
            "Phase 7.4 trained baseline estimators and computed evaluation metrics; it did "
            "NOT select a final model, rank models, tune hyperparameters, or cross-validate"
        ),
        (
            "preprocessing was executed strictly from the Phase 6.5 requirements and fitted "
            "only on the training partition (leakage-safe within this pipeline)"
        ),
        (
            "no model artifact was persisted; no fitted estimator, pipeline, prediction, or "
            "array is stored in this result"
        ),
        f"random seed: {MODEL_TRAINING_RANDOM_SEED} (fixed)",
        f"split strategy: {split.strategy.value if split.strategy else 'unspecified'}",
        f"task type: {task.value}",
    ]
    if target_column is not None:
        notes.append(f"target column '{target_column}' is excluded from the model features")
    if excluded_cols:
        notes.append(
            f"{len(excluded_cols)} datetime / unrecognised feature column(s) excluded "
            "(derivation / encoding of these is out of Phase-7.4 scope): "
            + ", ".join(excluded_cols)
        )
    if task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(
            "time-series forecasting is trained here as a baseline regression; Phase 7.4 "
            "builds no forecasting-specific model and no forecasting transformation beyond "
            "the Phase-6 lag / rolling recommendations; the task type came from Phase 5, "
            "never a datetime column"
        )
    if chronological_note is not None:
        notes.append(chronological_note)
        notes.append(
            "row order is the time axis for this time-ordered holdout; Phase 7.4 verified it "
            "against a datetime column and did not sort or reorder the rows"
        )
    temporal = feature_engineering.temporal
    if objective_used:
        notes.append("an objective was supplied and recorded; it did not change any training step")

    candidate_families: list[ModelFamily] = []
    for name in candidates.candidates:
        try:
            family = ModelFamily(name)
        except ValueError:
            continue
        if family not in candidate_families:
            candidate_families.append(family)

    if not candidate_families:
        return TrainingOutcome(
            status=ModelingStatus.COMPLETED,
            reason="no candidate model families were provided to train",
            runs=[],
            successful_runs=[],
            failed_runs=[],
            objective_used=objective_used,
            notes=notes,
        )

    # --- build the working frame (a copy; the input is never touched) ---
    is_time_ordered = split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    work = df.copy()
    work.columns = df_columns
    if is_supervised and target_column is not None:
        if is_time_ordered:
            # preserve contiguity for the positional time-ordered split: trim only
            # the leading / trailing run of missing-target rows (internal gaps are
            # blocked at Phase-5 feasibility).
            observed = work[target_column].notna().to_numpy()
            if observed.any():
                first_obs = int(observed.argmax())
                last_obs = len(observed) - 1 - int(observed[::-1].argmax())
                trimmed = len(work) - (last_obs - first_obs + 1)
                work = work.iloc[first_obs : last_obs + 1]
                if trimmed:
                    notes.append(
                        f"{trimmed} leading / trailing row(s) with a missing target were "
                        "trimmed (contiguity preserved for the time-ordered split)"
                    )
        else:
            before = len(work)
            work = work[work[target_column].notna()]
            dropped = before - len(work)
            if dropped:
                notes.append(
                    f"{dropped} row(s) with a missing target were excluded from supervised training"
                )
    work = work.reset_index(drop=True)

    # --- forecasting: execute the Phase-6 lag / rolling / calendar recommendations ---
    temporal_features_built = 0
    calendar_features_built = 0
    rows_consumed_as_history = 0
    forecasting_run = is_time_ordered and task is TaskType.TIME_SERIES_FORECASTING
    if forecasting_run and target_column is not None:
        built_temporal: list[str] = []
        built_calendar: list[str] = []
        try:
            if temporal.status is _FE_COMPLETED and temporal.recommendations:
                work, built_temporal, tf_notes = build_temporal_features(
                    work, temporal.recommendations
                )
                notes.extend(tf_notes)
            transformation_recs = feature_engineering.transformations
            if transformation_recs.status is _FE_COMPLETED and transformation_recs.recommendations:
                work, built_calendar, cal_notes = build_calendar_features(
                    work, transformation_recs.recommendations
                )
                notes.extend(cal_notes)
        except ValueError as exc:
            return _unavailable(
                "forecasting features could not be built: " + _normalise_error(str(exc)),
                objective_used=objective_used,
            )

        built_all = built_temporal + built_calendar
        if built_all:
            before = len(work)
            work = work.dropna(subset=built_all).reset_index(drop=True)
            rows_consumed_as_history = before - len(work)
            temporal_features_built = len(built_temporal)
            calendar_features_built = len(built_calendar)
            numeric_cols = sorted(set(numeric_cols) | set(built_all))
            feature_cols = sorted(set(feature_cols) | set(built_all))
            notes.append(
                f"{rows_consumed_as_history} leading / invalid-timestamp row(s) consumed as "
                "lag / rolling history"
            )
            notes.append(
                "primary evaluation is one-step-ahead: lag / rolling features use observed "
                "actuals; calendar features are stateless functions of the timestamp"
            )
            if forecast_horizon > 1:
                notes.append(
                    f"forecast_horizon = {forecast_horizon}: recursive rolling-origin "
                    "multi-step diagnostics (rmse_h2 …) are added per run; the calendar and "
                    "exogenous features are assumed known over the forecast window; the "
                    "selection metric is still the one-step rmse"
                )
        else:
            notes.append(
                "no Phase-6 lag / rolling / calendar recommendation referenced a column of "
                "the frame; trained as baseline regression on the eligible features"
            )
        if len(work) < MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS:
            return _unavailable(
                f"only {len(work)} modelable row(s) remain after consuming "
                f"{rows_consumed_as_history} row(s) as lag / rolling history; at least "
                f"{MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS} are required",
                objective_used=objective_used,
            )
    elif task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(
            "forecasting without a time-ordered holdout; trained as baseline regression on "
            "the currently-eligible features"
        )

    # canonicalise row order for the non-temporal strategies so the split
    # (and therefore every metric) is invariant to the input row order.
    # Sort by the model columns only (features + target) — orderable dtypes
    # by construction — never by excluded datetime / unknown / object
    # columns, whose comparability varies across pandas versions (audit M9).
    if split.strategy is not DataSplitStrategy.TIME_ORDERED_HOLDOUT and len(work) > 0:
        _feature_set = set(feature_cols)
        canonical_by = sorted(c for c in work.columns if c in _feature_set or c == target_column)
        if canonical_by:
            work = work.sort_values(
                by=canonical_by, kind="stable", ignore_index=True, na_position="last"
            )
        else:
            work = work.reset_index(drop=True)

    n = len(work)
    x_all = work[feature_cols] if feature_cols else work.iloc[:, :0]
    y_all = (
        work[target_column].to_numpy() if (is_supervised and target_column is not None) else None
    )

    stratify_y = y_all if (category == "classification" and y_all is not None) else None
    train_idx, val_idx, test_idx, split_notes = _split_indices(n, split, stratify_y)
    notes.extend(split_notes)

    target_temporal_spec: dict[str, tuple[str, str, int]] = {}
    if forecasting_run and target_column is not None and temporal.status is _FE_COMPLETED:
        target_temporal_spec = {
            name: meta
            for name, meta in temporal_feature_spec(temporal.recommendations).items()
            if meta[0] == target_column and name in feature_cols
        }

    runs: list[TrainingRun] = []
    for family in candidate_families:
        runs.append(
            _run_candidate(
                family=family,
                category=category,
                x_all=x_all,
                y_all=y_all,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                req_by_col=req_by_col,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                forecast_horizon=forecast_horizon if forecasting_run else 1,
                target_temporal_spec=target_temporal_spec,
            )
        )

    if forecasting_run:
        runs = [
            run.model_copy(
                update={
                    "temporal_features_built": temporal_features_built,
                    "calendar_features_built": calendar_features_built,
                    "rows_consumed_as_history": rows_consumed_as_history,
                    "forecast_horizon": forecast_horizon,
                }
            )
            for run in runs
        ]

    successful = [r.family.value for r in runs if r.status is TrainingRunStatus.COMPLETED]
    failed = [r.family.value for r in runs if r.status is not TrainingRunStatus.COMPLETED]

    if successful:
        reason = (
            None
            if not failed
            else f"{len(failed)} of {len(runs)} candidate(s) did not train: {', '.join(failed)}"
        )
    else:
        reason = f"all {len(runs)} candidate model family(ies) failed to train"

    return TrainingOutcome(
        status=ModelingStatus.COMPLETED,
        reason=reason,
        runs=runs,
        successful_runs=successful,
        failed_runs=failed,
        objective_used=objective_used,
        notes=notes,
    )


def _run_candidate(
    *,
    family: ModelFamily,
    category: str,
    x_all: pd.DataFrame,
    y_all: np.ndarray | None,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    req_by_col: dict[str, set[str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    forecast_horizon: int = 1,
    target_temporal_spec: dict[str, tuple[str, str, int]] | None = None,
) -> TrainingRun:
    built = _build_estimator(family, category)
    if built is None:
        return TrainingRun(
            family=family,
            estimator_name="(none)",
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=(
                f"no dependency-light baseline estimator is defined for the '{family.value}' "
                f"family on a {category} task"
            ),
        )
    estimator_name, estimator = built

    if not feature_cols:
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            test_rows=int(test_idx.size),
            reason="no usable numeric / categorical feature columns are available for training",
        )
    if (
        train_idx.size < MODEL_TRAINING_MIN_TRAIN_ROWS
        or test_idx.size < MODEL_TRAINING_MIN_TEST_ROWS
    ):
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=(
                f"insufficient rows after the split (train = {train_idx.size}, "
                f"test = {test_idx.size})"
            ),
        )

    preprocessor, preproc_error = _build_preprocessor(numeric_cols, categorical_cols, req_by_col)
    if preproc_error is not None:
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=preproc_error,
        )

    try:
        from sklearn.pipeline import Pipeline

        x_train = x_all.iloc[train_idx]
        x_test = x_all.iloc[test_idx]

        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        if category == "clustering":
            pipeline.fit(x_train)
            labels = np.asarray(pipeline.predict(x_test))
            transformed = pipeline.named_steps["preprocess"].transform(x_test)
            metrics = _clustering_metrics(np.asarray(transformed, dtype=float), labels)
            notes = (
                []
                if metrics
                else [
                    "fewer than 2 distinct clusters were assigned; no clustering metric is defined"
                ]
            )
        else:
            assert y_all is not None
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            pipeline.fit(x_train, y_train)
            y_pred = np.asarray(pipeline.predict(x_test))
            if category == "regression":
                metrics = _regression_metrics(y_test.astype(float), y_pred.astype(float))
                notes = []
                if (
                    forecast_horizon > 1
                    and target_temporal_spec
                    and int(test_idx.size) > forecast_horizon
                ):
                    metrics = {
                        **metrics,
                        **_recursive_horizon_metrics(
                            pipeline,
                            x_all,
                            y_all,
                            feature_cols,
                            target_temporal_spec,
                            test_start=int(test_idx[0]),
                            n_rows=int(x_all.shape[0]),
                            horizon=forecast_horizon,
                        ),
                    }
                    notes.append(
                        f"recursive rolling-origin multi-step diagnostics (rmse_h2 … "
                        f"rmse_h{forecast_horizon}); the selection metric is still the "
                        "one-step rmse"
                    )
            else:
                proba = None
                model = pipeline.named_steps["model"]
                if hasattr(model, "predict_proba"):
                    try:
                        proba_full = np.asarray(pipeline.predict_proba(x_test))
                        if proba_full.ndim == 2 and proba_full.shape[1] == 2:
                            proba = proba_full[:, 1]
                    except (ValueError, AttributeError):
                        proba = None
                n_classes = len(np.unique(y_train))
                metrics = _classification_metrics(y_test, y_pred, proba, n_classes)
                notes = []
    except Exception as exc:  # noqa: BLE001 - deterministic, normalised failure record
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.FAILED,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=f"{type(exc).__name__}: {_normalise_error(str(exc))}",
        )

    return TrainingRun(
        family=family,
        estimator_name=estimator_name,
        status=TrainingRunStatus.COMPLETED,
        train_rows=int(train_idx.size),
        validation_rows=int(val_idx.size),
        test_rows=int(test_idx.size),
        metrics=metrics,
        reason=None,
        notes=notes,
    )
