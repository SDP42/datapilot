"""Phase 7.2 — deterministic data-split *planning*.

:func:`recommend_data_split` makes a transparent, rule-based
recommendation for the eventual train / validation / test split, from the
Phase-5 task type, the Phase-6 eligible features, and the structural shape
of the supplied DataFrame.

It **recommends only**. It never shuffles, stratifies, orders, or copies
the DataFrame; never creates train/test datasets; never persists
anything; never trains a model, fits an estimator, generates a
prediction, computes a metric, creates lag / rolling features, or infers
a forecasting task from a datetime column.
"""

from __future__ import annotations

import pandas as pd

from data_engine.feature_engineering import FeatureEngineeringSpec
from data_engine.problem_understanding import (
    ProblemSpec,
    ProblemUnderstandingStatus,
    TaskType,
)

from .models import DataSplitPlan, DataSplitStrategy, ModelingStatus

# --- tunables (documented in docs/modeling.md) -------------------------

DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
SMALL_DATA_TRAIN_FRACTION = 0.8
SMALL_DATA_TEST_FRACTION = 0.2
# Below this row count no separate validation split is recommended.
MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION = 200
# Below this row count the split is flagged as unreliable (still recommended).
MODEL_SPLIT_MIN_ROWS = 20
# Each observed class must have at least this many members for stratification.
MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY = 2

_CLASSIFICATION_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)
_SUPERVISED_TASKS = _CLASSIFICATION_TASKS | frozenset(
    {TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING}
)
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})
_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED

_NOTE_RECOMMENDATION_ONLY = (
    "this is a split-strategy recommendation only — Phase 7.2 does not shuffle, stratify, "
    "order, copy, or split the DataFrame"
)


def _unavailable(reason: str) -> DataSplitPlan:
    return DataSplitPlan(status=ModelingStatus.UNAVAILABLE, reason=reason, notes=[])


def _fractions(n_rows: int) -> tuple[float, float | None, float, list[str]]:
    notes: list[str] = []
    if n_rows < MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION:
        notes.append(
            f"small dataset ({n_rows} rows, fewer than "
            f"{MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION}); recommending a train/test split only, "
            "with no separate validation split"
        )
        if n_rows < MODEL_SPLIT_MIN_ROWS:
            notes.append(
                f"very small dataset ({n_rows} rows); any split will be statistically "
                "unreliable — treat the recommendation with caution"
            )
        return SMALL_DATA_TRAIN_FRACTION, None, SMALL_DATA_TEST_FRACTION, notes
    return DEFAULT_TRAIN_FRACTION, DEFAULT_VALIDATION_FRACTION, DEFAULT_TEST_FRACTION, notes


def recommend_data_split(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    *,
    objective: str | None = None,
) -> DataSplitPlan:
    """Deterministically recommend a train / validation / test split.

    Parameters
    ----------
    df:
        The dataset. **Not mutated / not split.** A non-DataFrame raises
        ``TypeError``.
    problem:
        The **Phase-5** :class:`ProblemSpec` — the sole authority for the
        task type (never re-inferred). A non-model raises ``TypeError``.
    feature_engineering:
        The **Phase-6** :class:`FeatureEngineeringSpec`. A non-model raises
        ``TypeError``; it is **not mutated**.
    objective:
        The user's objective, **verbatim and optional** — recorded in the
        notes only; it never changes the recommendation.

    Returns
    -------
    DataSplitPlan
        ``status = completed`` with a recommended ``strategy`` and
        fractions; ``status = unavailable`` when the task type is not
        completed / absent / unsupported, or (for a supervised task) the
        target identification is not completed.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"recommend_data_split expects a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(problem, ProblemSpec):
        raise TypeError(f"recommend_data_split expects a ProblemSpec, got {type(problem).__name__}")
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            "recommend_data_split expects a FeatureEngineeringSpec, "
            f"got {type(feature_engineering).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED:
        return _unavailable(
            f"task-type inference is not completed (status = {task_inference.status.value})"
        )
    task = task_inference.task_type
    if task is None:
        return _unavailable("task-type inference completed without a task type")
    if task in _UNSUPPORTED_TASKS:
        return _unavailable(f"data-split planning does not support task type '{task.value}'")

    if task in _SUPERVISED_TASKS and problem.target.status is not _PU_COMPLETED:
        return _unavailable(
            f"target identification is not completed (status = {problem.target.status.value})"
        )

    n_rows = len(df)
    column_names = [str(c) for c in df.columns]
    train_fraction, validation_fraction, test_fraction, size_notes = _fractions(n_rows)

    notes: list[str] = [
        _NOTE_RECOMMENDATION_ONLY,
        f"task type: {task.value}",
        *size_notes,
    ]

    stratify = False
    preserve_temporal_order = False
    shuffle = True

    if task is TaskType.TIME_SERIES_FORECASTING:
        strategy = DataSplitStrategy.TIME_ORDERED_HOLDOUT
        preserve_temporal_order = True
        shuffle = False
        notes.append(
            "time-series forecasting: a chronological holdout is recommended — the earliest "
            "rows form the training set and the most recent rows form validation then test; "
            "rows must not be shuffled and the target must not be stratified"
        )
        notes.append(
            "Phase 7.2 does not infer a forecasting task from a datetime column, create lag "
            "or rolling features, or perform any forecasting"
        )
    elif task is TaskType.REGRESSION:
        strategy = DataSplitStrategy.RANDOM_HOLDOUT
        notes.append(
            "regression: a shuffled random holdout is recommended; stratification is not "
            "applied to a continuous target"
        )
    elif task in _CLASSIFICATION_TASKS:
        target_column = problem.target.target_column
        smallest_class = 0
        n_classes = 0
        if target_column is None:
            notes.append(
                "classification task but no target column was identified; recommending an "
                "un-stratified random holdout"
            )
            strategy = DataSplitStrategy.RANDOM_HOLDOUT
        elif target_column not in column_names:
            notes.append(
                f"the identified target column '{target_column}' is not in the DataFrame; "
                "recommending an un-stratified random holdout"
            )
            strategy = DataSplitStrategy.RANDOM_HOLDOUT
        else:
            classes = df.iloc[:, column_names.index(target_column)].dropna().value_counts()
            n_classes = int(classes.shape[0])
            smallest_class = int(classes.min()) if n_classes else 0
            if n_classes >= 2 and smallest_class >= MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY:
                strategy = DataSplitStrategy.STRATIFIED_HOLDOUT
                stratify = True
                notes.append(
                    f"classification: stratifying on the target '{target_column}' is "
                    f"recommended ({n_classes} classes, smallest has {smallest_class} members)"
                )
            else:
                strategy = DataSplitStrategy.RANDOM_HOLDOUT
                notes.append(
                    f"classification: the target '{target_column}' has {n_classes} observed "
                    f"class(es) with a smallest class of {smallest_class}; stratification is "
                    "not recommended (too few members per class) — using a random holdout"
                )
    elif task is TaskType.CLUSTERING:
        strategy = DataSplitStrategy.RANDOM_HOLDOUT
        notes.append(
            "clustering is targetless: a shuffled random holdout is recommended only for "
            "cluster-stability checks; no stratification and no temporal ordering"
        )
    else:  # pragma: no cover - every TaskType is handled above
        return _unavailable(f"data-split planning does not support task type '{task.value}'")

    if objective_used:
        notes.append(
            "an objective was supplied and recorded; it did not change the split recommendation"
        )

    return DataSplitPlan(
        status=ModelingStatus.COMPLETED,
        reason=None,
        strategy=strategy,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        stratify=stratify,
        preserve_temporal_order=preserve_temporal_order,
        shuffle=shuffle,
        notes=notes,
    )
