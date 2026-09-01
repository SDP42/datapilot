"""Phase 5.5 — deterministic, rule-based feasibility assessment.

:func:`assess_feasibility` answers one narrow question: *is this
problem structurally well-formed and data-supported enough to proceed to
later ML phases?* It **consumes** the Phase-5.2 / 5.3 / 5.4 results
(target, task type, candidate metrics) and never re-runs or overrides
them.

It is a **structural feasibility screen**, not a guarantee of model
performance: it looks only at row counts, target availability / variation,
class balance, finite-value counts, timestamp availability, and whether
any non-target input data exists. It performs **no** model training,
prediction, cross-validation, statistical testing, feature selection,
cleaning, imputation, or leakage detection.

Conservative: when an upstream Phase-5 result is unavailable (or no single
target was identified for a supervised task) the assessment is
``status = unavailable`` / ``feasible = None`` — never a fabricated
``False``.

Analysis-only: it never mutates ``df`` or any upstream result, writes a
file, creates a figure, or touches lineage / versions / a database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    CandidateMetrics,
    FeasibilityAssessment,
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
)

# --- deterministic thresholds (documented in docs/problem-understanding.md) ---

MIN_ROWS_HARD = 2  # fewer than this -> blocking issue
MIN_ROWS_WARNING = 20  # fewer than this (but >= MIN_ROWS_HARD) -> warning
TARGET_MISSING_WARNING = 0.20  # target missing-fraction above this -> warning
SEVERE_CLASS_IMBALANCE = 0.05  # smallest class below this share -> warning

_SUPERVISED_TASKS = frozenset(
    {
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.TIME_SERIES_FORECASTING,
    }
)
_CLASSIFICATION_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)

_STRUCTURAL_SCREEN_NOTE = (
    "this is a deterministic structural feasibility screen; a feasible result "
    "does not guarantee model performance"
)
_LEAKAGE_NOTE = "feature-target leakage has not been assessed at this stage"


# --- result helpers --------------------------------------------------------


def _unavailable(reason: str) -> FeasibilityAssessment:
    return FeasibilityAssessment(
        status=ProblemUnderstandingStatus.UNAVAILABLE,
        reason=reason,
        feasible=None,
        blocking_issues=[],
        warnings=[],
        notes=[],
    )


def _upstream_reason(
    name: str, result: TargetIdentification | TaskTypeInference | CandidateMetrics
) -> str | None:
    """Return a reason string if `result` is not usable, else None."""
    if result.status is ProblemUnderstandingStatus.UNAVAILABLE:
        return f"{name} is unavailable ({result.reason or 'no reason given'})"
    if result.status is not ProblemUnderstandingStatus.COMPLETED:
        return f"{name} has not been performed (status = {result.status.value})"
    return None


# --- public API ------------------------------------------------------------


def assess_feasibility(
    df: pd.DataFrame,
    target: TargetIdentification,
    task_type: TaskTypeInference,
    metrics: CandidateMetrics,
    *,
    objective: str | None = None,
) -> FeasibilityAssessment:
    """Deterministically screen a problem's structural feasibility.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    target:
        The **Phase-5.2** :class:`TargetIdentification` — authoritative for
        which column is the target. Not mutated; never re-selected.
    task_type:
        The **Phase-5.3** :class:`TaskTypeInference` — authoritative for
        the task type. Not mutated; never re-inferred.
    metrics:
        The **Phase-5.4** :class:`CandidateMetrics`. Not mutated.
    objective:
        The user's objective, **verbatim and optional** — recorded in the
        notes only; it never overrides an upstream decision.

    Returns
    -------
    FeasibilityAssessment
        ``status = completed`` with ``feasible = True`` (no blocking
        issues) or ``feasible = False`` (>= 1 blocking issue); or
        ``status = unavailable`` / ``feasible = None`` when an upstream
        Phase-5 result is unavailable or no single target was identified
        for a supervised task. ``warnings`` never flip ``feasible`` to
        ``False``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"assess_feasibility expects a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(target, TargetIdentification):
        raise TypeError(
            f"assess_feasibility expects a TargetIdentification, got {type(target).__name__}"
        )
    if not isinstance(task_type, TaskTypeInference):
        raise TypeError(
            f"assess_feasibility expects a TaskTypeInference, got {type(task_type).__name__}"
        )
    if not isinstance(metrics, CandidateMetrics):
        raise TypeError(
            f"assess_feasibility expects a CandidateMetrics, got {type(metrics).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    # --- respect the upstream Phase-5 pipeline ----------------------------
    for name, result in (
        ("target identification", target),
        ("task-type inference", task_type),
        ("candidate-metric recommendation", metrics),
    ):
        upstream = _upstream_reason(name, result)
        if upstream is not None:
            return _unavailable(f"cannot assess feasibility: {upstream}")

    task = task_type.task_type
    if task is None:
        return _unavailable(
            "cannot assess feasibility: task-type inference completed without a task type"
        )
    if task not in _SUPERVISED_TASKS and task is not TaskType.CLUSTERING:
        return _unavailable(f"feasibility assessment does not support task type '{task.value}'")

    target_column = target.target_column
    is_supervised = task in _SUPERVISED_TASKS
    if is_supervised and target_column is None:
        return _unavailable(
            "cannot assess feasibility: no single target column was identified for a "
            f"supervised ({task.value}) task"
        )

    column_names = [str(c) for c in df.columns]

    def column(name: str) -> pd.Series:
        return df.iloc[:, column_names.index(name)]

    n_rows = len(df)
    n_cols = df.shape[1]

    blocking: list[str] = []
    warnings: list[str] = []
    notes: list[str] = [
        (
            f"task '{task.value}', "
            + (f"target '{target_column}'" if target_column is not None else "targetless")
            + f"; {n_rows} row(s), {n_cols} column(s)"
        ),
        "objective supplied (used for notes only, not to override upstream decisions)"
        if objective_used
        else "no objective supplied",
        _STRUCTURAL_SCREEN_NOTE,
        _LEAKAGE_NOTE,
    ]

    # --- A. dataset size (all tasks) ------------------------------------
    if n_rows < MIN_ROWS_HARD:
        blocking.append(f"the dataset has {n_rows} row(s); at least {MIN_ROWS_HARD} are required")
    elif n_rows < MIN_ROWS_WARNING:
        warnings.append(
            f"the dataset has only {n_rows} rows (fewer than {MIN_ROWS_WARNING}); "
            "estimates from this little data may be unreliable"
        )

    # --- B/C/D. supervised target checks ------------------------------
    if is_supervised and target_column is not None:
        if target_column not in column_names:
            blocking.append(
                f"the identified target column '{target_column}' is not in the DataFrame"
            )
        else:
            target_series = column(target_column)
            non_null = target_series.dropna()
            n_non_missing = int(non_null.shape[0])
            n_missing = int(target_series.shape[0] - n_non_missing)
            missing_fraction = n_missing / target_series.shape[0] if target_series.shape[0] else 0.0
            notes.append(
                f"target '{target_column}': {n_non_missing} usable / {n_missing} missing "
                f"observation(s), {int(non_null.nunique())} distinct value(s)"
            )

            if n_non_missing == 0:
                blocking.append(
                    f"the target column '{target_column}' has no usable (non-missing) observations"
                )
            else:
                if int(non_null.nunique()) <= 1:
                    blocking.append(
                        f"the target column '{target_column}' is constant "
                        "(1 distinct value); there is nothing to predict"
                    )
                if missing_fraction > TARGET_MISSING_WARNING:
                    warnings.append(
                        f"{missing_fraction:.1%} of target '{target_column}' values are "
                        f"missing (above {TARGET_MISSING_WARNING:.0%}); the usable sample "
                        "is smaller than the row count"
                    )

                # --- D. regression -------------------------------------
                if task is TaskType.REGRESSION:
                    numeric = pd.to_numeric(target_series, errors="coerce").to_numpy(dtype=float)
                    finite = numeric[np.isfinite(numeric)]
                    n_finite = int(finite.size)
                    n_non_finite = n_non_missing - n_finite
                    if n_non_finite > 0:
                        notes.append(
                            f"regression target '{target_column}': {n_non_finite} non-finite "
                            "value(s) treated as unusable"
                        )
                    if n_finite < 2:
                        blocking.append(
                            f"the regression target '{target_column}' has {n_finite} finite "
                            "numeric observation(s); at least 2 are required"
                        )
                    elif np.unique(finite).size <= 1:
                        blocking.append(
                            f"the regression target '{target_column}' has a single distinct "
                            "finite value; there is nothing to predict"
                        )

                # --- C. classification -------------------------------
                if task in _CLASSIFICATION_TASKS:
                    class_counts = non_null.value_counts()
                    n_observed = int(class_counts.shape[0])
                    if n_observed < 2:
                        blocking.append(
                            f"the classification target '{target_column}' has {n_observed} "
                            "observed class(es); at least 2 are required"
                        )
                    else:
                        smallest_share = float(class_counts.min() / n_non_missing)
                        notes.append(
                            f"classification target '{target_column}': {n_observed} observed "
                            f"class(es), smallest is {smallest_share:.1%} of usable observations"
                        )
                        if smallest_share < SEVERE_CLASS_IMBALANCE:
                            warnings.append(
                                f"the smallest class in target '{target_column}' is "
                                f"{smallest_share:.1%} of usable observations (below "
                                f"{SEVERE_CLASS_IMBALANCE:.0%}); severe class imbalance"
                            )

    # --- E. forecasting: datetime availability ------------------------
    if task is TaskType.TIME_SERIES_FORECASTING:
        datetime_columns = sorted(
            name for name in column_names if infer_column_type(column(name)) is ColumnType.DATETIME
        )
        if not datetime_columns:
            blocking.append(
                "time-series forecasting requires at least one datetime column; none is present"
            )
        else:
            dt_name = (
                target_column
                if target_column is not None and target_column in datetime_columns
                else datetime_columns[0]
            )
            timestamps = pd.to_datetime(column(dt_name), errors="coerce").dropna()
            n_timestamps = int(timestamps.shape[0])
            n_distinct = int(timestamps.nunique())
            notes.append(
                f"forecasting datetime column '{dt_name}': {n_timestamps} usable, "
                f"{n_distinct} distinct timestamp(s)"
            )
            if n_timestamps < 2:
                blocking.append(
                    f"the datetime column '{dt_name}' has {n_timestamps} usable timestamp(s); "
                    "at least 2 are required"
                )
            elif n_distinct <= 1:
                blocking.append(
                    f"the datetime column '{dt_name}' has a single distinct timestamp; "
                    "there is no temporal variation to forecast over"
                )

    # --- feature availability (supervised) ---------------------------
    if is_supervised:
        non_target_columns = [name for name in column_names if name != target_column]
        if not non_target_columns:
            blocking.append(
                "the DataFrame contains only the target column; there is no input feature data"
            )
        elif all(column(name).dropna().empty for name in non_target_columns):
            blocking.append(
                "every non-target column is entirely missing; there is no usable input feature data"
            )

    # --- F. clustering feasibility ---------------------------------
    if task is TaskType.CLUSTERING:
        usable_features = [
            name for name in sorted(column_names) if int(column(name).dropna().nunique()) >= 2
        ]
        if not usable_features:
            blocking.append(
                "no column has at least two distinct non-missing values; clustering has no "
                "usable feature variation"
            )
        else:
            notes.append(
                f"clustering: {len(usable_features)} column(s) have usable variation "
                "(>= 2 distinct non-missing values)"
            )

    # --- final feasibility decision -------------------------------
    if blocking:
        return FeasibilityAssessment(
            status=ProblemUnderstandingStatus.COMPLETED,
            reason=(
                f"{len(blocking)} blocking issue(s) make the problem structurally unsuitable "
                f"to proceed: {blocking[0]}"
            ),
            feasible=False,
            blocking_issues=blocking,
            warnings=warnings,
            notes=notes,
        )
    return FeasibilityAssessment(
        status=ProblemUnderstandingStatus.COMPLETED,
        reason=None,
        feasible=True,
        blocking_issues=[],
        warnings=warnings,
        notes=notes,
    )
