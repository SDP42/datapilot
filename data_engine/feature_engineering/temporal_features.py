"""Forecasting foundation — deterministic temporal-feature recommendations.

:func:`recommend_temporal_features` identifies the **lag** and **rolling-window**
features a time-series-forecasting problem structurally needs — autoregressive lags
of the forecasting target, rolling summaries of its recent history, and lag features
for numeric exogenous columns.

It **recommends only**. It never computes a lag, never calls ``.shift`` / ``.rolling``,
never modifies the DataFrame, never selects features, never trains a model, and never
infers the task type or the time column (both come from the Phase-5.3
:class:`TaskTypeInference`). Building the recommended features is a later increment. A
recommendation never means *"this will improve model performance"* — only *"a
forecasting problem structurally needs this class of feature"*.

``status = unavailable`` for **every** non-forecasting task — that is the expected
result, not an error.

Analysis-only: ``df``, ``inventory``, and ``task_type`` are never mutated; no file,
figure, lineage, version, database, network, or LLM access.
"""

from __future__ import annotations

import re

import pandas as pd

from data_engine.problem_understanding import (
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)
from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    TemporalFeatureRecommendation,
    TemporalFeatureRecommendations,
)

# --- tunables (documented in docs/feature-engineering.md) ------------------

# Lag orders recommended for the forecasting target (and, on request, for exogenous
# features). A fixed deterministic set — no frequency inference.
FORECASTING_LAG_ORDERS: tuple[int, ...] = (1, 2, 3, 7, 14)
# Rolling-window sizes recommended for the forecasting target (mean + std each).
FORECASTING_ROLLING_WINDOWS: tuple[int, ...] = (7, 30)
# Below this row count no temporal feature is recommended (too little history).
FORECASTING_MIN_ROWS_FOR_TEMPORAL: int = 50
# A lag order / window is only recommended when n_rows exceeds it by at least this.
FORECASTING_TEMPORAL_ROW_MARGIN: int = 10

_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED

_SEPARATORS = re.compile(r"[\s_\-/]+")
_LAG_INTENT_TOKENS = frozenset({"lag", "lags", "lagged", "autoregressive", "autoregression"})
_LAG_INTENT_PHRASES = ("past values", "previous values", "moving average", "rolling window")

_OP_RANK = {
    FeatureOperationType.LAG_FEATURE: 0,
    FeatureOperationType.ROLLING_FEATURE: 1,
}

_NOTE_RECOMMENDATION_ONLY = (
    "recommendation only — Phase 6 does not build lag / rolling features; execution is a "
    "later increment and the DataFrame is unchanged"
)
_NOTE_TARGET_EXEMPT = (
    "the forecasting target's own lag / rolling features are legitimate autoregressive "
    "predictors and are recorded here rather than blocked by the target-safety rule that "
    "applies to Phase 6.3 / 6.4 / 6.5"
)


def _has_lag_intent(objective: str) -> bool:
    normalized = _SEPARATORS.sub(" ", objective.strip().lower()).strip()
    padded = f" {normalized} "
    tokens = frozenset(t for t in normalized.split() if t)
    return bool(tokens & _LAG_INTENT_TOKENS) or any(p in padded for p in _LAG_INTENT_PHRASES)


def _unavailable(reason: str, *, objective_used: bool) -> TemporalFeatureRecommendations:
    return TemporalFeatureRecommendations(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        time_column=None,
        target_column=None,
        recommended_operations=[],
        recommendations=[],
        objective_used=objective_used,
        notes=[],
    )


def _sort_key(rec: TemporalFeatureRecommendation) -> tuple[str, int, int, int]:
    n = int(rec.description.split()[-1])  # trailing lag order / rolling window
    stat_rank = 0 if "mean" in rec.description else 1
    return (rec.column, _OP_RANK[rec.operation], n, stat_rank)


def recommend_temporal_features(
    df: pd.DataFrame,
    inventory: FeatureInventory,
    task_type: TaskTypeInference,
    *,
    objective: str | None = None,
) -> TemporalFeatureRecommendations:
    """Deterministically recommend lag / rolling features for a forecasting problem.

    Parameters
    ----------
    df:
        The dataset. **Not mutated / never computed on.** A non-DataFrame raises
        ``TypeError``.
    inventory:
        The **Phase-6.2** :class:`FeatureInventory` — the authority for which
        columns are candidate features. A non-model raises ``TypeError``.
    task_type:
        The **Phase-5.3** :class:`TaskTypeInference` — the sole authority for the
        task type, the target column, and the forecasting time column. A non-model
        raises ``TypeError``; it is **not mutated** and never re-inferred.
    objective:
        The user's objective, **verbatim and optional** — matched against a small
        fixed vocabulary to widen exogenous lag recommendations only. It never adds
        or removes a column and never overrides a structural rule.

    Returns
    -------
    TemporalFeatureRecommendations
        ``status = completed`` with structured ``recommendations`` and the aligned
        ``recommended_operations`` (both deterministically ordered) for a
        forecasting task with a resolved time column and a completed inventory; a
        completed result with empty lists + an explicit ``reason`` when the frame is
        too small; ``status = unavailable`` for every non-forecasting task, an
        unresolved time column, an incomplete task inference, or an incomplete
        inventory.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recommend_temporal_features expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(inventory, FeatureInventory):
        raise TypeError(
            "recommend_temporal_features expects a FeatureInventory, "
            f"got {type(inventory).__name__}"
        )
    if not isinstance(task_type, TaskTypeInference):
        raise TypeError(
            "recommend_temporal_features expects a TaskTypeInference, "
            f"got {type(task_type).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    # --- fixed unavailable precedence ---------------------------------
    if task_type.status is not _PU_COMPLETED:
        return _unavailable(
            f"task-type inference is not completed (status = {task_type.status.value})",
            objective_used=objective_used,
        )
    if task_type.task_type is not TaskType.TIME_SERIES_FORECASTING:
        task_name = task_type.task_type.value if task_type.task_type is not None else "none"
        return _unavailable(
            "temporal feature recommendations apply only to time_series_forecasting "
            f"(task type = {task_name})",
            objective_used=objective_used,
        )
    if task_type.time_column is None:
        return _unavailable(
            "no time column was resolved for the forecasting task",
            objective_used=objective_used,
        )
    if inventory.status is not _FE_COMPLETED:
        return _unavailable(
            f"the feature inventory is not completed (status = {inventory.status.value})",
            objective_used=objective_used,
        )

    time_col = task_type.time_column
    target_col = task_type.target_column
    column_names = {str(c) for c in df.columns}
    n_rows = len(df)
    lag_intent = objective_used and objective is not None and _has_lag_intent(objective)

    if n_rows < FORECASTING_MIN_ROWS_FOR_TEMPORAL:
        return TemporalFeatureRecommendations(
            status=FeatureEngineeringStatus.COMPLETED,
            reason=(
                f"only {n_rows} row(s); at least {FORECASTING_MIN_ROWS_FOR_TEMPORAL} are needed "
                "before lag / rolling feature recommendations are meaningful"
            ),
            time_column=time_col,
            target_column=target_col,
            recommended_operations=[],
            recommendations=[],
            objective_used=objective_used,
            notes=[_NOTE_RECOMMENDATION_ONLY],
        )

    recs: list[TemporalFeatureRecommendation] = []
    margin = FORECASTING_TEMPORAL_ROW_MARGIN
    target_is_time = target_col is not None and target_col == time_col
    target_usable = target_col is not None and target_col in column_names and not target_is_time

    # --- target autoregression ---------------------------------------
    if target_usable and target_col is not None:
        for order in FORECASTING_LAG_ORDERS:
            if n_rows > order + margin:
                recs.append(
                    TemporalFeatureRecommendation(
                        column=target_col,
                        operation=FeatureOperationType.LAG_FEATURE,
                        description=f"lag {order}",
                        reason="autoregressive lag of the forecasting target",
                        evidence=[
                            f"{n_rows} rows > lag {order} + margin {margin}",
                            "the target's own past values are the primary predictor for forecasting",
                        ],
                    )
                )
        for window in FORECASTING_ROLLING_WINDOWS:
            if n_rows > window + margin:
                for stat in ("mean", "std"):
                    recs.append(
                        TemporalFeatureRecommendation(
                            column=target_col,
                            operation=FeatureOperationType.ROLLING_FEATURE,
                            description=f"rolling {stat} window {window}",
                            reason="rolling summary of the forecasting target's recent history",
                            evidence=[f"{n_rows} rows > window {window} + margin {margin}"],
                        )
                    )

    # --- exogenous numeric lags -------------------------------------
    exogenous = sorted(
        c.column
        for c in inventory.candidates
        if c.candidate
        and not c.is_target
        and c.column_type is ColumnType.NUMERIC
        and c.column != time_col
        and c.column in column_names
    )
    exo_orders = FORECASTING_LAG_ORDERS if lag_intent else (1,)
    for col in exogenous:
        for order in exo_orders:
            if n_rows > order + margin:
                recs.append(
                    TemporalFeatureRecommendation(
                        column=col,
                        operation=FeatureOperationType.LAG_FEATURE,
                        description=f"lag {order}",
                        reason="lag of a numeric exogenous feature for forecasting",
                        evidence=[f"{n_rows} rows > lag {order} + margin {margin}"]
                        + (
                            ["objective favours autoregressive / lag features"]
                            if lag_intent
                            else []
                        ),
                    )
                )

    recs.sort(key=_sort_key)
    recommended_operations = [f"{r.column}: {r.description}" for r in recs]
    n_target_recs = sum(1 for r in recs if r.column == target_col and not target_is_time)

    lead = (
        f"{len(recs)} temporal feature recommendation(s) for forecasting target "
        f"'{target_col}' on time axis '{time_col}'"
    )
    if target_is_time:
        lead += " (no target autoregression — the target is itself the time column)"
    notes: list[str] = [lead, _NOTE_RECOMMENDATION_ONLY]
    if n_target_recs:
        notes.append(_NOTE_TARGET_EXEMPT)
    if objective_used:
        notes.append(
            "objective favours autoregressive / lag features; exogenous lag recommendations "
            "were widened to the full lag set"
            if lag_intent
            else "objective recorded; no lag / autoregressive vocabulary matched"
        )

    return TemporalFeatureRecommendations(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None,
        time_column=time_col,
        target_column=target_col,
        recommended_operations=recommended_operations,
        recommendations=recs,
        objective_used=objective_used,
        notes=notes,
    )
