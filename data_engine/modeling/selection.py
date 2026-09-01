"""Phase 7.5 — deterministic model selection & recommendation.

:func:`select_model` picks a recommended model family / estimator by
comparing the evaluation metrics **already recorded** by Phase 7.4
(``TrainingOutcome.runs[*].metrics``). It is selection only:

* it retrains nothing and fits no estimator;
* it recomputes no metric;
* it executes no preprocessing, feature engineering, or data split;
* it modifies no upstream object;
* it tunes no hyperparameters, runs no cross-validation / grid search,
  computes no feature importance / SHAP / correlation, does no leakage
  detection or statistical significance testing;
* it introduces no new model family and persists no artifact.

The winner is exactly the first entry in a fully deterministic ranking:
by the fixed per-task selection metric, then the fixed Phase-7.3 family
ordering, then the estimator name.
"""

from __future__ import annotations

import math

from data_engine.feature_engineering import FeatureEngineeringSpec, FeatureEngineeringStatus
from data_engine.problem_understanding import ProblemSpec, ProblemUnderstandingStatus, TaskType

from .models import (
    DataSplitPlan,
    ModelCandidates,
    ModelFamily,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    ModelSelectionRank,
    TrainingOutcome,
    TrainingRunStatus,
)

# --- fixed, documented rules -----------------------------------------------

# (metric name, direction) per supported task type. The metric is the
# existing Phase-7.4 metric — never a substitute and never a composite.
_TASK_SELECTION_METRIC: dict[TaskType, tuple[str, str]] = {
    TaskType.REGRESSION: ("rmse", "minimize"),
    TaskType.TIME_SERIES_FORECASTING: ("rmse", "minimize"),
    TaskType.BINARY_CLASSIFICATION: ("f1", "maximize"),
    TaskType.MULTICLASS_CLASSIFICATION: ("f1", "maximize"),
    TaskType.CLUSTERING: ("silhouette_score", "maximize"),
}

# the Phase-7.3 family ranking, used only as a deterministic tie-break
_FAMILY_ORDER: dict[str, int] = {
    ModelFamily.LINEAR.value: 0,
    ModelFamily.TREE_BASED.value: 1,
    ModelFamily.ENSEMBLE.value: 2,
    ModelFamily.PROBABILISTIC.value: 3,
    ModelFamily.DISTANCE_BASED.value: 4,
    ModelFamily.NEURAL.value: 5,
}

_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})
_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED

_NOTE_SELECTION_ONLY = (
    "model selection is based only on the Phase 7.4 evaluation results; no model was "
    "retrained, no prediction generated, no metric recomputed, no hyperparameter tuned, no "
    "preprocessing or feature engineering executed, and the DataFrame was not modified"
)
_NOTE_NO_ARTIFACT = "no final estimator artifact was persisted"


def _unavailable(reason: str, *, objective_used: bool) -> ModelSelection:
    return ModelSelection(
        status=ModelingStatus.UNAVAILABLE,
        reason=reason,
        selected_family=None,
        selected_estimator=None,
        selection_metric=None,
        selection_direction=None,
        selected_score=None,
        ranking=[],
        objective_used=objective_used,
        notes=[],
    )


def _family_rank(family: str) -> int:
    return _FAMILY_ORDER.get(family, len(_FAMILY_ORDER))


def select_model(
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    candidates: ModelCandidates,
    training: TrainingOutcome,
    *,
    objective: str | None = None,
) -> ModelSelection:
    """Deterministically recommend a model from the Phase-7.4 training runs.

    Parameters
    ----------
    problem / feature_engineering / readiness / split / candidates / training:
        The existing Phase-5 / Phase-6 / Phase-7.2–7.4 contracts. A
        non-model for any of them raises ``TypeError``; none is mutated.
    objective:
        The user's objective, **verbatim and optional** — recorded in a
        note only; it never overrides the fixed metric-selection rules.

    Returns
    -------
    ModelSelection
        ``status = completed`` once every prerequisite is completed —
        with a recommendation when at least one training run carries the
        task's selection metric, otherwise with ``selected_* = None`` and
        an explicit reason. ``status = unavailable`` when a prerequisite
        is not usable (fixed precedence).
    """
    if not isinstance(problem, ProblemSpec):
        raise TypeError(f"select_model expects a ProblemSpec, got {type(problem).__name__}")
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            f"select_model expects a FeatureEngineeringSpec, got {type(feature_engineering).__name__}"
        )
    if not isinstance(readiness, ModelReadiness):
        raise TypeError(f"select_model expects a ModelReadiness, got {type(readiness).__name__}")
    if not isinstance(split, DataSplitPlan):
        raise TypeError(f"select_model expects a DataSplitPlan, got {type(split).__name__}")
    if not isinstance(candidates, ModelCandidates):
        raise TypeError(f"select_model expects a ModelCandidates, got {type(candidates).__name__}")
    if not isinstance(training, TrainingOutcome):
        raise TypeError(f"select_model expects a TrainingOutcome, got {type(training).__name__}")

    objective_used = objective is not None and objective.strip() != ""

    # --- fixed upstream precedence --------------------------------------
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
            f"model selection does not support task type '{task.value}'",
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
            f"model selection is blocked by model-readiness issues: {first}",
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
    if training.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"model training is not completed (status = {training.status.value})",
            objective_used=objective_used,
        )
    if feature_engineering.assessment.status is not _FE_COMPLETED:
        return _unavailable(
            "feature-engineering assessment is not completed "
            f"(status = {feature_engineering.assessment.status.value})",
            objective_used=objective_used,
        )

    metric, direction = _TASK_SELECTION_METRIC[task]
    known_families = set(candidates.candidates)

    notes: list[str] = [
        _NOTE_SELECTION_ONLY,
        _NOTE_NO_ARTIFACT,
        f"task type: {task.value}",
        f"selection metric for task '{task.value}': {metric} ({direction})",
    ]
    if task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(
            "Phase 5 supplied the time-series-forecasting task; Phase 7.5 does not infer "
            "forecasting from datetime columns, generates no lag or rolling features, and "
            "selects only among the existing Phase-7.4 baseline regression runs"
        )
    if objective_used:
        notes.append(
            "an objective was supplied and recorded; it did not change the fixed "
            "metric-selection rules"
        )

    if not training.runs:
        return ModelSelection(
            status=ModelingStatus.COMPLETED,
            reason="no model training runs are available for selection",
            selected_family=None,
            selected_estimator=None,
            selection_metric=metric,
            selection_direction=direction,
            selected_score=None,
            ranking=[],
            objective_used=objective_used,
            notes=notes,
        )

    # --- classify every run --------------------------------------------
    eligible: list[tuple[str, str, float]] = []  # (family, estimator, score)
    ineligible: list[ModelSelectionRank] = []
    for run in training.runs:
        family = run.family.value
        estimator = run.estimator_name
        run_status = run.status.value
        if run.status is not TrainingRunStatus.COMPLETED:
            detail = f"training run is {run_status}"
            if run.reason:
                detail += f": {run.reason}"
            ineligible.append(
                ModelSelectionRank(
                    family=family,
                    estimator_name=estimator,
                    status=run_status,
                    score=None,
                    metric=None,
                    rank=None,
                    reason=detail,
                )
            )
            continue
        if family not in known_families:
            ineligible.append(
                ModelSelectionRank(
                    family=family,
                    estimator_name=estimator,
                    status=run_status,
                    score=None,
                    metric=None,
                    rank=None,
                    reason=(
                        f"training run references model family '{family}', which is not a "
                        "Phase 7.3 candidate; it is not selectable"
                    ),
                )
            )
            continue
        value = run.metrics.get(metric)
        if value is None:
            ineligible.append(
                ModelSelectionRank(
                    family=family,
                    estimator_name=estimator,
                    status=run_status,
                    score=None,
                    metric=metric,
                    rank=None,
                    reason=f"selection metric '{metric}' is unavailable for this run",
                )
            )
            continue
        if not math.isfinite(value):
            ineligible.append(
                ModelSelectionRank(
                    family=family,
                    estimator_name=estimator,
                    status=run_status,
                    score=None,
                    metric=metric,
                    rank=None,
                    reason=f"selection metric '{metric}' is not a finite value for this run",
                )
            )
            continue
        eligible.append((family, estimator, float(value)))

    def _sort_key(item: tuple[str, str, float]) -> tuple[float, int, str]:
        family, estimator, score = item
        primary = score if direction == "minimize" else -score
        return (primary, _family_rank(family), estimator)

    eligible.sort(key=_sort_key)

    ranked: list[ModelSelectionRank] = []
    for position, (family, estimator, score) in enumerate(eligible, start=1):
        ranked.append(
            ModelSelectionRank(
                family=family,
                estimator_name=estimator,
                status=TrainingRunStatus.COMPLETED.value,
                score=score,
                metric=metric,
                rank=position,
                reason=(
                    f"eligible: {metric} = {score} ({direction}) from the Phase-7.4 test partition"
                ),
            )
        )

    ineligible.sort(key=lambda r: (_family_rank(r.family), r.estimator_name))
    ranking = ranked + ineligible

    if not eligible:
        return ModelSelection(
            status=ModelingStatus.COMPLETED,
            reason=f"no completed training run had a usable '{metric}' selection metric",
            selected_family=None,
            selected_estimator=None,
            selection_metric=metric,
            selection_direction=direction,
            selected_score=None,
            ranking=ranking,
            objective_used=objective_used,
            notes=notes,
        )

    winner_family, winner_estimator, winner_score = eligible[0]
    tied = sum(1 for _, _, s in eligible if s == winner_score)
    notes.append(
        f"selected '{winner_family}' / '{winner_estimator}' as the {direction} of '{metric}' "
        f"({metric} = {winner_score}) among {len(eligible)} eligible run(s)"
    )
    if tied > 1:
        notes.append(
            f"multiple eligible runs ({tied}) tied on {metric}; the deterministic family / "
            "estimator ordering was used as the tie-break — the tie is an ordering choice, "
            "not a claim that either model performs better"
        )

    return ModelSelection(
        status=ModelingStatus.COMPLETED,
        reason=None,
        selected_family=winner_family,
        selected_estimator=winner_estimator,
        selection_metric=metric,
        selection_direction=direction,
        selected_score=winner_score,
        ranking=ranking,
        objective_used=objective_used,
        notes=notes,
    )
