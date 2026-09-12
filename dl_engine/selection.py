"""Phase 8.6 — deterministic DL model selection & comparison.

:func:`select_dl_models` compares multiple explicit
:class:`~dl_engine.contracts.DLCandidate` configurations **against each
other only** — never against a Phase-7 classical model; that broader
comparison is out of scope here, as is model selection *within* the
Phase-7 pipeline (``run_modeling_pipeline`` and
:func:`data_engine.modeling.select_model` are untouched).

It is selection only, mirroring
:mod:`data_engine.modeling.selection`'s own boundary exactly::

* it retrains nothing — every candidate is executed **once**, via the
  existing :func:`dl_engine.execution.run_mlp_modeling`;
* it recomputes no metric — the selection metric is read from the
  candidate's own :class:`~dl_engine.contracts.DLEvaluationResult`;
* it introduces no new metric semantic — the (metric, direction) pair
  per task is the **exact** one :mod:`data_engine.modeling.selection`
  already established;
* it creates no experiment record and persists no artifact.

The winner is exactly the first entry in a fully deterministic ranking:
by the fixed per-task selection metric, then by architecture name, then
by the candidate's own serialised training configuration — see
:func:`select_dl_models` for the documented tie-break rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from importlib import import_module
from types import ModuleType

import numpy as np

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType

from .contracts import DLCandidate, DLCandidateRank, DLSelectionResult
from .execution import run_mlp_modeling

# The exact (metric, direction) pair data_engine.modeling.selection already
# established per task — reused verbatim, never a DL-specific substitute.
_TASK_SELECTION_METRIC: dict[TaskType, tuple[str, str]] = {
    TaskType.REGRESSION: ("rmse", "minimize"),
    TaskType.BINARY_CLASSIFICATION: ("f1", "maximize"),
    TaskType.MULTICLASS_CLASSIFICATION: ("f1", "maximize"),
}

_NOTE_SELECTION_ONLY = (
    "DL model selection is based only on each candidate's own run_mlp_modeling result; no "
    "candidate was retrained, no metric was recomputed, and this compares Phase-8 DL candidates "
    "against each other only — never against a Phase-7 classical model"
)


def _no_candidates(reason: str) -> DLSelectionResult:
    return DLSelectionResult(
        status=TrainingRunStatus.FAILED,
        task_type=TaskType.OTHER,
        selection_metric=None,
        selection_direction=None,
        ranking=[],
        reason=reason,
    )


def select_dl_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    candidates: list[DLCandidate],
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLSelectionResult:
    """Execute every candidate once via :func:`run_mlp_modeling` and rank the results.

    ``X_train`` / ``y_train`` / ``X_eval`` / ``y_eval`` are shared across
    every candidate — this compares configurations on the **same**
    train/evaluation split, exactly like
    :func:`data_engine.modeling.select_model` compares classical
    candidates on the same Phase-7.4 split. Every candidate must target
    the same ``task_type`` (on both its architecture and its training
    config) — a mismatch across the candidate set is reported as a
    structured failure, since no single selection metric could compare
    candidates from different tasks meaningfully.

    Eligibility
    -----------
    A candidate is eligible only when its ``run_mlp_modeling`` result is
    ``completed`` (both training and evaluation completed), the task's
    selection metric is present in its evaluation metrics, and that
    value is finite. An ineligible candidate remains visible in
    ``ranking`` with ``rank = None`` and a ``reason`` — it is never
    silently dropped.

    Deterministic ranking / tie-break
    ----------------------------------
    Eligible candidates are sorted by ``(metric value, oriented so lower
    is always better), architecture_name, the candidate's own
    training_config.model_dump_json()``. Architecture name and a fully
    serialised configuration are both intrinsic to each candidate and
    never change between runs, so this tie-break is reproducible by
    construction — never dictionary order, an object id, a timestamp, or
    a random number. Ineligible candidates are sorted the same way
    (without the metric) and appended after every eligible one.

    Never retrains: each candidate is executed **exactly once**, via one
    call to ``run_mlp_modeling``; ranking only reads the results already
    produced.
    """
    if not candidates:
        return _no_candidates("no candidates were supplied for selection")

    task_types = {c.architecture.task_type for c in candidates} | {
        c.training_config.task_type for c in candidates
    }
    if len(task_types) != 1:
        return _no_candidates(
            "all candidates must share the same task_type on both their architecture and "
            f"training_config; got: {sorted(t.value for t in task_types)}"
        )

    task_type = candidates[0].architecture.task_type
    if task_type not in _TASK_SELECTION_METRIC:
        return _no_candidates(f"DL model selection does not support task type '{task_type.value}'")
    metric, direction = _TASK_SELECTION_METRIC[task_type]

    notes = [
        _NOTE_SELECTION_ONLY,
        f"task type: {task_type.value}",
        f"selection metric for task '{task_type.value}': {metric} ({direction})",
    ]

    eligible: list[tuple[float, DLCandidateRank]] = []
    ineligible: list[DLCandidateRank] = []

    for candidate in candidates:
        candidate_id = candidate.candidate_id
        architecture_name = candidate.training_config.architecture_name
        modeling_result = run_mlp_modeling(
            X_train,
            y_train,
            X_eval,
            y_eval,
            candidate.architecture,
            candidate.training_config,
            _import=_import,
        )

        if modeling_result.status is not TrainingRunStatus.COMPLETED:
            ineligible.append(
                DLCandidateRank(
                    candidate_id=candidate_id,
                    architecture_name=architecture_name,
                    status=modeling_result.status,
                    score=None,
                    metric=None,
                    rank=None,
                    reason=f"modeling execution did not complete: {modeling_result.reason}",
                    modeling_result=modeling_result,
                )
            )
            continue

        evaluation = modeling_result.evaluation
        value = evaluation.metrics.get(metric) if evaluation is not None else None
        if value is None:
            ineligible.append(
                DLCandidateRank(
                    candidate_id=candidate_id,
                    architecture_name=architecture_name,
                    status=modeling_result.status,
                    score=None,
                    metric=metric,
                    rank=None,
                    reason=f"selection metric '{metric}' is unavailable for this candidate",
                    modeling_result=modeling_result,
                )
            )
            continue

        if not math.isfinite(value):
            ineligible.append(
                DLCandidateRank(
                    candidate_id=candidate_id,
                    architecture_name=architecture_name,
                    status=modeling_result.status,
                    score=None,
                    metric=metric,
                    rank=None,
                    reason=f"selection metric '{metric}' is not a finite value for this candidate",
                    modeling_result=modeling_result,
                )
            )
            continue

        oriented = float(value) if direction == "minimize" else -float(value)
        rank_entry = DLCandidateRank(
            candidate_id=candidate_id,
            architecture_name=architecture_name,
            status=modeling_result.status,
            score=float(value),
            metric=metric,
            rank=None,  # filled in after sorting
            reason=f"eligible: {metric} = {value} ({direction})",
            modeling_result=modeling_result,
        )
        eligible.append((oriented, rank_entry))

    def _tie_break_key(item: tuple[float, DLCandidateRank]) -> tuple[float, str, str]:
        oriented, rank_entry = item
        config_json = _candidate_training_config_json(candidates, rank_entry.candidate_id)
        return (oriented, rank_entry.architecture_name or "", config_json)

    eligible.sort(key=_tie_break_key)

    ranking: list[DLCandidateRank] = []
    for position, (_, rank_entry) in enumerate(eligible, start=1):
        ranking.append(rank_entry.model_copy(update={"rank": position}))

    ineligible.sort(
        key=lambda r: (
            r.architecture_name or "",
            _candidate_training_config_json(candidates, r.candidate_id),
        )
    )
    ranking.extend(ineligible)

    if not eligible:
        return DLSelectionResult(
            status=TrainingRunStatus.FAILED,
            task_type=task_type,
            selection_metric=metric,
            selection_direction=direction,
            ranking=ranking,
            reason=f"no candidate had a usable '{metric}' selection metric",
            notes=notes,
        )

    winner = ranking[0]
    winner_oriented = eligible[0][0]
    tied = sum(1 for oriented, _ in eligible if oriented == winner_oriented)
    notes.append(
        f"selected candidate '{winner.candidate_id}' as the {direction} of '{metric}' "
        f"({metric} = {winner.score}) among {len(eligible)} eligible candidate(s)"
    )
    if tied > 1:
        notes.append(
            f"multiple eligible candidates ({tied}) tied on {metric}; the deterministic "
            "architecture-name / serialised-training-config ordering was used as the tie-break "
            "— the tie is an ordering choice, not a claim that either candidate performs better"
        )

    return DLSelectionResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=task_type,
        selection_metric=metric,
        selection_direction=direction,
        ranking=ranking,
        selected_candidate_id=winner.candidate_id,
        selected_architecture_name=winner.architecture_name,
        selected_score=winner.score,
        notes=notes,
    )


def _candidate_training_config_json(candidates: list[DLCandidate], candidate_id: str) -> str:
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate.training_config.model_dump_json()
    return ""  # unreachable in practice: every rank entry originates from `candidates`


__all__ = ["select_dl_models"]
