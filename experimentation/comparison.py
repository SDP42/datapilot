"""Phase 9.3 — deterministic comparison of already-recorded experiments.

:func:`compare_experiments` is selection-only, mirroring
:func:`data_engine.modeling.select_model` /
:func:`dl_engine.select_dl_models`'s own boundary exactly: it retrains,
re-evaluates, and recomputes nothing — it only reads each
:class:`~experimentation.contracts.ExperimentRecord`'s own
*already-established* selection metric (``classical_result.selection`` for
a ``CLASSICAL_MODELING`` record, ``dl_selection_result`` directly for a
``DL_SELECTION`` record) and ranks the records that have one.
"""

from __future__ import annotations

import math

from data_engine.modeling import ModelSelection, TrainingRunStatus
from dl_engine import DLSelectionResult

from .contracts import (
    ExperimentComparisonEntry,
    ExperimentComparisonResult,
    ExperimentRecord,
    ExperimentSource,
)


def _no_records(reason: str) -> ExperimentComparisonResult:
    return ExperimentComparisonResult(status=TrainingRunStatus.FAILED, entries=[], reason=reason)


def _established_selection(
    record: ExperimentRecord,
) -> tuple[str, str, float] | None:
    """Return (metric, direction, selected_score) if `record` has an established selection."""
    selection: ModelSelection | DLSelectionResult | None
    if record.source is ExperimentSource.CLASSICAL_MODELING and record.classical_result is not None:
        selection = record.classical_result.selection
    elif record.source is ExperimentSource.DL_SELECTION:
        selection = record.dl_selection_result
    else:
        return None

    if selection is None:
        return None
    metric = selection.selection_metric
    direction = selection.selection_direction
    score = selection.selected_score
    if metric is None or direction is None or score is None or not math.isfinite(score):
        return None
    return metric, direction, score


def compare_experiments(records: list[ExperimentRecord]) -> ExperimentComparisonResult:
    """Rank already-recorded experiments by their own established selection metric.

    Every record remains visible in ``entries`` — an ineligible record
    (a ``DL_MODELING`` record, which has no selection; a
    ``CLASSICAL_MODELING`` / ``DL_SELECTION`` record whose own selection
    never picked a winner; or a non-finite score) gets ``rank = None`` and
    an explicit reason, never silently dropped.

    All eligible records must share the same ``(metric, direction)`` pair
    — a mismatch (e.g. one record's ``rmse`` against another's ``f1``)
    returns ``status = failed`` with an explicit reason rather than
    silently picking one metric to compare by.

    Deterministic tie-break: eligible records sort by ``(score oriented so
    lower is always better, experiment_id)`` — ``experiment_id`` is each
    record's own fixed, already-assigned identity, so this ordering is
    reproducible for any given fixed list of records (not a claim that the
    identity itself is deterministic across separate recordings of the
    "same" run — see ``experimentation.contracts``'s own docstring for why
    experiment identity is deliberately random).
    """
    if not records:
        return _no_records("no experiment records were supplied for comparison")

    eligible: list[tuple[float, ExperimentComparisonEntry]] = []
    ineligible: list[ExperimentComparisonEntry] = []
    seen_metrics: set[tuple[str, str]] = set()

    for record in records:
        established = _established_selection(record)
        if established is None:
            reason = (
                "no established selection metric for this record's source "
                f"('{record.source.value if record.source else 'unknown'}')"
                if record.source is ExperimentSource.DL_MODELING
                else "this record's own selection did not pick a winner"
            )
            ineligible.append(
                ExperimentComparisonEntry(
                    experiment_id=record.experiment_id,
                    source=record.source or ExperimentSource.DL_MODELING,
                    created_at=record.created_at,
                    score=None,
                    rank=None,
                    reason=reason,
                )
            )
            continue

        metric, direction, score = established
        seen_metrics.add((metric, direction))
        oriented = score if direction == "minimize" else -score
        eligible.append(
            (
                oriented,
                ExperimentComparisonEntry(
                    experiment_id=record.experiment_id,
                    source=record.source,  # type: ignore[arg-type]
                    created_at=record.created_at,
                    score=score,
                    rank=None,  # filled in after sorting
                    reason=f"eligible: {metric} = {score} ({direction})",
                ),
            )
        )

    if len(seen_metrics) > 1:
        return ExperimentComparisonResult(
            status=TrainingRunStatus.FAILED,
            entries=[entry for _, entry in eligible] + ineligible,
            reason=(
                "eligible records disagree on (selection metric, direction); got: "
                f"{sorted(seen_metrics)}"
            ),
        )

    eligible.sort(key=lambda item: (item[0], item[1].experiment_id))

    entries: list[ExperimentComparisonEntry] = []
    for position, (_, entry) in enumerate(eligible, start=1):
        entries.append(entry.model_copy(update={"rank": position}))
    ineligible.sort(key=lambda e: e.experiment_id)
    entries.extend(ineligible)

    if not eligible:
        return ExperimentComparisonResult(
            status=TrainingRunStatus.FAILED,
            entries=entries,
            reason="no supplied record has an established, finite selection metric",
        )

    metric, direction = next(iter(seen_metrics))
    winner = entries[0]
    return ExperimentComparisonResult(
        status=TrainingRunStatus.COMPLETED,
        metric=metric,
        direction=direction,
        entries=entries,
        selected_experiment_id=winner.experiment_id,
        selected_score=winner.score,
    )


__all__ = ["compare_experiments"]
