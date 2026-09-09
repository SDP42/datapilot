"""Phase 7 (stabilization) — deterministic evaluation summary.

:func:`summarize_evaluation` turns the authoritative Phase-7.4
:class:`TrainingOutcome` into an :class:`EvaluationResults` **status
mirror**. It exists so that ``ModelingSpec.evaluation`` is an explicit,
honest reference to where evaluation actually lives
(``ModelingSpec.training``) rather than a permanently-empty contract
section.

It **recomputes nothing, re-runs nothing, fits nothing, and stores no
metric value** — it only reads ``TrainingOutcome.status`` /
``TrainingOutcome.runs[*].status`` / ``TrainingOutcome.runs[*].metrics``
(the metric *names*, never the values) and reports counts + a pointer.
Byte-identical for equal inputs; no timestamp / UUID / randomness / file
/ network / LLM.
"""

from __future__ import annotations

from .models import (
    EvaluationResults,
    ModelingStatus,
    TrainingOutcome,
    TrainingRunStatus,
)

_NOTE_SOURCE_OF_TRUTH = (
    "evaluation metrics are recorded per training run in ModelingSpec.training "
    "(TrainingOutcome.runs[*].metrics), computed once by Phase 7.4 on the test partition; "
    "this section mirrors that result and recomputes nothing"
)


def summarize_evaluation(training: TrainingOutcome) -> EvaluationResults:
    """Deterministically summarise the Phase-7.4 :class:`TrainingOutcome`.

    Parameters
    ----------
    training:
        The **Phase-7.4** :class:`TrainingOutcome`. A non-model raises
        ``TypeError``; it is **not mutated** and no metric is recomputed.

    Returns
    -------
    EvaluationResults
        ``status = completed`` (mirroring a completed training outcome)
        with ``source = "training_outcome"``, the run counts, and the
        sorted union of metric *names* present across completed runs;
        ``status = unavailable`` when ``training`` is not completed.
    """
    if not isinstance(training, TrainingOutcome):
        raise TypeError(
            f"summarize_evaluation expects a TrainingOutcome, got {type(training).__name__}"
        )

    if training.status is not ModelingStatus.COMPLETED:
        return EvaluationResults(
            status=ModelingStatus.UNAVAILABLE,
            reason=(
                f"model training is not completed (status = {training.status.value}); "
                "no evaluation results are available to summarise"
            ),
            source="training_outcome",
            notes=[_NOTE_SOURCE_OF_TRUTH],
        )

    completed = [r for r in training.runs if r.status is TrainingRunStatus.COMPLETED]
    metric_names = sorted({name for run in completed for name in run.metrics})

    notes = [
        _NOTE_SOURCE_OF_TRUTH,
        (
            f"{len(completed)} of {len(training.runs)} training run(s) produced evaluation "
            "metrics on the test partition"
        ),
    ]
    if training.reason:
        notes.append(f"training outcome note: {training.reason}")

    return EvaluationResults(
        status=ModelingStatus.COMPLETED,
        reason=None,
        source="training_outcome",
        evaluated_run_count=len(training.runs),
        successful_run_count=len(completed),
        metric_names=metric_names,
        notes=notes,
    )
