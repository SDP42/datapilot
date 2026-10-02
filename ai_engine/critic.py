"""Phase 12.2 — the deterministic critic.

:func:`evaluate_step` is intentionally **not** a second LLM call — it is
a plain, deterministic inspection of an already-produced
:class:`~ai_engine.execution.ExecutionResult`'s own ``status``. Making
the stop/continue decision deterministic (rather than asking the LLM
"should we continue?") keeps the autonomous loop's termination behaviour
reproducible and auditable: given the same sequence of execution
results, the same verdicts always follow.
"""

from __future__ import annotations

from enum import Enum

from data_engine.modeling import TrainingRunStatus

from .execution import ExecutionResult


class StepVerdict(str, Enum):
    """Whether an autonomous run should continue after one executed step."""

    CONTINUE = "continue"
    STOP = "stop"


def evaluate_step(result: ExecutionResult) -> tuple[StepVerdict, str]:
    """Decide whether to continue after `result`, and why.

    `CONTINUE` only when the tool actually completed; `STOP` for both
    `unavailable` (the recommended tool couldn't run given the supplied
    context — not necessarily an error, but nothing further can build on
    a result that doesn't exist) and `failed` (something broke). The two
    are distinguished only in the returned reason text, never in the
    verdict itself — both mean "this step produced no usable output to
    continue from."
    """
    if result.status is TrainingRunStatus.COMPLETED:
        return StepVerdict.CONTINUE, f"'{result.tool}' completed successfully"
    if result.status is TrainingRunStatus.UNAVAILABLE:
        return StepVerdict.STOP, f"'{result.tool}' was unavailable: {result.reason}"
    return StepVerdict.STOP, f"'{result.tool}' failed: {result.reason}"


__all__ = ["StepVerdict", "evaluate_step"]
