"""Phase 12.3/12.4 — the planner -> executor -> critic loop, under budget,
producing a human-reviewable trace.

:func:`run_autonomous_experimentation` is the composition that closes
Phase 11/12 together: at each step it asks an LLM for recommendations
(:func:`ai_engine.analysis.recommend_next_steps`, Phase 11.4), executes
the first one (:func:`ai_engine.execution.execute_tool`, Phase 12.1),
and deterministically decides whether to continue
(:func:`ai_engine.critic.evaluate_step`, Phase 12.2) — exactly the
planner / executor / critic loop the roadmap names, stopping at
``max_steps`` (the budget) or on the critic's first `STOP` verdict,
whichever comes first.

Every step — what was recommended, why, what actually happened, and why
the loop did or didn't continue — is recorded in an
:class:`AutonomousRunTrace`, a fully JSON-serialisable, ordered record a
human can review after the fact. This module never executes a tool
outside :func:`~ai_engine.execution.execute_tool`'s own validation, and
never lets the loop run unbounded — ``max_steps`` is a required,
explicit argument with no silent "run forever" default.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from data_engine.modeling import TrainingRunStatus
from pydantic import BaseModel, ConfigDict, Field

from .analysis import recommend_next_steps
from .context import AnalysisContext, add_section
from .critic import StepVerdict, evaluate_step
from .execution import ExecutionContext, ExecutionResult, execute_tool
from .providers.base import LLMProvider
from .tools import ToolSchema

DEFAULT_MAX_STEPS = 5


class ExecutionStep(BaseModel):
    """One planner -> executor -> critic cycle in an :class:`AutonomousRunTrace`."""

    model_config = ConfigDict(protected_namespaces=())

    step_number: int = Field(description="1-based position in the run.")
    tool: str
    parameters: dict
    rationale: str = Field(description="The LLM's own stated reason for proposing this step.")
    result: ExecutionResult
    verdict: StepVerdict
    verdict_reason: str


class AutonomousRunTrace(BaseModel):
    """A complete, human-reviewable record of one autonomous run.

    Like :class:`~experimentation.contracts.ExperimentRecord`, this
    deliberately carries a timestamp and a random ``run_id`` — a record
    of *when a specific run happened* cannot be deterministic by nature,
    the same reasoning Phase 9's own contract already established.
    """

    model_config = ConfigDict(protected_namespaces=())

    run_id: str = Field(description="A random UUID4, generated once when the run starts.")
    created_at: datetime
    dataset_id: str
    objective: str | None = None

    status: TrainingRunStatus = Field(
        description="completed (the run finished — by budget, by exhausting recommendations, "
        "or by an unavailable step — without an actual execution failure), or failed (some "
        "step's execution genuinely failed)."
    )
    steps: list[ExecutionStep] = Field(default_factory=list)
    stop_reason: str


def run_autonomous_experimentation(
    provider: LLMProvider,
    context: AnalysisContext,
    execution_context: ExecutionContext,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    tools: list[ToolSchema] | None = None,
) -> AutonomousRunTrace:
    """Run up to `max_steps` planner -> executor -> critic cycles and return the full trace.

    Each cycle: ask `provider` for recommendations against the
    **current** context (folding every prior step's own result back in
    via :func:`ai_engine.context.add_section`, so later steps can build
    on earlier ones); take the first recommendation; execute it; let the
    deterministic critic decide whether to continue. Stops immediately
    (never runs a second recommendation from the same call) on:

    * the recommendation call itself failing or returning zero
      recommendations ("nothing left to do" — `status = completed`);
    * the critic's `STOP` verdict (an unavailable or failed step —
      `status = failed` only when the step itself genuinely failed, not
      merely unavailable);
    * reaching `max_steps` (`status = completed` — the budget, not a
      failure).

    Never raises for any of the above — every stop condition is recorded
    in the returned trace's `stop_reason`.
    """
    steps: list[ExecutionStep] = []
    working_context = context

    for step_number in range(1, max_steps + 1):
        rec_result = recommend_next_steps(provider, working_context, tools=tools)
        if rec_result.status is not TrainingRunStatus.COMPLETED:
            return AutonomousRunTrace(
                run_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
                dataset_id=context.dataset_id,
                objective=context.objective,
                status=TrainingRunStatus.FAILED,
                steps=steps,
                stop_reason=f"recommendation call failed at step {step_number}: {rec_result.reason}",
            )
        if not rec_result.recommendations:
            return AutonomousRunTrace(
                run_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
                dataset_id=context.dataset_id,
                objective=context.objective,
                status=TrainingRunStatus.COMPLETED,
                steps=steps,
                stop_reason=f"no further recommendations at step {step_number}",
            )

        recommendation = rec_result.recommendations[0]
        result = execute_tool(recommendation.tool, recommendation.parameters, execution_context)
        verdict, verdict_reason = evaluate_step(result)
        steps.append(
            ExecutionStep(
                step_number=step_number,
                tool=recommendation.tool,
                parameters=recommendation.parameters,
                rationale=recommendation.rationale,
                result=result,
                verdict=verdict,
                verdict_reason=verdict_reason,
            )
        )

        if verdict is StepVerdict.STOP:
            overall_status = (
                TrainingRunStatus.FAILED
                if result.status is TrainingRunStatus.FAILED
                else TrainingRunStatus.COMPLETED
            )
            return AutonomousRunTrace(
                run_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
                dataset_id=context.dataset_id,
                objective=context.objective,
                status=overall_status,
                steps=steps,
                stop_reason=verdict_reason,
            )

        if result.output is not None:
            working_context = add_section(
                working_context, f"step_{step_number}_{recommendation.tool}", result.output
            )

    return AutonomousRunTrace(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        dataset_id=context.dataset_id,
        objective=context.objective,
        status=TrainingRunStatus.COMPLETED,
        steps=steps,
        stop_reason=f"reached max_steps budget ({max_steps})",
    )


__all__ = [
    "DEFAULT_MAX_STEPS",
    "AutonomousRunTrace",
    "ExecutionStep",
    "run_autonomous_experimentation",
]
