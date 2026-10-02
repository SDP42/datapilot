"""AI Scientist / Agent (Phase 11) and Autonomous Experimentation (Phase 12).

Per ``docs/architecture-principles.md`` #11 ("AI agents use tools, not
internal state") and #6 ("An LLM recommendation is executed only after
translation into a typed, parameterised call to a deterministic tool,
followed by validation"), this package never gives an LLM direct access
to a dataframe or a fitted model, and never executes an LLM-proposed
action without going through the validated tool layer.

**Phase 11.1** adds :func:`build_analysis_context` /
:func:`render_context_as_text` — deterministic bundling of any
already-produced Phase 1-10 report (``model_dump``-generic, so it never
needs updating when a new report type is added) into one
:class:`AnalysisContext`, rendered to sorted, byte-identical text.

**Phase 11.2** adds the fixed :data:`TOOL_NAMES` vocabulary
(:mod:`ai_engine.tools`) — JSON-schema-described deterministic
capabilities an LLM may *reference by name*, never invent or execute
directly.

**Phase 11.3** adds the first concrete
:class:`~ai_engine.providers.base.LLMProvider`:
:class:`~ai_engine.providers.anthropic_provider.AnthropicProvider`
(Phase 0's decision 0004 deferred concrete providers to this phase). The
``anthropic`` SDK is an **optional** dependency (the ``ai`` extra),
detected by :mod:`ai_engine.providers.availability`, mirroring
:mod:`dl_engine.availability`'s own ``torch`` boundary exactly.

**Phase 11.4** adds :func:`interpret_results` (natural-language summary)
and :func:`recommend_next_steps` (structured, tool-validated
recommendations — every proposed ``tool`` name is checked against
:data:`TOOL_NAMES`; an unrecognised one is dropped, never silently kept).
Both take an already-constructed ``LLMProvider`` — fully testable with a
trivial in-memory provider, no network or API key required.

**Phase 12.1** adds :func:`execute_tool` — the deterministic executor
that turns a validated :class:`Recommendation` into a real call against
the Phase 1-10 function it names, wrapping the result in a structured
:class:`ExecutionResult`. :class:`ExecutionContext` holds the runtime
resources (a dataframe, a fitted model, …) a handler needs, mirroring how
:func:`dl_engine.run_mlp_modeling` requires already-prepared data rather
than sourcing or fitting anything itself.

**Phase 12.2** adds :func:`~ai_engine.critic.evaluate_step` — a
*deterministic* critic (never a second LLM call) deciding whether an
autonomous run should continue after one executed step, for reproducible,
auditable termination behaviour.

**Phase 12.3/12.4** adds :func:`run_autonomous_experimentation` — the
planner (:func:`recommend_next_steps`) -> executor (:func:`execute_tool`)
-> critic (:func:`~ai_engine.critic.evaluate_step`) loop under an
explicit ``max_steps`` budget, producing a fully JSON-serialisable,
ordered :class:`AutonomousRunTrace` a human can review after the fact.

Out of scope for Phase 11/12 (until explicitly implemented): a second
concrete provider (OpenAI, local models); multi-turn conversation /
tool-use loops within one LLM call; natural-language-to-SQL or any other
data-access path that would give the LLM something other than a
structured report; parallel / branching plans (the loop is strictly
sequential, one recommendation executed per step).
"""

from __future__ import annotations

from .agent import (
    DEFAULT_MAX_STEPS,
    AutonomousRunTrace,
    ExecutionStep,
    run_autonomous_experimentation,
)
from .analysis import (
    AnalysisResult,
    Recommendation,
    RecommendationResult,
    interpret_results,
    recommend_next_steps,
)
from .context import AnalysisContext, add_section, build_analysis_context, render_context_as_text
from .critic import StepVerdict, evaluate_step
from .execution import ExecutionContext, ExecutionResult, execute_tool
from .providers.base import LLMMessage, LLMProvider, LLMResponse
from .tools import TOOL_NAMES, TOOLS_BY_NAME, ToolSchema, get_tool_schema, list_tools

__all__ = [
    "DEFAULT_MAX_STEPS",
    "TOOLS_BY_NAME",
    "TOOL_NAMES",
    "AnalysisContext",
    "AnalysisResult",
    "AutonomousRunTrace",
    "ExecutionContext",
    "ExecutionResult",
    "ExecutionStep",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "Recommendation",
    "RecommendationResult",
    "StepVerdict",
    "ToolSchema",
    "add_section",
    "build_analysis_context",
    "evaluate_step",
    "execute_tool",
    "get_tool_schema",
    "interpret_results",
    "list_tools",
    "recommend_next_steps",
    "render_context_as_text",
    "run_autonomous_experimentation",
]
