"""AI Scientist / Agent (Phase 11) — LLM reasoning over structured results.

Per ``docs/architecture-principles.md`` #11 ("AI agents use tools, not
internal state") and #6 ("An LLM recommendation is executed only after
translation into a typed, parameterised call to a deterministic tool,
followed by validation"), this package never gives an LLM direct access
to a dataframe or a fitted model, and never executes an LLM-proposed
action itself.

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

Out of scope for Phase 11 (until explicitly implemented): actually
*executing* a recommended tool (Phase 12's planner -> executor -> critic
loop); a second concrete provider (OpenAI, local models); multi-turn
conversation / tool-use loops; natural-language-to-SQL or any other
data-access path that would give the LLM something other than a
structured report.
"""

from __future__ import annotations

from .analysis import (
    AnalysisResult,
    Recommendation,
    RecommendationResult,
    interpret_results,
    recommend_next_steps,
)
from .context import AnalysisContext, build_analysis_context, render_context_as_text
from .providers.base import LLMMessage, LLMProvider, LLMResponse
from .tools import TOOL_NAMES, TOOLS_BY_NAME, ToolSchema, get_tool_schema, list_tools

__all__ = [
    "TOOLS_BY_NAME",
    "TOOL_NAMES",
    "AnalysisContext",
    "AnalysisResult",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "Recommendation",
    "RecommendationResult",
    "ToolSchema",
    "build_analysis_context",
    "get_tool_schema",
    "interpret_results",
    "list_tools",
    "recommend_next_steps",
    "render_context_as_text",
]
