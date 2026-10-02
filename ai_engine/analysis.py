"""Phase 11.4 — LLM-backed interpretation and recommendations.

:func:`interpret_results` and :func:`recommend_next_steps` both take an
already-constructed :class:`~ai_engine.providers.base.LLMProvider` (any
concrete implementation — :class:`~ai_engine.providers.anthropic_provider.AnthropicProvider`,
or a test double) and an :class:`~ai_engine.context.AnalysisContext`
(Phase 11.1) — neither function ever constructs a provider or a context
itself, and neither ever reads a dataframe or a fitted model directly
(``docs/architecture-principles.md`` #11).

Per principle #6 ("An LLM recommendation is executed only after
translation into a typed, parameterised call to a deterministic tool,
followed by validation"), :func:`recommend_next_steps` validates every
proposed recommendation's ``tool`` name against the known
:data:`~ai_engine.tools.TOOL_NAMES` vocabulary and **drops** (never
silently keeps, always reports in ``dropped``) one that doesn't match.
Nothing here ever executes a recommended tool — that remains Phase 12's
concern.
"""

from __future__ import annotations

import json

from data_engine.modeling import TrainingRunStatus
from pydantic import BaseModel, Field

from .context import AnalysisContext, render_context_as_text
from .providers.base import LLMMessage, LLMProvider
from .tools import ToolSchema, list_tools

_DEFAULT_INTERPRET_SYSTEM_PROMPT = (
    "You are a data-science assistant analysing structured results already produced by a "
    "deterministic pipeline. You are given JSON-serialised reports only — never raw data. "
    "Write a concise, factual natural-language summary of what the reports show. Never invent "
    "a number, metric, or finding that is not present in the supplied data."
)

_RECOMMEND_SYSTEM_PROMPT = (
    "You are a data-science assistant recommending next steps from a fixed list of tools. "
    "Respond with a single JSON object of the exact shape "
    '{"recommendations": [{"tool": "<tool name>", "rationale": "<why>", '
    '"parameters": {<tool-specific arguments>}}]} and nothing else — no prose outside the JSON. '
    "`tool` must be exactly one of the tool names you were given; never invent a tool name."
)


class AnalysisResult(BaseModel):
    """The result of one :func:`interpret_results` call."""

    status: TrainingRunStatus = Field(
        description="completed (text produced), or failed (the LLM call itself raised)."
    )
    analysis_text: str | None = Field(
        default=None, description="The LLM's natural-language summary."
    )
    reason: str | None = Field(
        default=None, description="Why status is failed; None when completed."
    )


class Recommendation(BaseModel):
    """One LLM-proposed, tool-validated next step.

    ``tool`` is guaranteed to be a name from :data:`~ai_engine.tools.TOOL_NAMES`
    by the time this object exists — :func:`recommend_next_steps` never
    constructs one for an unrecognised tool name.
    """

    tool: str
    rationale: str
    parameters: dict = Field(default_factory=dict)


class RecommendationResult(BaseModel):
    """The result of one :func:`recommend_next_steps` call."""

    status: TrainingRunStatus = Field(
        description="completed (the LLM responded with parseable JSON, possibly zero "
        "recommendations), or failed (the LLM call raised, or the response could not be "
        "parsed as the expected JSON shape)."
    )
    recommendations: list[Recommendation] = Field(default_factory=list)
    dropped: list[str] = Field(
        default_factory=list,
        description="One entry per LLM-proposed recommendation that referenced an unknown "
        "tool name — never silently discarded without a reason.",
    )
    raw_response: str | None = Field(
        default=None,
        description="The provider's raw response text; populated on a parse failure so the "
        "unparseable text is never lost, and always populated alongside a successful parse.",
    )
    reason: str | None = Field(
        default=None, description="Why status is failed; None when completed."
    )


def interpret_results(
    provider: LLMProvider, context: AnalysisContext, *, system_prompt: str | None = None
) -> AnalysisResult:
    """Ask `provider` for a natural-language summary of `context`.

    Never raises for a provider-level failure (a network error, an
    invalid API key, a malformed response) — reported as `status =
    failed` with `reason` naming the underlying error. Never fabricates
    analysis text of its own.
    """
    messages = [
        LLMMessage(role="system", content=system_prompt or _DEFAULT_INTERPRET_SYSTEM_PROMPT),
        LLMMessage(role="user", content=render_context_as_text(context)),
    ]
    try:
        response = provider.complete(messages)
    except Exception as exc:  # noqa: BLE001 - any provider's own exception hierarchy is not public API
        return AnalysisResult(status=TrainingRunStatus.FAILED, reason=f"LLM call failed: {exc}")

    return AnalysisResult(status=TrainingRunStatus.COMPLETED, analysis_text=response.text)


def _build_recommendation_prompt(context: AnalysisContext, tools: list[ToolSchema]) -> str:
    tool_descriptions = [
        {"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools
    ]
    return (
        f"{render_context_as_text(context)}\n\n"
        "## Available tools\n"
        f"{json.dumps(tool_descriptions, indent=2, sort_keys=True)}"
    )


def recommend_next_steps(
    provider: LLMProvider,
    context: AnalysisContext,
    *,
    tools: list[ToolSchema] | None = None,
) -> RecommendationResult:
    """Ask `provider` to recommend next steps from `tools` (default: every declared tool).

    Every recommendation in the result has already been validated: its
    `tool` is one of `tools`' own names. An LLM-proposed recommendation
    naming an unrecognised tool is **dropped** (never silently kept),
    with a corresponding entry in `RecommendationResult.dropped`.

    `status = failed` for a provider-level failure (same as
    :func:`interpret_results`) or when the response cannot be parsed as
    the expected `{"recommendations": [...]}` JSON shape — the raw text
    is never guessed at or partially salvaged beyond that.
    """
    tool_list = tools if tools is not None else list_tools()
    allowed_names = {tool.name for tool in tool_list}

    messages = [
        LLMMessage(role="system", content=_RECOMMEND_SYSTEM_PROMPT),
        LLMMessage(role="user", content=_build_recommendation_prompt(context, tool_list)),
    ]
    try:
        response = provider.complete(messages)
    except Exception as exc:  # noqa: BLE001 - any provider's own exception hierarchy is not public API
        return RecommendationResult(
            status=TrainingRunStatus.FAILED, reason=f"LLM call failed: {exc}"
        )

    try:
        parsed = json.loads(response.text)
        raw_items = parsed["recommendations"]
        if not isinstance(raw_items, list):
            raise TypeError(f"'recommendations' must be a list, got {type(raw_items).__name__}")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return RecommendationResult(
            status=TrainingRunStatus.FAILED,
            raw_response=response.text,
            reason=f"could not parse the LLM response as the expected JSON shape: {exc}",
        )

    recommendations: list[Recommendation] = []
    dropped: list[str] = []
    for item in raw_items:
        if not isinstance(item, dict) or "tool" not in item:
            dropped.append(f"dropped a malformed recommendation entry: {item!r}")
            continue
        tool_name = item["tool"]
        if tool_name not in allowed_names:
            dropped.append(f"dropped recommendation for unknown tool {tool_name!r}")
            continue
        recommendations.append(
            Recommendation(
                tool=tool_name,
                rationale=str(item.get("rationale", "")),
                parameters=item.get("parameters", {}) or {},
            )
        )

    return RecommendationResult(
        status=TrainingRunStatus.COMPLETED,
        recommendations=recommendations,
        dropped=dropped,
        raw_response=response.text,
    )


__all__ = [
    "AnalysisResult",
    "Recommendation",
    "RecommendationResult",
    "interpret_results",
    "recommend_next_steps",
]
