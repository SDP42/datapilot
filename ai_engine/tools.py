"""Phase 11.2 — the fixed tool-schema vocabulary.

Per ``docs/architecture-principles.md`` #6 ("An LLM recommendation is
executed only after translation into a typed, parameterised call to a
deterministic tool, followed by validation") and #11 ("AI agents use
tools, not internal state"), an LLM never reasons about or invents a
call directly — it is given this fixed, closed set of named,
JSON-schema-described tools and may only reference them by name.
:func:`ai_engine.recommendations.recommend_next_steps` validates every
LLM-proposed recommendation's ``tool`` field against
:data:`TOOL_NAMES` and drops (never silently keeps) one that doesn't
match.

Declarative only — **no tool here is ever executed by this module**.
Actually invoking one of these deterministic functions from an
LLM-approved recommendation is explicitly Phase 12's concern
(planner -> executor -> critic loop), out of scope here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """One deterministic DataPilot capability, described for an LLM.

    ``parameters`` is a JSON Schema object (the same shape both Anthropic
    and OpenAI's tool-use / function-calling APIs already expect) —
    never a free-text parameter description.
    """

    name: str
    description: str
    parameters: dict = Field(description="A JSON Schema object describing the tool's arguments.")


_TOOL_LIST: list[ToolSchema] = [
    ToolSchema(
        name="analyze_quality",
        description="Run Phase-2 deterministic data-quality analysis (missing values, "
        "duplicates, outliers, skew, class imbalance, type mismatches) on a dataset.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_version_id": {"type": "string"},
                "target_column": {"type": "string"},
            },
            "required": ["dataset_version_id"],
        },
    ),
    ToolSchema(
        name="analyze_dataframe",
        description="Run Phase-4 deterministic exploratory data analysis (univariate / "
        "bivariate statistics, hypothesis tests, effect sizes, visualization recommendations).",
        parameters={
            "type": "object",
            "properties": {"dataset_version_id": {"type": "string"}},
            "required": ["dataset_version_id"],
        },
    ),
    ToolSchema(
        name="understand_problem",
        description="Run Phase-5 deterministic problem understanding (target identification, "
        "task-type inference, candidate metrics, feasibility assessment) given a stated objective.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_version_id": {"type": "string"},
                "objective": {"type": "string"},
            },
            "required": ["dataset_version_id", "objective"],
        },
    ),
    ToolSchema(
        name="run_modeling_pipeline",
        description="Run the full deterministic Phase-7 modeling pipeline (readiness, split "
        "planning, candidate generation, baseline training & evaluation, model selection) end "
        "to end on a dataset given a stated objective.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_version_id": {"type": "string"},
                "objective": {"type": "string"},
                "forecast_horizon": {"type": "integer", "minimum": 1},
            },
            "required": ["dataset_version_id", "objective"],
        },
    ),
    ToolSchema(
        name="select_dl_models",
        description="Run Phase-8 deterministic deep-learning candidate selection (build, "
        "train, evaluate each supplied MLP / CNN / LSTM / Transformer candidate exactly once, "
        "rank by the task's fixed selection metric).",
        parameters={
            "type": "object",
            "properties": {
                "dataset_version_id": {"type": "string"},
                "architecture_names": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["mlp", "cnn", "lstm", "transformer"]},
                },
            },
            "required": ["dataset_version_id", "architecture_names"],
        },
    ),
    ToolSchema(
        name="compute_permutation_importance",
        description="Run Phase-10 permutation feature importance on an already-fitted model.",
        parameters={
            "type": "object",
            "properties": {
                "dataset_version_id": {"type": "string"},
                "model_id": {"type": "string"},
            },
            "required": ["dataset_version_id", "model_id"],
        },
    ),
    ToolSchema(
        name="compare_experiments",
        description="Run Phase-9 deterministic comparison of multiple already-recorded "
        "experiments by each one's own established selection metric.",
        parameters={
            "type": "object",
            "properties": {
                "experiment_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2}
            },
            "required": ["experiment_ids"],
        },
    ),
]

TOOLS_BY_NAME: dict[str, ToolSchema] = {tool.name: tool for tool in _TOOL_LIST}
TOOL_NAMES: frozenset[str] = frozenset(TOOLS_BY_NAME)


def get_tool_schema(name: str) -> ToolSchema:
    """The named tool's schema. Raises `KeyError` for an unknown name."""
    return TOOLS_BY_NAME[name]


def list_tools() -> list[ToolSchema]:
    """Every declared tool, in a fixed, stable order (declaration order)."""
    return list(_TOOL_LIST)


__all__ = ["TOOLS_BY_NAME", "TOOL_NAMES", "ToolSchema", "get_tool_schema", "list_tools"]
