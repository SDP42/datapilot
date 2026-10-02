"""Phase 11.1 — deterministic context building for LLM reasoning.

Per ``docs/architecture-principles.md`` #11 ("AI agents use tools, not
internal state... never read or write engine internals or dataframes
directly") and ``ai_engine.providers.base``'s own docstring ("never gives
the model direct access to datasets or execution state"), the LLM only
ever sees **already-computed, structured results** — a
:class:`~data_engine.profiling.DatasetProfile`, a
:class:`~data_engine.quality.QualityReport`, an
:class:`~data_engine.eda.EDAReport`, a
:class:`~data_engine.problem_understanding.ProblemSpec`, a
:class:`~data_engine.modeling.ModelingSpec`, a
:class:`~explainability.contracts.FeatureImportanceResult`, an
:class:`~experimentation.contracts.ExperimentRecord` — never a raw
``pandas.DataFrame`` or a fitted estimator.

:func:`build_analysis_context` accepts any already-produced Pydantic
report generically (via ``model_dump``), so it never needs updating when
a new report type is added to an earlier phase, and never risks
mis-naming one of that report's own fields. Deterministic: section keys
and every nested dict's keys are rendered in sorted order by
:func:`render_context_as_text`, so the same context renders to
byte-identical text on every call.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class AnalysisContext(BaseModel):
    """A bundle of already-computed, structured results about one dataset.

    ``sections`` maps a caller-chosen name (e.g. ``"quality"``,
    ``"modeling"``) to that report's own ``model_dump(mode="json")`` — the
    full nested structure, never flattened or summarised, since the LLM
    is meant to read the real structured result, not a lossy paraphrase
    of it.
    """

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = None
    sections: dict[str, dict] = Field(default_factory=dict)


def build_analysis_context(
    dataset_id: str,
    *,
    dataset_version_id: str | None = None,
    objective: str | None = None,
    **reports: BaseModel | None,
) -> AnalysisContext:
    """Bundle any number of named, already-produced reports into one context.

    Each keyword argument names a section (e.g.
    ``build_analysis_context("ds1", quality=quality_report, modeling=spec)``);
    a ``None`` value is skipped (the caller may not have run every phase).
    Never computes, infers, or re-derives anything from the reports —
    purely a structural bundling. Never mutates an input report.
    """
    sections = {
        name: report.model_dump(mode="json", exclude_none=True)
        for name, report in reports.items()
        if report is not None
    }
    return AnalysisContext(
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        objective=objective,
        sections=sections,
    )


def render_context_as_text(context: AnalysisContext) -> str:
    """Render `context` as deterministic, LLM-readable plain text.

    Sections are rendered in sorted-name order; every nested dict is
    serialised with ``sort_keys=True`` — the same context always renders
    to byte-identical text, regardless of the order sections were added
    in :func:`build_analysis_context`.
    """
    lines = [f"Dataset: {context.dataset_id}"]
    if context.dataset_version_id:
        lines.append(f"Dataset version: {context.dataset_version_id}")
    if context.objective:
        lines.append(f"Objective: {context.objective}")

    for name in sorted(context.sections):
        lines.append(f"\n## {name}")
        lines.append(json.dumps(context.sections[name], indent=2, sort_keys=True))

    return "\n".join(lines)


def add_section(context: AnalysisContext, name: str, data: dict) -> AnalysisContext:
    """Return a new `AnalysisContext` with one more raw JSON section added.

    Unlike :func:`build_analysis_context` (which takes a Pydantic report
    and calls ``model_dump`` itself), this accepts an already-JSON
    ``dict`` directly — the shape
    :class:`~ai_engine.execution.ExecutionResult.output` already is, so
    Phase 12's agent loop can fold a tool's result back into the context
    for the next planning round without re-wrapping it in a model first.
    Never mutates `context`; a `name` already present is overwritten in
    the returned copy (never silently merged field-by-field).
    """
    return context.model_copy(update={"sections": {**context.sections, name: data}})


__all__ = ["AnalysisContext", "add_section", "build_analysis_context", "render_context_as_text"]
