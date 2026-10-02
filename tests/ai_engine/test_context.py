"""Phase 11.1 — deterministic context building (`ai_engine.context`)."""

from __future__ import annotations

from data_engine.modeling import ModelingSpec
from data_engine.problem_understanding import ProblemSpec
from ai_engine.context import AnalysisContext, build_analysis_context, render_context_as_text


def test_build_context_with_no_reports():
    context = build_analysis_context("ds1")
    assert context.dataset_id == "ds1"
    assert context.sections == {}


def test_build_context_skips_none_reports():
    spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    context = build_analysis_context("ds1", modeling=spec, problem=None)
    assert set(context.sections) == {"modeling"}


def test_build_context_includes_multiple_sections():
    modeling = ModelingSpec(dataset_id="ds1", objective_provided=False)
    problem = ProblemSpec(dataset_id="ds1", objective_provided=False)
    context = build_analysis_context("ds1", modeling=modeling, problem=problem)
    assert set(context.sections) == {"modeling", "problem"}
    assert context.sections["modeling"]["dataset_id"] == "ds1"


def test_build_context_never_mutates_input_report():
    spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    before = spec.model_dump_json()
    build_analysis_context("ds1", modeling=spec)
    assert spec.model_dump_json() == before


def test_render_includes_dataset_and_objective():
    context = build_analysis_context("ds1", objective="predict churn")
    text = render_context_as_text(context)
    assert "Dataset: ds1" in text
    assert "Objective: predict churn" in text


def test_render_omits_objective_when_absent():
    context = build_analysis_context("ds1")
    text = render_context_as_text(context)
    assert "Objective" not in text


def test_render_sections_in_sorted_order():
    modeling = ModelingSpec(dataset_id="ds1", objective_provided=False)
    problem = ProblemSpec(dataset_id="ds1", objective_provided=False)
    context = build_analysis_context("ds1", modeling=modeling, problem=problem)
    text = render_context_as_text(context)
    assert text.index("## modeling") < text.index("## problem")


def test_render_is_deterministic():
    modeling = ModelingSpec(dataset_id="ds1", objective_provided=False)
    context_a = build_analysis_context("ds1", modeling=modeling)
    context_b = build_analysis_context("ds1", modeling=modeling)
    assert render_context_as_text(context_a) == render_context_as_text(context_b)


def test_context_is_json_roundtrippable():
    modeling = ModelingSpec(dataset_id="ds1", objective_provided=False)
    context = build_analysis_context("ds1", modeling=modeling)
    restored = AnalysisContext.model_validate_json(context.model_dump_json())
    assert restored == context
