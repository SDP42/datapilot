"""Phase 4 — statistical-strength ranking of existing visualization specs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine import eda
from data_engine.eda import (
    EDAReport,
    PValueAvailability,
    VisualizationStatisticalStrength,
    VisualizationStatisticalStrengthAnalysis,
    VisualizationStatisticalStrengthStatus,
    analyze_dataframe,
    analyze_visualizations,
    rank_visualizations_by_statistical_strength,
    recommend_visualizations,
    render_plotly_visualization,
    render_visualization,
)
from data_engine.eda.bivariate import analyze_bivariate
from data_engine.eda.effects import analyze_effect_sizes
from data_engine.eda.statistics import analyze_statistics


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 150
    base = np.linspace(0.0, 10.0, n)
    return pd.DataFrame(
        {
            "size": base,
            "price": base * 2.0 + rng.normal(0.0, 0.2, n),  # near-perfect corr with size
            "weakling": rng.normal(0.0, 1.0, n),  # ~ independent
            "grade": (["A"] * 50 + ["B"] * 50 + ["C"] * 50),  # separates size/price
            "flavour": (["x", "y", "z"] * 50),  # ~ independent of grade
        }
    )


def _rec(analysis: VisualizationStatisticalStrengthAnalysis, **match):
    for r in analysis.recommendations:
        if all(getattr(r, k) == v for k, v in match.items()):
            return r
    raise AssertionError(f"no recommendation matching {match}")


# --- API / models -------------------------------------------------


def test_public_function_importable():
    assert eda.rank_visualizations_by_statistical_strength is (
        rank_visualizations_by_statistical_strength
    )


def test_return_type(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    assert isinstance(result, VisualizationStatisticalStrengthAnalysis)


def test_json_round_trip(df):
    result = rank_visualizations_by_statistical_strength(df, "grade")
    dumped = result.model_dump_json()
    assert VisualizationStatisticalStrengthAnalysis.model_validate_json(dumped) == result


def test_no_runtime_object_in_model(df):
    payload = rank_visualizations_by_statistical_strength(df, "price").model_dump(mode="json")
    # model_dump(mode="json") only succeeds for fully JSON-primitive content
    assert isinstance(payload["recommendations"], list)
    for entry in payload["recommendations"]:
        for value in entry.values():
            assert value is None or isinstance(value, (str, int, float, bool, list))


# --- numeric target ---------------------------------------------


def test_numeric_target_numeric_predictor_produces_candidate(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    rec = _rec(result, source_family="scatter_plots", predictor_column="size")
    assert rec.relationship == "numeric-numeric"
    assert rec.kind.value == "scatter_plot"


def test_correlation_evidence_matches_bivariate_layer(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    rec = _rec(result, predictor_column="size", relationship="numeric-numeric")
    pearson = next(
        c
        for c in analyze_bivariate(df).numeric_correlations
        if {c.column_a, c.column_b} == {"size", "price"}
    )
    assert rec.effect_size_name == "pearson_abs_r"
    assert rec.effect_size_value == pytest.approx(round(abs(pearson.correlation), 10))


def test_categorical_numeric_effect_matches_effect_size_layer(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    rec = _rec(result, predictor_column="grade", relationship="categorical-numeric")
    eta = next(
        c
        for c in analyze_effect_sizes(df).correlation_ratio
        if set(c.columns) == {"grade", "price"}
    )
    assert rec.effect_size_name == "correlation_ratio_eta"
    assert rec.effect_size_value == pytest.approx(round(eta.effect_size, 10))
    anova = next(r for r in analyze_statistics(df).anova if set(r.columns) == {"grade", "price"})
    assert rec.statistic_name == "anova_f"
    assert rec.p_value == pytest.approx(round(anova.p_value, 10))


def test_target_histogram_gets_no_fabricated_strength(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    assert not any(r.kind.value == "histogram" for r in result.recommendations)


def test_multiple_numeric_predictors_rank_deterministically(df):
    a = rank_visualizations_by_statistical_strength(df, "price")
    b = rank_visualizations_by_statistical_strength(df, "price")
    assert a.model_dump() == b.model_dump()
    assert [r.rank for r in a.recommendations] == list(range(1, len(a.recommendations) + 1))


def test_stronger_relationship_ranks_higher(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    strong = _rec(result, predictor_column="size")
    weak = _rec(result, predictor_column="weakling")
    assert strong.rank < weak.rank
    assert (strong.strength_score or 0) > (weak.strength_score or 0)


# --- categorical target --------------------------------------


def test_categorical_target_numeric_predictor_uses_effect_and_anova(df):
    result = rank_visualizations_by_statistical_strength(df, "grade")
    rec = _rec(result, predictor_column="price", relationship="categorical-numeric")
    assert rec.effect_size_name == "correlation_ratio_eta"
    assert rec.statistic_name == "anova_f"
    assert rec.p_value_availability is PValueAvailability.AVAILABLE


def test_categorical_target_categorical_predictor_uses_cramers_and_chi_square(df):
    result = rank_visualizations_by_statistical_strength(df, "grade")
    rec = _rec(result, predictor_column="flavour", relationship="categorical-categorical")
    assert rec.effect_size_name == "cramers_v"
    assert rec.statistic_name == "chi_square"
    assert rec.kind.value == "bar_chart"


def test_evidence_maps_back_to_correct_spec(df):
    eda_report = analyze_dataframe(df)
    result = rank_visualizations_by_statistical_strength(df, "grade")
    for rec in result.recommendations:
        spec = getattr(eda_report.visualizations, rec.source_family)[rec.source_index]
        assert spec.columns == rec.columns
        assert spec.kind == rec.kind


def test_target_bar_chart_gets_no_fabricated_strength(df):
    result = rank_visualizations_by_statistical_strength(df, "grade")
    assert not any(r.columns == ["grade"] for r in result.recommendations)


# --- p-values / effect sizes -------------------------------


def test_available_p_value_preserved_exactly(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    rec = _rec(result, predictor_column="grade", relationship="categorical-numeric")
    anova = next(r for r in analyze_statistics(df).anova if set(r.columns) == {"grade", "price"})
    assert rec.p_value == pytest.approx(round(anova.p_value, 10))
    assert rec.p_value_availability is PValueAvailability.AVAILABLE


def test_available_effect_size_preserved_exactly(df):
    result = rank_visualizations_by_statistical_strength(df, "grade")
    rec = _rec(result, predictor_column="flavour", relationship="categorical-categorical")
    cramers = next(
        c for c in analyze_effect_sizes(df).cramers_v if set(c.columns) == {"grade", "flavour"}
    )
    assert rec.effect_size_value == pytest.approx(round(cramers.effect_size, 10))


def test_missing_p_value_is_none_with_reason():
    # a constant numeric predictor: Spearman is unavailable for the pair
    df = pd.DataFrame({"target": [1.0, 2, 3, 4, 5, 6], "flat": [7.0] * 6})
    result = rank_visualizations_by_statistical_strength(df, "target")
    rec = _rec(result, predictor_column="flat")
    assert rec.p_value is None
    assert rec.p_value_availability is PValueAvailability.UNAVAILABLE
    assert rec.p_value_reason is not None


def test_missing_effect_size_is_none_with_reason():
    df = pd.DataFrame({"target": [1.0, 2, 3, 4, 5, 6], "flat": [7.0] * 6})
    result = rank_visualizations_by_statistical_strength(df, "target")
    rec = _rec(result, predictor_column="flat")
    assert rec.effect_size_value is None
    assert rec.effect_size_reason is not None
    assert rec.strength_score is None
    assert rec.strength_score_reason is not None


def test_ranking_not_by_p_value_alone(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    # size and weakling can both have p == 0.0-ish is not guaranteed; assert that
    # the ordering follows strength_score, not p_value
    scored = [r for r in result.recommendations if r.strength_score is not None]
    assert scored == sorted(scored, key=lambda r: -r.strength_score)


def test_strength_score_is_deterministic(df):
    a = [
        r.strength_score
        for r in rank_visualizations_by_statistical_strength(df, "price").recommendations
    ]
    b = [
        r.strength_score
        for r in rank_visualizations_by_statistical_strength(df, "price").recommendations
    ]
    assert a == b


# --- target handling --------------------------------------


def test_missing_target_unavailable(df):
    result = rank_visualizations_by_statistical_strength(df, "ghost")
    assert result.status is VisualizationStatisticalStrengthStatus.UNAVAILABLE
    assert "not in the DataFrame" in result.reason
    assert result.recommendations == []


def test_datetime_target_unavailable(df):
    frame = df.assign(when=pd.to_datetime(["2021-01-01"] * len(df)))
    result = rank_visualizations_by_statistical_strength(frame, "when")
    assert result.status is VisualizationStatisticalStrengthStatus.UNAVAILABLE
    assert "datetime" in result.reason


def test_all_missing_target_unavailable(df):
    frame = df.assign(empty=[np.nan] * len(df))
    result = rank_visualizations_by_statistical_strength(frame, "empty")
    assert result.status is VisualizationStatisticalStrengthStatus.UNAVAILABLE
    assert "no non-null observations" in result.reason


def test_high_cardinality_categorical_target_unavailable():
    frame = pd.DataFrame({"id": [f"v{i}" for i in range(120)], "n": range(120)})
    result = rank_visualizations_by_statistical_strength(frame, "id")
    assert result.status is VisualizationStatisticalStrengthStatus.UNAVAILABLE
    assert "cardinality" in result.reason


def test_target_is_required_positional():
    with pytest.raises(TypeError):
        rank_visualizations_by_statistical_strength(pd.DataFrame({"a": [1, 2, 3]}))  # type: ignore[call-arg]


# --- determinism ----------------------------------------


def test_row_shuffle_identical_dump(df):
    a = rank_visualizations_by_statistical_strength(df, "price")
    shuffled = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    b = rank_visualizations_by_statistical_strength(shuffled, "price")
    assert a.model_dump() == b.model_dump()


def test_repeated_calls_identical_json(df):
    dumps = {
        rank_visualizations_by_statistical_strength(df, "grade").model_dump_json() for _ in range(4)
    }
    assert len(dumps) == 1


def test_tie_break_is_total_and_explicit(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    keys = [
        (
            r.strength_score is None,
            -(r.strength_score or 0.0),
            r.p_value if r.p_value is not None else 2.0,
            r.kind.value,
            tuple(r.columns),
        )
        for r in result.recommendations
    ]
    assert keys == sorted(keys)


def test_ranks_unique_and_sequential(df):
    ranks = [
        r.rank for r in rank_visualizations_by_statistical_strength(df, "grade").recommendations
    ]
    assert ranks == list(range(1, len(ranks) + 1))


def test_recommendation_cap_is_deterministic(df):
    result = rank_visualizations_by_statistical_strength(df, "price", max_recommendations=2)
    assert len(result.recommendations) == 2
    assert [r.rank for r in result.recommendations] == [1, 2]
    assert any("exceed max_recommendations=2" in n for n in result.notes)


@pytest.mark.parametrize("bad", [-1, None])
def test_invalid_cap_handled_safely(df, bad):
    result = rank_visualizations_by_statistical_strength(df, "price", max_recommendations=bad)  # type: ignore[arg-type]
    assert result.status is VisualizationStatisticalStrengthStatus.RANKED
    assert result.notes


# --- safety --------------------------------------------


def test_input_dataframe_unchanged(df):
    before = df.copy(deep=True)
    rank_visualizations_by_statistical_strength(df, "price")
    rank_visualizations_by_statistical_strength(df, "grade")
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rank_visualizations_by_statistical_strength(df, "price")
    assert list(tmp_path.iterdir()) == []


def test_visualization_analysis_unchanged(df):
    before = analyze_visualizations(df).model_dump()
    rank_visualizations_by_statistical_strength(df, "price")
    assert analyze_visualizations(df).model_dump() == before


def test_eda_report_visualizations_unchanged(df):
    before = analyze_dataframe(df).visualizations.model_dump()
    rank_visualizations_by_statistical_strength(df, "price")
    assert analyze_dataframe(df).visualizations.model_dump() == before


def test_recommend_visualizations_output_unchanged(df):
    before = recommend_visualizations(df, "price").model_dump()
    rank_visualizations_by_statistical_strength(df, "price")
    after = recommend_visualizations(df, "price").model_dump()
    assert before == after
    # and the two layers use different score meanings
    assert "strength_score" not in str(before)


# --- integration / backward compat -------------------


def test_analyze_dataframe_signature_unchanged():
    import inspect

    sig = inspect.signature(analyze_dataframe)
    assert list(sig.parameters) == ["df", "dataset_id", "dataset_version_id"]


def test_existing_eda_sections_still_populated(df):
    report = analyze_dataframe(df)
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.statistical_tests.anova
    assert report.effect_sizes.correlation_ratio
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.visualizations.scatter_plots
    assert report.visualization_recommendations.recommendations == []


def test_plotly_and_matplotlib_rendering_still_work(df):
    import plotly.graph_objects as go
    from matplotlib.figure import Figure

    spec = analyze_visualizations(df).scatter_plots[0]
    assert isinstance(render_visualization(df, spec), Figure)
    assert isinstance(render_plotly_visualization(df, spec), go.Figure)


def test_analyze_dataframe_default_strength_section(df):
    section = analyze_dataframe(df).visualization_statistical_strength
    assert section.status is VisualizationStatisticalStrengthStatus.UNAVAILABLE
    assert section.target_column is None
    assert section.recommendations == []


def test_old_eda_json_without_new_field_still_validates(df):
    payload = analyze_dataframe(df).model_dump(mode="json")
    payload.pop("visualization_statistical_strength")
    restored = EDAReport.model_validate(payload)
    assert restored.visualization_statistical_strength.status is (
        VisualizationStatisticalStrengthStatus.UNAVAILABLE
    )


def test_model_copy_merge_pattern(df):
    eda_report = analyze_dataframe(df)
    merged = eda_report.model_copy(
        update={
            "visualization_statistical_strength": rank_visualizations_by_statistical_strength(
                df, "price"
            )
        }
    )
    assert merged.visualization_statistical_strength.status is (
        VisualizationStatisticalStrengthStatus.RANKED
    )
    assert eda_report.visualization_statistical_strength.recommendations == []


def test_result_is_a_visualization_statistical_strength_list(df):
    result = rank_visualizations_by_statistical_strength(df, "price")
    assert all(isinstance(r, VisualizationStatisticalStrength) for r in result.recommendations)
