"""Phase 4 — deterministic target-aware visualization recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.eda import (
    EDAReport,
    VisualizationRecommendation,
    VisualizationRecommendationAnalysis,
    VisualizationRecommendationKind,
    VisualizationRecommendationStatus,
    analyze_dataframe,
    recommend_visualizations,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "size": [12.0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "grade": list("AABBCCAABBCC"),
            "region": ["north", "south"] * 6,
            "when": pd.to_datetime(["2021-01-01"] * 12),
        }
    )


# --- correctness -------------------------------------------------


def test_numeric_target_recommends_scatter_involving_target(df):
    result = recommend_visualizations(df, "price")
    assert result.status is VisualizationRecommendationStatus.RECOMMENDED
    scatters = [
        r for r in result.recommendations if r.kind is VisualizationRecommendationKind.SCATTER_PLOT
    ]
    assert scatters
    assert all("price" in r.columns for r in scatters)
    assert scatters[0].score == 90.0


def test_categorical_target_recommends_box_plots_involving_target(df):
    result = recommend_visualizations(df, "grade")
    boxes = [
        r for r in result.recommendations if r.kind is VisualizationRecommendationKind.BOX_PLOT
    ]
    assert boxes
    assert all("grade" in r.columns for r in boxes)
    assert boxes[0].score == 90.0


def test_numeric_target_histogram_is_recommended(df):
    result = recommend_visualizations(df, "price")
    hist = [
        r for r in result.recommendations if r.kind is VisualizationRecommendationKind.HISTOGRAM
    ]
    assert any(r.columns == ["price"] for r in hist)


def test_categorical_target_bar_chart_is_recommended(df):
    result = recommend_visualizations(df, "grade")
    bars = [
        r for r in result.recommendations if r.kind is VisualizationRecommendationKind.BAR_CHART
    ]
    assert [r.columns for r in bars] == [["grade"]]


def test_ranks_are_unique_and_sequential(df):
    result = recommend_visualizations(df, "price", max_recommendations=100)
    ranks = [r.rank for r in result.recommendations]
    assert ranks == list(range(1, len(ranks) + 1))
    assert len(set(ranks)) == len(ranks)


def test_scores_are_deterministic_and_ordered(df):
    a = recommend_visualizations(df, "grade")
    b = recommend_visualizations(df, "grade")
    assert [r.score for r in a.recommendations] == [r.score for r in b.recommendations]
    scores = [r.score for r in a.recommendations]
    assert scores == sorted(scores, reverse=True)


def test_tie_break_is_kind_then_columns(df):
    result = recommend_visualizations(df, "grade")
    tied = [(r.kind.value, tuple(r.columns)) for r in result.recommendations if r.score == 90.0]
    assert tied == sorted(tied)


def test_source_spec_pointer_round_trips_to_the_real_spec(df):
    eda = analyze_dataframe(df)
    result = recommend_visualizations(df, "price")
    for rec in result.recommendations:
        spec = getattr(eda.visualizations, rec.source_family)[rec.source_index]
        assert spec.columns == rec.columns
        assert spec.kind.value == rec.kind.value


# --- target handling -------------------------------------------


def test_missing_target_column_is_unavailable(df):
    result = recommend_visualizations(df, "nope")
    assert result.status is VisualizationRecommendationStatus.UNAVAILABLE
    assert "not in the DataFrame" in result.reason
    assert result.recommendations == []


def test_datetime_target_is_unavailable(df):
    result = recommend_visualizations(df, "when")
    assert result.status is VisualizationRecommendationStatus.UNAVAILABLE
    assert "datetime" in result.reason


def test_all_missing_target_is_unavailable(df):
    frame = df.assign(empty=[np.nan] * len(df))
    result = recommend_visualizations(frame, "empty")
    assert result.status is VisualizationRecommendationStatus.UNAVAILABLE
    assert "no non-null observations" in result.reason


def test_high_cardinality_categorical_target_is_unavailable():
    frame = pd.DataFrame({"id": [f"v{i}" for i in range(120)], "n": range(120)})
    result = recommend_visualizations(frame, "id")
    assert result.status is VisualizationRecommendationStatus.UNAVAILABLE
    assert "cardinality" in result.reason


def test_target_is_a_required_positional_argument():
    with pytest.raises(TypeError):
        recommend_visualizations(pd.DataFrame({"a": [1, 2, 3]}))  # type: ignore[call-arg]


def test_no_target_inference_two_targets_two_results(df):
    by_price = recommend_visualizations(df, "price")
    by_grade = recommend_visualizations(df, "grade")
    assert by_price.target_column == "price"
    assert by_grade.target_column == "grade"
    assert by_price.model_dump() != by_grade.model_dump()


# --- determinism ----------------------------------------------


def test_row_shuffle_produces_identical_dump(df):
    a = recommend_visualizations(df, "price")
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = recommend_visualizations(shuffled, "price")
    assert a.model_dump() == b.model_dump()


def test_repeated_calls_are_identical(df):
    dumps = {recommend_visualizations(df, "grade").model_dump_json() for _ in range(5)}
    assert len(dumps) == 1


def test_recommendation_cap_is_deterministic(df):
    result = recommend_visualizations(df, "price", max_recommendations=2)
    assert len(result.recommendations) == 2
    assert [r.rank for r in result.recommendations] == [1, 2]
    assert any("exceed max_recommendations=2" in n for n in result.notes)


@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_cap_is_handled_safely(df, bad):
    result = recommend_visualizations(df, "price", max_recommendations=bad)
    assert result.status is VisualizationRecommendationStatus.RECOMMENDED
    assert result.recommendations == []
    assert any("negative" in n for n in result.notes)


def test_non_int_cap_is_handled_safely(df):
    result = recommend_visualizations(df, "price", max_recommendations=None)  # type: ignore[arg-type]
    assert result.status is VisualizationRecommendationStatus.RECOMMENDED
    assert any("not an int" in n for n in result.notes)
    assert len(result.recommendations) <= 10


# --- safety --------------------------------------------------


def test_input_dataframe_unchanged(df):
    before = df.copy(deep=True)
    recommend_visualizations(df, "price")
    recommend_visualizations(df, "grade")
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recommend_visualizations(df, "price")
    assert list(tmp_path.iterdir()) == []


def test_no_dataset_version_or_lineage_object_touched(df):
    # the function only takes a DataFrame + a string; it cannot reach a
    # DatasetVersion, a store, or lineage. Assert the return type is a
    # plain analysis with no side-channel.
    result = recommend_visualizations(df, "price")
    assert isinstance(result, VisualizationRecommendationAnalysis)


# --- serialization ------------------------------------------


def test_recommendation_json_round_trip():
    rec = VisualizationRecommendation(
        kind=VisualizationRecommendationKind.SCATTER_PLOT,
        columns=["a", "b"],
        rank=1,
        score=90.0,
        reason="scatter plot of 'b' against the numeric target 'a'",
        target_column="a",
        source_family="scatter_plots",
        source_index=0,
    )
    assert VisualizationRecommendation.model_validate_json(rec.model_dump_json()) == rec


def test_analysis_json_round_trip(df):
    result = recommend_visualizations(df, "grade")
    dumped = result.model_dump_json()
    assert VisualizationRecommendationAnalysis.model_validate_json(dumped) == result


def test_old_eda_json_without_recommendations_still_validates(df):
    payload = analyze_dataframe(df).model_dump(mode="json")
    payload.pop("visualization_recommendations")
    restored = EDAReport.model_validate(payload)
    assert restored.visualization_recommendations.status is (
        VisualizationRecommendationStatus.UNAVAILABLE
    )
    assert restored.visualization_recommendations.recommendations == []


# --- integration ------------------------------------------


def test_analyze_dataframe_leaves_recommendations_default(df):
    report = analyze_dataframe(df)
    section = report.visualization_recommendations
    assert section.status is VisualizationRecommendationStatus.UNAVAILABLE
    assert section.target_column is None
    assert section.recommendations == []


def test_analyze_dataframe_still_populates_all_prior_sections(df):
    report = analyze_dataframe(df)
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.statistical_tests.t_tests
    assert report.effect_sizes.mutual_information
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.visualizations.histograms
    # visualizations section is untouched by the recommendation layer
    before = report.visualizations.model_dump()
    _ = recommend_visualizations(df, "price")
    assert analyze_dataframe(df).visualizations.model_dump() == before


def test_explicit_merge_pattern(df):
    eda = analyze_dataframe(df)
    merged = eda.model_copy(
        update={"visualization_recommendations": recommend_visualizations(df, "price")}
    )
    assert merged.visualization_recommendations.status is (
        VisualizationRecommendationStatus.RECOMMENDED
    )
    assert eda.visualization_recommendations.recommendations == []  # original untouched
