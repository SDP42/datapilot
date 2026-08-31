"""Phase 4 completion — the three final layers do not disturb anything else."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from data_engine import quality
from data_engine.eda import (
    EDAReport,
    analyze_dataframe,
    analyze_distribution,
    analyze_effect_sizes,
    analyze_nonparametric,
    analyze_statistics,
    analyze_visualizations,
    correct_multiple_testing,
    cross_reference_eda_quality,
    estimate_mutual_information_datetime,
    estimate_mutual_information_knn,
    export_visualization,
    friedman_test,
    rank_visualizations_by_statistical_strength,
    recommend_visualizations,
    render_plotly_visualization,
    render_visualization,
    sign_test,
    wilcoxon_signed_rank,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 90
    base = np.linspace(0.0, 9.0, n)
    return pd.DataFrame(
        {
            "t": pd.to_datetime(["2022-01-01"]).repeat(n) + pd.to_timedelta(np.arange(n), "h"),
            "x": base,
            "y": base * 1.5 + rng.normal(0.0, 0.3, n),
            "g": (["a", "b", "c"] * (n // 3)),
        }
    )


def _run_new_layers(frame: pd.DataFrame) -> None:
    estimate_mutual_information_datetime(frame, "t", "y")
    wilcoxon_signed_rank(frame["x"], frame["y"])
    sign_test(frame["x"], frame["y"])
    friedman_test(frame["x"], frame["y"], frame["x"] + 1.0)
    correct_multiple_testing([0.01, 0.2, 0.9])


def test_analyze_dataframe_signature_unchanged():
    assert list(inspect.signature(analyze_dataframe).parameters) == [
        "df",
        "dataset_id",
        "dataset_version_id",
    ]


def test_all_existing_eda_sections_still_populated(frame):
    report = analyze_dataframe(frame[["x", "y", "g"]])
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.statistical_tests.t_tests
    assert report.effect_sizes.mutual_information
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.visualizations.scatter_plots
    # the standalone/defaulted sections are untouched by analyze_dataframe
    assert report.quality_cross_reference.entries == []
    assert report.visualization_recommendations.recommendations == []
    assert report.visualization_statistical_strength.recommendations == []


def test_no_new_eda_report_field_was_added(frame):
    report = analyze_dataframe(frame[["x", "y"]])
    for missing in (
        "datetime_mutual_information",
        "paired_nonparametric_tests",
        "multiple_testing_correction",
        "wilcoxon",
        "friedman",
    ):
        assert not hasattr(report, missing)


def test_old_eda_report_json_still_validates(frame):
    payload = analyze_dataframe(frame[["x", "y", "g"]]).model_dump(mode="json")
    for optional in (
        "visualization_statistical_strength",
        "visualization_recommendations",
        "quality_cross_reference",
        "distribution",
    ):
        payload.pop(optional, None)
    restored = EDAReport.model_validate(payload)
    assert restored.n_rows == 90


def test_new_layers_do_not_change_existing_analysis_outputs(frame):
    small = frame[["x", "y", "g"]]
    before = {
        "statistics": analyze_statistics(small).model_dump(),
        "effects": analyze_effect_sizes(small).model_dump(),
        "nonparametric": analyze_nonparametric(small).model_dump(),
        "distribution": analyze_distribution(small).model_dump(),
        "visualizations": analyze_visualizations(small).model_dump(),
        "recommend": recommend_visualizations(small, "y").model_dump(),
        "strength": rank_visualizations_by_statistical_strength(small, "y").model_dump(),
    }
    _run_new_layers(frame)
    assert analyze_statistics(small).model_dump() == before["statistics"]
    assert analyze_effect_sizes(small).model_dump() == before["effects"]
    assert analyze_nonparametric(small).model_dump() == before["nonparametric"]
    assert analyze_distribution(small).model_dump() == before["distribution"]
    assert analyze_visualizations(small).model_dump() == before["visualizations"]
    assert recommend_visualizations(small, "y").model_dump() == before["recommend"]
    assert (
        rank_visualizations_by_statistical_strength(small, "y").model_dump() == before["strength"]
    )


def test_existing_knn_mi_and_cross_reference_unchanged(frame):
    knn_before = estimate_mutual_information_knn(frame, "x", "y").model_dump()
    eda_report = analyze_dataframe(frame[["x", "y", "g"]])
    quality_report = quality.analyze_dataframe(frame[["x", "y", "g"]])
    xref_before = cross_reference_eda_quality(eda_report, quality_report).model_dump()
    _run_new_layers(frame)
    assert estimate_mutual_information_knn(frame, "x", "y").model_dump() == knn_before
    assert cross_reference_eda_quality(eda_report, quality_report).model_dump() == xref_before


def test_rendering_and_export_still_work(frame, tmp_path):
    spec = analyze_visualizations(frame[["x", "y", "g"]]).scatter_plots[0]
    assert isinstance(render_visualization(frame, spec), Figure)
    plotly_fig = render_plotly_visualization(frame, spec)
    assert isinstance(plotly_fig, go.Figure)
    target = tmp_path / "chart.html"
    assert export_visualization(plotly_fig, target) == target
    assert target.is_file()


def test_input_frame_unchanged_by_all_new_layers(frame):
    before = frame.copy(deep=True)
    _run_new_layers(frame)
    pd.testing.assert_frame_equal(frame, before)
