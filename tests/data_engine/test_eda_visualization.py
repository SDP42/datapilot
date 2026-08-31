"""Phase 4 — deterministic EDA visualization foundation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from data_engine.eda import (
    EDAReport,
    VisualizationAnalysis,
    VisualizationError,
    VisualizationKind,
    VisualizationSpec,
    VisualizationStatus,
    analyze_dataframe,
    analyze_visualizations,
    render_visualization,
)
from data_engine.eda.visualization_models import (
    MAX_BAR_CHARTS,
    MAX_HISTOGRAMS,
    MAX_SCATTER_PLOTS,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "b_num": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
            "a_num": [10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "const": [5.0] * 11,
            "cat": list("xxxxyyyzzz") + ["x"],
            "empty": [np.nan] * 11,
        }
    )


def _spec(analysis: VisualizationAnalysis, family: str, **match) -> VisualizationSpec:
    for s in getattr(analysis, family):
        if all(getattr(s, k) == v for k, v in match.items()):
            return s
    raise AssertionError(f"no {family} spec matching {match}")


# --- model round trips --------------------------------------------


def test_visualization_spec_json_round_trip():
    spec = VisualizationSpec(
        kind=VisualizationKind.HISTOGRAM,
        title="Histogram of x",
        columns=["x"],
        status=VisualizationStatus.AVAILABLE,
        x_label="x",
        y_label="Frequency",
        metadata={"value_column": "x", "n_bins": 4, "categories": ["a", "b"]},
    )
    dumped = spec.model_dump_json()
    assert VisualizationSpec.model_validate_json(dumped) == spec


def test_visualization_analysis_json_round_trip(df):
    analysis = analyze_visualizations(df)
    dumped = analysis.model_dump_json()
    assert VisualizationAnalysis.model_validate_json(dumped) == analysis


# --- selection ----------------------------------------------------


def test_numeric_column_selects_histogram(df):
    spec = _spec(analyze_visualizations(df), "histograms", columns=["a_num"])
    assert spec.kind is VisualizationKind.HISTOGRAM
    assert spec.status is VisualizationStatus.AVAILABLE
    assert spec.metadata["bin_rule"] == "sturges"


def test_categorical_column_selects_bar_chart(df):
    spec = _spec(analyze_visualizations(df), "bar_charts", columns=["cat"])
    assert spec.kind is VisualizationKind.BAR_CHART
    assert spec.status is VisualizationStatus.AVAILABLE
    assert spec.metadata["categories"] == ["x", "y", "z"]  # (-count, value) order
    assert spec.metadata["counts"] == [5, 3, 3]


def test_numeric_pair_selects_scatter_plot(df):
    spec = _spec(analyze_visualizations(df), "scatter_plots", columns=["a_num", "b_num"])
    assert spec.kind is VisualizationKind.SCATTER_PLOT
    assert spec.metadata["x_column"] == "a_num"
    assert spec.metadata["y_column"] == "b_num"
    assert spec.metadata["n_observations"] == 10  # one NaN row dropped


def test_categorical_numeric_pair_selects_box_plot(df):
    spec = _spec(analyze_visualizations(df), "box_plots", columns=["cat", "a_num"])
    assert spec.kind is VisualizationKind.BOX_PLOT
    assert spec.metadata["categories"] == ["x", "y", "z"]  # category_asc


def test_deterministic_ordering(df):
    a = analyze_visualizations(df)
    b = analyze_visualizations(df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    assert a.model_dump() == b.model_dump()
    hist_cols = [s.columns[0] for s in a.histograms]
    assert hist_cols == sorted(hist_cols)
    scatter_pairs = [tuple(s.columns) for s in a.scatter_plots]
    assert scatter_pairs == sorted(scatter_pairs)


def test_histogram_cap_and_truncation_note():
    wide = pd.DataFrame({f"c{i:03d}": [float(i), float(i + 1), float(i + 2)] for i in range(70)})
    analysis = analyze_visualizations(wide)
    assert len(analysis.histograms) == MAX_HISTOGRAMS
    assert any("histograms" in n and "cap" in n for n in analysis.notes)


def test_scatter_cap_is_enforced():
    wide = pd.DataFrame({f"c{i:02d}": np.linspace(i, i + 1, 8) for i in range(12)})
    analysis = analyze_visualizations(wide)
    assert len(analysis.scatter_plots) == MAX_SCATTER_PLOTS
    assert any("scatter_plots" in n for n in analysis.notes)


def test_high_cardinality_categorical_excluded():
    df = pd.DataFrame({"id": [f"v{i}" for i in range(200)], "n": range(200)})
    analysis = analyze_visualizations(df)
    assert not any(s.columns == ["id"] for s in analysis.bar_charts)
    assert any("cardinality" in note for note in analysis.notes)


def test_caps_match_documented_constants():
    assert (MAX_HISTOGRAMS, MAX_BAR_CHARTS, MAX_SCATTER_PLOTS) == (50, 50, 50)


# --- degenerate selection ---------------------------------------


def test_constant_numeric_histogram_is_unavailable(df):
    spec = _spec(analyze_visualizations(df), "histograms", columns=["const"])
    assert spec.status is VisualizationStatus.UNAVAILABLE
    assert "constant" in spec.reason


def test_all_missing_numeric_histogram_is_unavailable(df):
    spec = _spec(analyze_visualizations(df), "histograms", columns=["empty"])
    assert spec.status is VisualizationStatus.UNAVAILABLE
    assert spec.reason == "no finite observations"


# --- rendering --------------------------------------------------


def test_render_histogram_returns_figure(df):
    fig = render_visualization(
        df, _spec(analyze_visualizations(df), "histograms", columns=["a_num"])
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_render_bar_chart_returns_figure(df):
    fig = render_visualization(df, _spec(analyze_visualizations(df), "bar_charts", columns=["cat"]))
    assert isinstance(fig, Figure)


def test_render_scatter_plot_returns_figure(df):
    fig = render_visualization(
        df, _spec(analyze_visualizations(df), "scatter_plots", columns=["a_num", "b_num"])
    )
    assert isinstance(fig, Figure)


def test_render_box_plot_returns_figure(df):
    fig = render_visualization(
        df, _spec(analyze_visualizations(df), "box_plots", columns=["cat", "a_num"])
    )
    assert isinstance(fig, Figure)


def test_render_excludes_missing_values(df):
    spec = _spec(analyze_visualizations(df), "scatter_plots", columns=["a_num", "b_num"])
    fig = render_visualization(df, spec)
    offsets = fig.axes[0].collections[0].get_offsets()
    assert len(offsets) == 10  # the NaN row in b_num is excluded, not filled


def test_render_unavailable_spec_raises(df):
    spec = _spec(analyze_visualizations(df), "histograms", columns=["const"])
    with pytest.raises(VisualizationError):
        render_visualization(df, spec)


def test_render_histogram_uses_sturges_bin_count(df):
    spec = _spec(analyze_visualizations(df), "histograms", columns=["a_num"])
    fig = render_visualization(df, spec)
    # a_num has 11 finite values -> Sturges: ceil(log2(11)) + 1 == 5 bins == 5 patches
    assert len(fig.axes[0].patches) == spec.metadata["n_bins"] == 5


# --- safety ----------------------------------------------------


def test_input_dataframe_unchanged_by_selection_and_render(df):
    before = df.copy(deep=True)
    analysis = analyze_visualizations(df)
    for family in ("histograms", "bar_charts", "scatter_plots", "box_plots"):
        for spec in getattr(analysis, family):
            if spec.status is VisualizationStatus.AVAILABLE:
                render_visualization(df, spec)
    pd.testing.assert_frame_equal(df, before)


def test_no_files_written(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    analysis = analyze_visualizations(df)
    for family in ("histograms", "bar_charts", "scatter_plots", "box_plots"):
        for spec in getattr(analysis, family):
            if spec.status is VisualizationStatus.AVAILABLE:
                render_visualization(df, spec)
    assert list(tmp_path.iterdir()) == []


def test_selection_invents_no_values(df):
    spec = _spec(analyze_visualizations(df), "bar_charts", columns=["cat"])
    assert sum(spec.metadata["counts"]) == int(df["cat"].notna().sum())


# --- integration ---------------------------------------------


def test_analyze_dataframe_populates_visualizations(df):
    report = analyze_dataframe(df)
    assert isinstance(report.visualizations, VisualizationAnalysis)
    assert report.visualizations.histograms


def test_existing_eda_sections_still_populated(df):
    report = analyze_dataframe(df)
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.statistical_tests.t_tests
    assert report.effect_sizes.mutual_information
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.column_kinds  # unchanged classification


def test_old_eda_json_without_visualizations_still_validates(df):
    payload = analyze_dataframe(df).model_dump(mode="json")
    payload.pop("visualizations")
    payload.pop("distribution")
    payload.pop("quality_cross_reference")
    restored = EDAReport.model_validate(payload)
    assert restored.visualizations.histograms == []
    assert isinstance(restored.visualizations, VisualizationAnalysis)
