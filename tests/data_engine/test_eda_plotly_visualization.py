"""Phase 4 — Plotly rendering backend + deterministic chart export."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from data_engine import eda
from data_engine.eda import (
    PlotlyVisualizationError,
    analyze_dataframe,
    analyze_visualizations,
    export_visualization,
    render_plotly_visualization,
    render_visualization,
)
from data_engine.eda.distribution import sturges_bin_count
from data_engine.eda.visualization_models import (
    VisualizationKind,
    VisualizationSpec,
    VisualizationStatus,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "b_num": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.inf],
            "a_num": [10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1],
            "const": [5.0] * 12,
            "cat": list("xxxxyyyzzz") + ["x", None],
        }
    )


def _spec(df: pd.DataFrame, family: str, **match) -> VisualizationSpec:
    for spec in getattr(analyze_visualizations(df), family):
        if all(getattr(spec, k) == v for k, v in match.items()):
            return spec
    raise AssertionError(f"no {family} spec matching {match}")


# --- models / API -----------------------------------------------


def test_public_symbols_importable():
    assert eda.render_plotly_visualization is render_plotly_visualization
    assert eda.export_visualization is export_visualization
    assert issubclass(eda.PlotlyVisualizationError, RuntimeError)


def test_returns_plotly_figure(df):
    fig = render_plotly_visualization(df, _spec(df, "histograms", columns=["a_num"]))
    assert isinstance(fig, go.Figure)


# --- histogram --------------------------------------------------


def test_histogram_produces_figure(df):
    fig = render_plotly_visualization(df, _spec(df, "histograms", columns=["a_num"]))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_histogram_uses_shared_sturges_bin_count(df):
    spec = _spec(df, "histograms", columns=["a_num"])
    fig = render_plotly_visualization(df, spec)
    expected = sturges_bin_count(spec.metadata["n_observations"])
    assert len(fig.data[0].x) == expected == spec.metadata["n_bins"]


def test_histogram_excludes_missing_and_infinite(df):
    # b_num has one NaN and one +inf -> 10 finite values
    spec = _spec(df, "histograms", columns=["b_num"])
    assert spec.metadata["n_observations"] == 10
    fig = render_plotly_visualization(df, spec)
    assert sum(int(v) for v in fig.data[0].y) == 10


def test_constant_histogram_spec_is_unavailable_and_raises(df):
    spec = _spec(df, "histograms", columns=["const"])
    assert spec.status is VisualizationStatus.UNAVAILABLE
    with pytest.raises(PlotlyVisualizationError, match="constant"):
        render_plotly_visualization(df, spec)


def test_histogram_titles_and_labels_come_from_spec(df):
    spec = _spec(df, "histograms", columns=["a_num"])
    fig = render_plotly_visualization(df, spec)
    assert fig.layout.title.text == spec.title
    assert fig.layout.xaxis.title.text == spec.x_label
    assert fig.layout.yaxis.title.text == spec.y_label


# --- bar chart -------------------------------------------------


def test_bar_chart_produces_figure(df):
    fig = render_plotly_visualization(df, _spec(df, "bar_charts", columns=["cat"]))
    assert isinstance(fig, go.Figure)


def test_bar_chart_category_order_matches_spec(df):
    spec = _spec(df, "bar_charts", columns=["cat"])
    fig = render_plotly_visualization(df, spec)
    assert list(fig.data[0].x) == spec.metadata["categories"]
    assert fig.layout.xaxis.categoryorder == "array"
    assert list(fig.layout.xaxis.categoryarray) == spec.metadata["categories"]


def test_bar_chart_counts_match_spec_metadata(df):
    spec = _spec(df, "bar_charts", columns=["cat"])
    fig = render_plotly_visualization(df, spec)
    assert [int(v) for v in fig.data[0].y] == spec.metadata["counts"]


def test_bar_chart_excludes_null_categories(df):
    spec = _spec(df, "bar_charts", columns=["cat"])
    fig = render_plotly_visualization(df, spec)
    assert sum(int(v) for v in fig.data[0].y) == int(df["cat"].notna().sum())


# --- scatter -------------------------------------------------


def test_scatter_produces_figure(df):
    fig = render_plotly_visualization(df, _spec(df, "scatter_plots", columns=["a_num", "b_num"]))
    assert isinstance(fig, go.Figure)


def test_scatter_only_plots_rows_with_both_finite(df):
    spec = _spec(df, "scatter_plots", columns=["a_num", "b_num"])
    fig = render_plotly_visualization(df, spec)
    paired = df[["a_num", "b_num"]].to_numpy(dtype=float)
    expected = int(np.isfinite(paired).all(axis=1).sum())
    assert len(fig.data[0].x) == expected == spec.metadata["n_observations"]


def test_scatter_respects_x_and_y_columns(df):
    spec = _spec(df, "scatter_plots", columns=["a_num", "b_num"])
    fig = render_plotly_visualization(df, spec)
    assert fig.layout.xaxis.title.text == "a_num"
    assert fig.layout.yaxis.title.text == "b_num"


# --- box plot -----------------------------------------------


def test_box_plot_produces_figure(df):
    fig = render_plotly_visualization(df, _spec(df, "box_plots", columns=["cat", "a_num"]))
    assert isinstance(fig, go.Figure)


def test_box_plot_one_box_per_usable_category(df):
    spec = _spec(df, "box_plots", columns=["cat", "a_num"])
    fig = render_plotly_visualization(df, spec)
    assert [trace.name for trace in fig.data] == spec.metadata["categories"] == ["x", "y", "z"]


def test_box_plot_category_order_is_deterministic(df):
    spec = _spec(df, "box_plots", columns=["cat", "a_num"])
    fig = render_plotly_visualization(df, spec)
    assert list(fig.layout.xaxis.categoryarray) == sorted(fig.layout.xaxis.categoryarray)


def test_box_plot_excludes_missing_and_non_finite(df):
    spec = _spec(df, "box_plots", columns=["cat", "b_num"])
    fig = render_plotly_visualization(df, spec)
    total = sum(len(trace.y) for trace in fig.data)
    assert total == int(sum(spec.metadata["counts"]))


# --- invalid / unavailable --------------------------------


def test_unavailable_spec_raises(df):
    spec = VisualizationSpec(
        kind=VisualizationKind.HISTOGRAM,
        title="x",
        columns=["a_num"],
        status=VisualizationStatus.UNAVAILABLE,
        reason="no finite observations",
    )
    with pytest.raises(PlotlyVisualizationError, match="no finite observations"):
        render_plotly_visualization(df, spec)


def test_missing_required_metadata_raises(df):
    spec = VisualizationSpec(
        kind=VisualizationKind.SCATTER_PLOT,
        title="x",
        columns=["a_num", "b_num"],
        status=VisualizationStatus.AVAILABLE,
        metadata={},
    )
    with pytest.raises(PlotlyVisualizationError, match="missing required metadata"):
        render_plotly_visualization(df, spec)


def test_absent_column_raises(df):
    spec = VisualizationSpec(
        kind=VisualizationKind.HISTOGRAM,
        title="x",
        columns=["ghost"],
        status=VisualizationStatus.AVAILABLE,
        metadata={"value_column": "ghost"},
    )
    with pytest.raises(PlotlyVisualizationError, match="not in the DataFrame"):
        render_plotly_visualization(df, spec)


def test_no_usable_observations_raises(df):
    spec = VisualizationSpec(
        kind=VisualizationKind.HISTOGRAM,
        title="x",
        columns=["const"],
        status=VisualizationStatus.AVAILABLE,
        metadata={"value_column": "const"},
    )
    with pytest.raises(PlotlyVisualizationError):
        render_plotly_visualization(df, spec)


# --- determinism ------------------------------------------


def test_row_shuffle_does_not_change_figure(df):
    spec = _spec(df, "histograms", columns=["a_num"])
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    assert (
        render_plotly_visualization(df, spec).to_dict()
        == render_plotly_visualization(shuffled, spec).to_dict()
    )


def test_repeated_rendering_is_equivalent(df):
    spec = _spec(df, "box_plots", columns=["cat", "a_num"])
    dumps = {str(render_plotly_visualization(df, spec).to_dict()) for _ in range(4)}
    assert len(dumps) == 1


def test_plotly_category_order_is_frozen(df):
    spec = _spec(df, "bar_charts", columns=["cat"])
    fig = render_plotly_visualization(df, spec)
    # explicitly frozen, so a reversed-value DataFrame gives the same x order
    reordered = df.iloc[::-1].reset_index(drop=True)
    assert list(render_plotly_visualization(reordered, spec).data[0].x) == list(fig.data[0].x)


# --- safety ---------------------------------------------


def test_rendering_does_not_mutate_dataframe(df):
    before = df.copy(deep=True)
    for family in ("histograms", "bar_charts", "scatter_plots", "box_plots"):
        for spec in getattr(analyze_visualizations(df), family):
            if spec.status is VisualizationStatus.AVAILABLE:
                render_plotly_visualization(df, spec)
    pd.testing.assert_frame_equal(df, before)


def test_rendering_creates_no_files(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for family in ("histograms", "bar_charts", "scatter_plots", "box_plots"):
        for spec in getattr(analyze_visualizations(df), family):
            if spec.status is VisualizationStatus.AVAILABLE:
                render_plotly_visualization(df, spec)
    assert list(tmp_path.iterdir()) == []


def test_rendering_does_not_alter_visualization_analysis(df):
    before = analyze_visualizations(df).model_dump()
    render_plotly_visualization(df, _spec(df, "histograms", columns=["a_num"]))
    assert analyze_visualizations(df).model_dump() == before


def test_matplotlib_backend_still_works_and_is_separate(df):
    spec = _spec(df, "histograms", columns=["a_num"])
    from matplotlib.figure import Figure

    assert isinstance(render_visualization(df, spec), Figure)
    assert isinstance(render_plotly_visualization(df, spec), go.Figure)


# --- export -------------------------------------------


@pytest.fixture
def figure(df) -> go.Figure:
    return render_plotly_visualization(df, _spec(df, "scatter_plots", columns=["a_num", "b_num"]))


def test_html_export_creates_exactly_the_requested_file(figure, tmp_path):
    target = tmp_path / "chart.html"
    returned = export_visualization(figure, target)
    assert returned == target
    assert target.is_file()
    assert list(tmp_path.iterdir()) == [target]
    assert "<html" in target.read_text().lower()


def test_export_format_kwarg_overrides_extension(figure, tmp_path):
    target = tmp_path / "chart.out"
    export_visualization(figure, target, format="html")
    assert target.is_file()


def test_unsupported_format_raises(figure, tmp_path):
    with pytest.raises(PlotlyVisualizationError, match="unsupported export format"):
        export_visualization(figure, tmp_path / "chart.xlsx")
    with pytest.raises(PlotlyVisualizationError, match="unsupported export format"):
        export_visualization(figure, tmp_path / "chart", format="gif")


def test_missing_format_raises(figure, tmp_path):
    with pytest.raises(PlotlyVisualizationError, match="no export format"):
        export_visualization(figure, tmp_path / "chart")


def test_export_refuses_silent_overwrite(figure, tmp_path):
    target = tmp_path / "chart.html"
    export_visualization(figure, target)
    with pytest.raises(PlotlyVisualizationError, match="refusing to overwrite"):
        export_visualization(figure, target)
    export_visualization(figure, target, overwrite=True)  # explicit opt-in works


def test_export_does_not_create_parent_directories(figure, tmp_path):
    with pytest.raises(PlotlyVisualizationError, match="parent directory"):
        export_visualization(figure, tmp_path / "sub" / "chart.html")


def test_export_writes_only_to_requested_path(figure, tmp_path):
    target = tmp_path / "only.html"
    export_visualization(figure, target)
    assert {p.name for p in tmp_path.iterdir()} == {"only.html"}


def test_export_does_not_mutate_figure_or_data(figure, df, tmp_path):
    fig_before = figure.to_dict()
    df_before = df.copy(deep=True)
    export_visualization(figure, tmp_path / "c.html")
    assert figure.to_dict() == fig_before
    pd.testing.assert_frame_equal(df, df_before)


def test_export_rejects_non_plotly_figure(tmp_path):
    from matplotlib.figure import Figure

    with pytest.raises(PlotlyVisualizationError, match="plotly"):
        export_visualization(Figure(), tmp_path / "c.html")


def test_static_image_export_png(figure, tmp_path):
    pytest.importorskip("kaleido")
    target = tmp_path / "chart.png"
    export_visualization(figure, target)
    assert target.is_file()
    assert target.stat().st_size > 0


# --- integration -------------------------------------


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "size": [12.0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "grade": list("AABBCCAABBCC"),
        }
    )


def test_analyze_dataframe_sections_unaffected(clean_df):
    report = analyze_dataframe(clean_df)
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.statistical_tests.t_tests
    assert report.effect_sizes.mutual_information
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.visualizations.histograms
    assert report.quality_cross_reference.entries == []
    assert report.visualization_recommendations.recommendations == []
    # no new EDAReport field was added for Plotly
    assert not hasattr(report, "plotly_visualizations")


def test_plotly_render_does_not_change_eda_report_visualizations(clean_df):
    before = analyze_dataframe(clean_df).visualizations.model_dump()
    render_plotly_visualization(clean_df, _spec(clean_df, "histograms", columns=["price"]))
    assert analyze_dataframe(clean_df).visualizations.model_dump() == before
