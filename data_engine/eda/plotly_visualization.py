"""Plotly rendering + deterministic export for existing ``VisualizationSpec``.

An **additional** rendering backend alongside the Matplotlib path in
:mod:`data_engine.eda.visualization`. Both backends consume the same
:class:`VisualizationSpec`; chart *selection* (``analyze_visualizations``)
is unchanged and backend-independent, and the target-aware recommendation
layer is untouched.

* :func:`render_plotly_visualization` — ``(df, spec) -> plotly.graph_objects.Figure``
* :func:`export_visualization` — write an already-rendered Plotly figure
  to an explicit path (HTML always; PNG / SVG / PDF when ``kaleido`` is
  installed).

Nothing here is target-aware, ranks nothing, uses no randomness/sampling,
and stores no ``Figure`` in any Pydantic model or in ``EDAReport``.

    selection  -> VisualizationSpec   (analyze_visualizations)
    rendering  -> Figure              (render_visualization / render_plotly_visualization)
    export     -> file                (export_visualization)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .distribution import sturges_bin_count
from .visualization import _finite, _ordered_value_counts
from .visualization_models import (
    VisualizationKind,
    VisualizationSpec,
    VisualizationStatus,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go


class PlotlyVisualizationError(RuntimeError):
    """A ``VisualizationSpec`` cannot be rendered with Plotly, or an export
    request is invalid / unsupported."""


# HTML needs no extra tooling; the static formats need ``kaleido``.
HTML_EXPORT_FORMAT = "html"
STATIC_EXPORT_FORMATS = ("png", "svg", "pdf")
SUPPORTED_EXPORT_FORMATS = (HTML_EXPORT_FORMAT, *STATIC_EXPORT_FORMATS)


# --------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------


def _require_available(spec: VisualizationSpec) -> None:
    if spec.status is not VisualizationStatus.AVAILABLE:
        raise PlotlyVisualizationError(
            f"cannot render an unavailable {spec.kind.value}: {spec.reason or 'no reason given'}"
        )


def _require_metadata(spec: VisualizationSpec, *keys: str) -> None:
    missing = [key for key in keys if key not in spec.metadata]
    if missing:
        raise PlotlyVisualizationError(
            f"{spec.kind.value} spec is missing required metadata: {', '.join(missing)}"
        )


def _require_columns(df: pd.DataFrame, *columns: str) -> None:
    present = {str(c) for c in df.columns}
    absent = [c for c in columns if c not in present]
    if absent:
        raise PlotlyVisualizationError(f"column(s) not in the DataFrame: {', '.join(absent)}")


def _layout(fig: go.Figure, spec: VisualizationSpec, x_default: str, y_default: str) -> None:
    fig.update_layout(
        title=spec.title,
        xaxis_title=spec.x_label or x_default,
        yaxis_title=spec.y_label or y_default,
    )


def _plotly_histogram(df: pd.DataFrame, spec: VisualizationSpec, fig: go.Figure, go: Any) -> None:
    _require_metadata(spec, "value_column")
    col = str(spec.metadata["value_column"])
    _require_columns(df, col)
    finite = _finite(df[col])
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        raise PlotlyVisualizationError(f"'{col}' has no non-constant finite data to plot")
    # Same deterministic binning as the Matplotlib backend / distribution layer:
    # numpy histogram over the shared Sturges bin count.
    n_bins = sturges_bin_count(int(finite.size))
    counts, edges = np.histogram(finite, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    fig.add_trace(
        go.Bar(
            x=[round(float(c), 10) for c in centers],
            y=[int(v) for v in counts],
            width=[round(float(w), 10) for w in widths],
            name=col,
        )
    )
    fig.update_layout(bargap=0.0)
    _layout(fig, spec, col, "Frequency")


def _plotly_bar_chart(df: pd.DataFrame, spec: VisualizationSpec, fig: go.Figure, go: Any) -> None:
    _require_metadata(spec, "category_column")
    col = str(spec.metadata["category_column"])
    _require_columns(df, col)
    ordered = _ordered_value_counts(df[col])
    if not ordered:
        raise PlotlyVisualizationError(f"'{col}' has no non-null data to plot")
    labels = [label for label, _ in ordered]
    heights = [count for _, count in ordered]
    fig.add_trace(go.Bar(x=labels, y=heights, name=col))
    # Freeze the deterministic (-count, value) order; never let Plotly reorder.
    fig.update_xaxes(categoryorder="array", categoryarray=labels)
    _layout(fig, spec, col, "Count")


def _plotly_scatter_plot(
    df: pd.DataFrame, spec: VisualizationSpec, fig: go.Figure, go: Any
) -> None:
    _require_metadata(spec, "x_column", "y_column")
    x_col = str(spec.metadata["x_column"])
    y_col = str(spec.metadata["y_column"])
    _require_columns(df, x_col, y_col)
    paired = df[[x_col, y_col]].dropna()
    x = paired[x_col].to_numpy(dtype=float)
    y = paired[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        raise PlotlyVisualizationError(f"no rows where both '{x_col}' and '{y_col}' are finite")
    fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name=f"{x_col} vs {y_col}"))
    _layout(fig, spec, x_col, y_col)


def _plotly_box_plot(df: pd.DataFrame, spec: VisualizationSpec, fig: go.Figure, go: Any) -> None:
    _require_metadata(spec, "category_column", "value_column")
    cat = str(spec.metadata["category_column"])
    num = str(spec.metadata["value_column"])
    _require_columns(df, cat, num)
    grouped = df[[cat, num]].dropna(subset=[cat])
    label_col = grouped[cat].astype("object").astype(str)
    used: list[str] = []
    for category in sorted(str(v) for v in grouped[cat].astype("object").unique()):
        values = _finite(grouped.loc[label_col == category, num])
        if values.size:
            fig.add_trace(go.Box(y=values, name=category))
            used.append(category)
    if not used:
        raise PlotlyVisualizationError(f"no category of '{cat}' has a finite '{num}' observation")
    fig.update_xaxes(categoryorder="array", categoryarray=used)
    _layout(fig, spec, cat, num)


_RENDERERS = {
    VisualizationKind.HISTOGRAM: _plotly_histogram,
    VisualizationKind.BAR_CHART: _plotly_bar_chart,
    VisualizationKind.SCATTER_PLOT: _plotly_scatter_plot,
    VisualizationKind.BOX_PLOT: _plotly_box_plot,
}


def render_plotly_visualization(df: pd.DataFrame, spec: VisualizationSpec) -> go.Figure:
    """Render one *available* :class:`VisualizationSpec` to an in-memory
    ``plotly.graph_objects.Figure``.

    * Consumes the existing spec — performs no chart selection, no target
      inference, no ranking.
    * ``df`` is never modified; missing / non-finite values are excluded,
      never filled or invented; rows are not sorted or sampled.
    * An unavailable spec, an unknown ``kind``, missing metadata, an
      absent column, or data that has become unplottable raises
      :class:`PlotlyVisualizationError` — a misleading empty figure is
      never returned.
    * Same four chart kinds as the Matplotlib backend; the histogram uses
      the shared :func:`sturges_bin_count`.
    """
    _require_available(spec)
    renderer = _RENDERERS.get(spec.kind)
    if renderer is None:  # pragma: no cover - guarded by the enum
        raise PlotlyVisualizationError(f"unsupported visualization kind: {spec.kind!r}")

    import plotly.graph_objects as go

    figure = go.Figure()
    renderer(df, spec, figure, go)
    return figure


# --------------------------------------------------------------------
# export
# --------------------------------------------------------------------


def _resolve_format(output_path: Path, explicit: str | None) -> str:
    if explicit is not None:
        fmt = explicit.strip().lower().lstrip(".")
    else:
        fmt = output_path.suffix.lower().lstrip(".")
    if not fmt:
        raise PlotlyVisualizationError(
            "no export format: pass format=... or give output_path a file extension"
        )
    if fmt not in SUPPORTED_EXPORT_FORMATS:
        raise PlotlyVisualizationError(
            f"unsupported export format {fmt!r}; supported: {', '.join(SUPPORTED_EXPORT_FORMATS)}"
        )
    return fmt


def export_visualization(
    figure: go.Figure,
    output_path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write an already-rendered Plotly figure to ``output_path``.

    * Only ``plotly.graph_objects.Figure`` is accepted — never a
      Matplotlib figure.
    * The format is taken from ``format`` if given, else from the
      ``output_path`` extension. Supported: ``html`` (no extra tooling),
      ``png`` / ``svg`` / ``pdf`` (require the optional ``kaleido``
      package). There is no fallback between formats.
    * Writes **only** to ``output_path``. The parent directory must
      already exist — it is never created implicitly. No location is ever
      chosen implicitly.
    * Refuses to overwrite an existing file unless ``overwrite=True``.
    * The figure, the DataFrame, and any spec are not modified.

    Returns the :class:`~pathlib.Path` that was written.
    """
    path = Path(output_path)
    fmt = _resolve_format(path, format)

    import plotly.graph_objects as go

    if not isinstance(figure, go.Figure):
        raise PlotlyVisualizationError(
            "export_visualization only accepts a plotly.graph_objects.Figure"
        )

    if path.exists() and not overwrite:
        raise PlotlyVisualizationError(
            f"refusing to overwrite existing file {path}; pass overwrite=True to replace it"
        )
    if not path.parent.exists():
        raise PlotlyVisualizationError(
            f"parent directory {path.parent} does not exist; it is not created implicitly"
        )

    if fmt == HTML_EXPORT_FORMAT:
        figure.write_html(str(path))
    else:
        try:
            figure.write_image(str(path), format=fmt)
        except Exception as exc:
            raise PlotlyVisualizationError(
                f"static image export to {fmt!r} failed. PNG/SVG/PDF export needs the "
                f"optional 'kaleido' package (pip install kaleido). Underlying error: {exc}"
            ) from exc

    return path
