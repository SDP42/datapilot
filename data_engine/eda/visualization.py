"""Deterministic EDA visualization foundation.

Two clearly separated responsibilities:

* :func:`analyze_visualizations` — **selection**. Chooses which charts to
  describe, using only the DataFrame's structure and values. Pure,
  deterministic, no rendering, no Matplotlib import needed.
* :func:`render_visualization` — **rendering**. Turns one *available*
  :class:`VisualizationSpec` into an in-memory ``matplotlib.figure.Figure``.
  Nothing is written to disk; the input DataFrame is never modified;
  missing values are excluded, never filled.

Supported chart kinds (exactly four): histogram, bar chart, scatter plot,
box plot. No Plotly. No target-aware recommendation. No dashboard / API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .distribution import sturges_bin_count
from .models import EDAColumnKind
from .univariate import classify_columns
from .visualization_models import (
    MAX_BAR_CHARTS,
    MAX_BOX_PLOTS,
    MAX_HISTOGRAMS,
    MAX_SCATTER_PLOTS,
    MAX_VISUALIZATION_CATEGORIES,
    VisualizationAnalysis,
    VisualizationKind,
    VisualizationSpec,
    VisualizationStatus,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class VisualizationError(RuntimeError):
    """Raised when a :class:`VisualizationSpec` cannot be rendered."""


# --------------------------------------------------------------------
# selection helpers
# --------------------------------------------------------------------


def _finite(series: pd.Series) -> np.ndarray:
    values = series.dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _ordered_value_counts(series: pd.Series) -> list[tuple[str, int]]:
    """Category -> count, ordered by ``(-count, value)`` — the same
    deterministic ordering the univariate layer uses for top values."""
    counts = series.dropna().value_counts()
    return sorted(
        ((str(v), int(c)) for v, c in counts.items()),
        key=lambda kv: (-kv[1], kv[0]),
    )


def _truncate(
    candidates: list[tuple[str, ...]], cap: int, family: str, notes: list[str]
) -> list[tuple[str, ...]]:
    if len(candidates) > cap:
        notes.append(
            f"{family}: {len(candidates)} candidates exceed the cap of {cap}; "
            "kept the first (deterministic order)"
        )
        return candidates[:cap]
    return candidates


def _unavailable(
    kind: VisualizationKind, title: str, columns: list[str], reason: str
) -> VisualizationSpec:
    return VisualizationSpec(
        kind=kind,
        title=title,
        columns=columns,
        status=VisualizationStatus.UNAVAILABLE,
        reason=reason,
    )


# --------------------------------------------------------------------
# selection — one function per family
# --------------------------------------------------------------------


def _histograms(
    df: pd.DataFrame, numeric_cols: list[str], notes: list[str]
) -> list[VisualizationSpec]:
    selected = [
        c for (c,) in _truncate([(c,) for c in numeric_cols], MAX_HISTOGRAMS, "histograms", notes)
    ]
    specs: list[VisualizationSpec] = []
    for col in selected:
        title = f"Histogram of {col}"
        finite = _finite(df[col])
        if finite.size == 0:
            specs.append(
                _unavailable(VisualizationKind.HISTOGRAM, title, [col], "no finite observations")
            )
            continue
        if float(finite.min()) == float(finite.max()):
            specs.append(
                _unavailable(
                    VisualizationKind.HISTOGRAM,
                    title,
                    [col],
                    "constant column: a histogram needs a non-zero value range",
                )
            )
            continue
        specs.append(
            VisualizationSpec(
                kind=VisualizationKind.HISTOGRAM,
                title=title,
                columns=[col],
                status=VisualizationStatus.AVAILABLE,
                x_label=col,
                y_label="Frequency",
                metadata={
                    "value_column": col,
                    "n_observations": int(finite.size),
                    "bin_rule": "sturges",
                    "n_bins": sturges_bin_count(int(finite.size)),
                    "minimum": round(float(finite.min()), 10),
                    "maximum": round(float(finite.max()), 10),
                },
            )
        )
    return specs


def _bar_charts(
    df: pd.DataFrame, categorical_cols: list[str], notes: list[str]
) -> list[VisualizationSpec]:
    selected = [
        c
        for (c,) in _truncate([(c,) for c in categorical_cols], MAX_BAR_CHARTS, "bar_charts", notes)
    ]
    specs: list[VisualizationSpec] = []
    for col in selected:
        title = f"Bar chart of {col}"
        ordered = _ordered_value_counts(df[col])
        if not ordered:
            specs.append(
                _unavailable(VisualizationKind.BAR_CHART, title, [col], "no non-null observations")
            )
            continue
        specs.append(
            VisualizationSpec(
                kind=VisualizationKind.BAR_CHART,
                title=title,
                columns=[col],
                status=VisualizationStatus.AVAILABLE,
                x_label=col,
                y_label="Count",
                metadata={
                    "category_column": col,
                    "categories": [v for v, _ in ordered],
                    "counts": [c for _, c in ordered],
                    "order": "count_desc_then_value",
                    "n_observations": int(sum(c for _, c in ordered)),
                },
            )
        )
    return specs


def _scatter_plots(
    df: pd.DataFrame, numeric_cols: list[str], notes: list[str]
) -> list[VisualizationSpec]:
    pairs = [(a, b) for i, a in enumerate(numeric_cols) for b in numeric_cols[i + 1 :]]
    selected = _truncate(list(pairs), MAX_SCATTER_PLOTS, "scatter_plots", notes)
    specs: list[VisualizationSpec] = []
    for a, b in selected:
        title = f"Scatter plot of {a} vs {b}"
        paired = df[[a, b]].dropna()
        finite = paired[
            np.isfinite(paired[a].to_numpy(dtype=float))
            & np.isfinite(paired[b].to_numpy(dtype=float))
        ]
        n = len(finite)
        if n == 0:
            specs.append(
                _unavailable(
                    VisualizationKind.SCATTER_PLOT,
                    title,
                    [a, b],
                    "no rows where both columns are finite",
                )
            )
            continue
        specs.append(
            VisualizationSpec(
                kind=VisualizationKind.SCATTER_PLOT,
                title=title,
                columns=[a, b],
                status=VisualizationStatus.AVAILABLE,
                x_label=a,
                y_label=b,
                metadata={"x_column": a, "y_column": b, "n_observations": n},
            )
        )
    return specs


def _box_plots(
    df: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str], notes: list[str]
) -> list[VisualizationSpec]:
    combos = [(c, n) for c in categorical_cols for n in numeric_cols]
    selected = _truncate(list(combos), MAX_BOX_PLOTS, "box_plots", notes)
    specs: list[VisualizationSpec] = []
    for cat, num in selected:
        title = f"Box plot of {num} by {cat}"
        grouped = df[[cat, num]].dropna(subset=[cat])
        categories = sorted(str(v) for v in grouped[cat].astype("object").unique())
        label_col = grouped[cat].astype("object").astype(str)
        used: list[str] = []
        per_category: list[int] = []
        for category in categories:
            vals = _finite(grouped.loc[label_col == category, num])
            if vals.size:
                used.append(category)
                per_category.append(int(vals.size))
        if not used:
            specs.append(
                _unavailable(
                    VisualizationKind.BOX_PLOT,
                    title,
                    [cat, num],
                    "no category has a finite numeric observation",
                )
            )
            continue
        specs.append(
            VisualizationSpec(
                kind=VisualizationKind.BOX_PLOT,
                title=title,
                columns=[cat, num],
                status=VisualizationStatus.AVAILABLE,
                x_label=cat,
                y_label=num,
                metadata={
                    "category_column": cat,
                    "value_column": num,
                    "categories": used,
                    "counts": per_category,
                    "order": "category_asc",
                    "n_observations": int(sum(per_category)),
                },
            )
        )
    return specs


def analyze_visualizations(df: pd.DataFrame) -> VisualizationAnalysis:
    """Deterministically select the visualizations to describe for ``df``.

    Structural selection only — no target inference, no importance
    ranking, no randomness, no sampling. Numeric and categorical columns
    are taken in **alphabetical order**; numeric pairs and
    ``(categorical, numeric)`` combinations are generated in alphabetical
    order; each family is capped (``MAX_HISTOGRAMS`` / ``MAX_BAR_CHARTS``
    / ``MAX_SCATTER_PLOTS`` / ``MAX_BOX_PLOTS``) with a truncation note.
    ``df`` is not modified.
    """
    kinds = classify_columns(df)
    numeric_cols = sorted(c for c, k in kinds.items() if k is EDAColumnKind.NUMERIC)

    notes: list[str] = []
    categorical_cols: list[str] = []
    for col in sorted(c for c, k in kinds.items() if k is EDAColumnKind.CATEGORICAL):
        cardinality = int(df[col].dropna().nunique())
        if cardinality <= MAX_VISUALIZATION_CATEGORIES:
            categorical_cols.append(col)
        else:
            notes.append(
                f"'{col}' excluded from visualizations: cardinality {cardinality} exceeds "
                f"{MAX_VISUALIZATION_CATEGORIES}"
            )

    return VisualizationAnalysis(
        histograms=_histograms(df, numeric_cols, notes),
        bar_charts=_bar_charts(df, categorical_cols, notes),
        scatter_plots=_scatter_plots(df, numeric_cols, notes),
        box_plots=_box_plots(df, categorical_cols, numeric_cols, notes),
        notes=notes,
    )


# --------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------


def _require_available(spec: VisualizationSpec) -> None:
    if spec.status is not VisualizationStatus.AVAILABLE:
        raise VisualizationError(
            f"cannot render an unavailable {spec.kind.value}: {spec.reason or 'no reason given'}"
        )


def _render_histogram(df: pd.DataFrame, spec: VisualizationSpec, ax: object) -> None:
    col = str(spec.metadata["value_column"])
    finite = _finite(df[col])
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        raise VisualizationError(f"'{col}' has no non-constant finite data to plot")
    ax.hist(finite, bins=sturges_bin_count(int(finite.size)))  # type: ignore[attr-defined]
    ax.set_xlabel(spec.x_label or col)  # type: ignore[attr-defined]
    ax.set_ylabel(spec.y_label or "Frequency")  # type: ignore[attr-defined]


def _render_bar_chart(df: pd.DataFrame, spec: VisualizationSpec, ax: object) -> None:
    col = str(spec.metadata["category_column"])
    ordered = _ordered_value_counts(df[col])
    if not ordered:
        raise VisualizationError(f"'{col}' has no non-null data to plot")
    labels = [v for v, _ in ordered]
    heights = [c for _, c in ordered]
    positions = range(len(labels))
    ax.bar(positions, heights)  # type: ignore[attr-defined]
    ax.set_xticks(list(positions))  # type: ignore[attr-defined]
    ax.set_xticklabels(labels, rotation=45, ha="right")  # type: ignore[attr-defined]
    ax.set_xlabel(spec.x_label or col)  # type: ignore[attr-defined]
    ax.set_ylabel(spec.y_label or "Count")  # type: ignore[attr-defined]


def _render_scatter_plot(df: pd.DataFrame, spec: VisualizationSpec, ax: object) -> None:
    x_col = str(spec.metadata["x_column"])
    y_col = str(spec.metadata["y_column"])
    paired = df[[x_col, y_col]].dropna()
    x = paired[x_col].to_numpy(dtype=float)
    y = paired[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        raise VisualizationError(f"no rows where both '{x_col}' and '{y_col}' are finite")
    ax.scatter(x[mask], y[mask])  # type: ignore[attr-defined]
    ax.set_xlabel(spec.x_label or x_col)  # type: ignore[attr-defined]
    ax.set_ylabel(spec.y_label or y_col)  # type: ignore[attr-defined]


def _render_box_plot(df: pd.DataFrame, spec: VisualizationSpec, ax: object) -> None:
    cat = str(spec.metadata["category_column"])
    num = str(spec.metadata["value_column"])
    grouped = df[[cat, num]].dropna(subset=[cat])
    label_col = grouped[cat].astype("object").astype(str)
    data: list[np.ndarray] = []
    labels: list[str] = []
    for category in sorted(str(v) for v in grouped[cat].astype("object").unique()):
        vals = _finite(grouped.loc[label_col == category, num])
        if vals.size:
            data.append(vals)
            labels.append(category)
    if not data:
        raise VisualizationError(f"no category of '{cat}' has a finite '{num}' observation")
    ax.boxplot(data, tick_labels=labels)  # type: ignore[attr-defined]
    ax.set_xlabel(spec.x_label or cat)  # type: ignore[attr-defined]
    ax.set_ylabel(spec.y_label or num)  # type: ignore[attr-defined]


_RENDERERS = {
    VisualizationKind.HISTOGRAM: _render_histogram,
    VisualizationKind.BAR_CHART: _render_bar_chart,
    VisualizationKind.SCATTER_PLOT: _render_scatter_plot,
    VisualizationKind.BOX_PLOT: _render_box_plot,
}


def render_visualization(df: pd.DataFrame, spec: VisualizationSpec) -> Figure:
    """Render one *available* :class:`VisualizationSpec` to an in-memory
    ``matplotlib.figure.Figure``.

    * Matplotlib only, object-oriented API (no ``pyplot`` global state).
    * The figure stays in memory — **no file is written**.
    * ``df`` is not modified; missing / non-finite values are excluded,
      never filled or invented.
    * An unavailable spec, an unknown kind, or data that turns out to be
      unplottable raises :class:`VisualizationError` — a misleading empty
      figure is never returned.
    """
    _require_available(spec)
    renderer = _RENDERERS.get(spec.kind)
    if renderer is None:  # pragma: no cover - guarded by the enum
        raise VisualizationError(f"unsupported visualization kind: {spec.kind!r}")

    from matplotlib.figure import Figure as _Figure

    figure = _Figure()
    axes = figure.subplots()
    renderer(df, spec, axes)
    axes.set_title(spec.title)
    figure.tight_layout()
    return figure
