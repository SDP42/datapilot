"""Structured result models for the EDA visualization foundation.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe.

A :class:`VisualizationSpec` is a *description* of a chart — enough
structured, deterministic information to render one later — and it never
holds a Matplotlib ``Figure`` (or any other live object). Rendering is a
separate step (:func:`data_engine.eda.visualization.render_visualization`)
and its output stays in memory: nothing is written to disk.

This is **not** a dashboard, frontend, or API layer, and it does no
target-aware chart recommendation — selection is purely structural
(see :mod:`data_engine.eda.visualization`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

VISUALIZATION_ENGINE_VERSION = "1"

# Deterministic caps, one per visualization family. A very wide DataFrame
# can never generate an unlimited number of specs — when a cap is hit the
# first candidates (in deterministic order) are kept and a truncation
# note is recorded.
MAX_HISTOGRAMS = 50
MAX_BAR_CHARTS = 50
MAX_SCATTER_PLOTS = 50
MAX_BOX_PLOTS = 50

# A categorical column with more than this many distinct non-null values
# is not used for bar charts or box plots (kept consistent with the
# bivariate layer's ``MAX_BIVARIATE_CARDINALITY``).
MAX_VISUALIZATION_CATEGORIES = 50


class VisualizationKind(str, Enum):
    HISTOGRAM = "histogram"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    BOX_PLOT = "box_plot"


class VisualizationStatus(str, Enum):
    AVAILABLE = "available"  # the spec can be rendered
    UNAVAILABLE = "unavailable"  # described but not renderable (see `reason`)


class VisualizationSpec(BaseModel):
    """A deterministic, render-free description of one chart."""

    kind: VisualizationKind
    title: str
    columns: list[str] = Field(description="Columns used, in a deterministic order.")

    status: VisualizationStatus
    reason: str | None = Field(
        default=None,
        description="Why the spec is unavailable; None when status == available.",
    )

    x_label: str | None = None
    y_label: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Deterministic, JSON-primitive rendering metadata (e.g. the value column, "
            "the Sturges bin count, the ordered category list). No live objects."
        ),
    )
    notes: list[str] = Field(default_factory=list)


class VisualizationAnalysis(BaseModel):
    """The visualization section of an :class:`EDAReport`.

    Specs are grouped by family; every list is deterministically ordered
    and every truncation is recorded in ``notes``. Additive and defaulted
    on ``EDAReport`` so reports serialised before this layer still
    validate.
    """

    visualization_engine_version: str = VISUALIZATION_ENGINE_VERSION
    histograms: list[VisualizationSpec] = Field(default_factory=list)
    bar_charts: list[VisualizationSpec] = Field(default_factory=list)
    scatter_plots: list[VisualizationSpec] = Field(default_factory=list)
    box_plots: list[VisualizationSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
