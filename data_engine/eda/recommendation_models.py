"""Structured models for the target-aware visualization recommendation layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe.

This layer **ranks existing** ``VisualizationSpec`` objects (produced by
:func:`data_engine.eda.visualization.analyze_visualizations`) with respect
to an **explicitly supplied** target column. It creates no new chart
kinds, runs no model, infers no target, and uses no randomness.

The ``score`` is a *visualization-usefulness heuristic* on a fixed
``0-100`` scale — it does **not** represent predictive importance or any
statistical quantity.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

RECOMMENDATION_ENGINE_VERSION = "1"

# Default cap for ``recommend_visualizations(..., max_recommendations=...)``.
MAX_VISUALIZATION_RECOMMENDATIONS = 10

# ---------------------------------------------------------------------------
# Scoring convention (documented, fixed). Higher = more useful for looking at
# a relationship with the target. NOT predictive importance.
#
#   numeric target      scatter plot involving the target        90
#                       box plot: numeric target by a category   80
#                       histogram of the target                  70
#   categorical target  box plot: numeric by the target category 90
#                       bar chart of the target                  80
#                       histogram of a numeric predictor that
#                         also has a box plot against the target 50
# ---------------------------------------------------------------------------
SCORE_NUMERIC_TARGET_SCATTER = 90.0
SCORE_NUMERIC_TARGET_BOX_PLOT = 80.0
SCORE_NUMERIC_TARGET_HISTOGRAM = 70.0
SCORE_CATEGORICAL_TARGET_BOX_PLOT = 90.0
SCORE_CATEGORICAL_TARGET_BAR_CHART = 80.0
SCORE_CATEGORICAL_TARGET_PREDICTOR_HISTOGRAM = 50.0


class VisualizationRecommendationKind(str, Enum):
    HISTOGRAM = "histogram"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    BOX_PLOT = "box_plot"


class VisualizationRecommendationStatus(str, Enum):
    RECOMMENDED = "recommended"  # a valid target was supplied; see `recommendations`
    UNAVAILABLE = "unavailable"  # no usable target (see `reason`)


class VisualizationRecommendation(BaseModel):
    """One ranked recommendation, pointing back at an existing spec."""

    kind: VisualizationRecommendationKind
    columns: list[str] = Field(description="Columns of the underlying visualization spec.")
    rank: int = Field(description="1-based rank within the analysis; unique.")
    score: float = Field(
        description="Visualization-usefulness heuristic (0-100). NOT predictive importance."
    )
    reason: str = Field(description="Deterministic, template-generated explanation.")
    target_column: str

    source_family: str = Field(
        description="Which VisualizationAnalysis list the spec came from "
        "('histograms' / 'bar_charts' / 'scatter_plots' / 'box_plots')."
    )
    source_index: int = Field(
        description="Index of the spec within that list — deterministic given the DataFrame."
    )
    notes: list[str] = Field(default_factory=list)


class VisualizationRecommendationAnalysis(BaseModel):
    """The recommendation section of an :class:`EDAReport`.

    A bare instance (the default on ``EDAReport``) is an explicit
    "no target supplied" result: ``status = unavailable``, empty
    ``recommendations``. Additive and defaulted, so reports serialised
    before this layer still validate.
    """

    recommendation_engine_version: str = RECOMMENDATION_ENGINE_VERSION
    target_column: str | None = None
    status: VisualizationRecommendationStatus = VisualizationRecommendationStatus.UNAVAILABLE
    reason: str | None = "no target column supplied"
    recommendations: list[VisualizationRecommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
