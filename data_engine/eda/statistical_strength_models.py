"""Models for statistical-strength ranking of existing visualization specs.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. No model
here holds a DataFrame, a Figure (Matplotlib or Plotly), a NumPy array,
or any other non-JSON runtime object.

This layer is **distinct** from the target-aware structural
recommendation (`recommend_visualizations`):

* structural recommendation → *visualisation usefulness* (fixed heuristic
  constants; answers "which chart is worth looking at?");
* statistical-strength ranking → *strength of the statistical evidence*
  for the relationship a chart depicts (real effect sizes and p-values
  taken from the existing EDA statistical / effect-size / bivariate /
  non-parametric layers; answers "which relationship has the strongest
  measured association?").

The ``strength_score`` is an **association-magnitude** value on a 0-1
scale — it is **not** feature importance, not predictive performance, and
not a p-value.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .visualization_models import VisualizationKind

STRENGTH_ENGINE_VERSION = "1"

# Default cap for ``rank_visualizations_by_statistical_strength(...,
# max_recommendations=...)`` — mirrors the structural recommendation layer.
MAX_STRENGTH_RECOMMENDATIONS = 10

RANKING_POLICY = (
    "strength_score = the association-magnitude effect size for the relationship, on a "
    "0-1 scale: |Pearson r| for numeric-numeric, correlation ratio eta for "
    "categorical-numeric, Cramer's V for categorical-categorical. Entries are ordered "
    "by: (1) an available strength_score first, (2) strength_score descending, "
    "(3) effect-size magnitude descending, (4) p_value ascending (tie-break only, never "
    "the primary key), (5) visualization kind, (6) column names. Ranks are 1..N, unique "
    "and sequential. The p_value is supporting evidence only; a small p_value is never "
    "treated as a large practical effect. strength_score is NOT feature importance and "
    "NOT predictive performance."
)


class VisualizationStatisticalStrengthStatus(str, Enum):
    RANKED = "ranked"  # a valid target was supplied; see `recommendations`
    UNAVAILABLE = "unavailable"  # no usable target (see `reason`)


class PValueAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class VisualizationStatisticalStrength(BaseModel):
    """One ranked relationship, pointing back at an existing spec."""

    kind: VisualizationKind
    columns: list[str] = Field(description="Columns of the underlying visualization spec.")
    predictor_column: str
    target_column: str
    relationship: str = Field(
        description="'numeric-numeric' / 'categorical-numeric' / 'categorical-categorical'."
    )

    rank: int = Field(description="1-based rank within the analysis; unique and sequential.")
    strength_score: float | None = Field(
        description="Association magnitude in [0, 1]; None when no effect size is available."
    )
    strength_score_reason: str | None = Field(
        default=None, description="Why strength_score is None; None when it is present."
    )

    statistic_name: str | None = None
    statistic_value: float | None = None

    p_value: float | None = None
    p_value_availability: PValueAvailability = PValueAvailability.UNAVAILABLE
    p_value_reason: str | None = None

    effect_size_name: str | None = None
    effect_size_value: float | None = None
    effect_size_reason: str | None = None

    source_family: str = Field(
        description="VisualizationAnalysis list the spec came from "
        "('histograms' / 'bar_charts' / 'scatter_plots' / 'box_plots')."
    )
    source_index: int = Field(description="Index of the spec within that list.")
    notes: list[str] = Field(default_factory=list)


class VisualizationStatisticalStrengthAnalysis(BaseModel):
    """The statistical-strength section of an :class:`EDAReport`.

    A bare instance (the default on ``EDAReport``) is an explicit
    "no target supplied" result. Additive and defaulted, so reports
    serialised before this layer still validate.
    """

    strength_engine_version: str = STRENGTH_ENGINE_VERSION
    target_column: str | None = None
    status: VisualizationStatisticalStrengthStatus = (
        VisualizationStatisticalStrengthStatus.UNAVAILABLE
    )
    reason: str | None = "no target column supplied"
    ranking_policy: str = RANKING_POLICY
    recommendations: list[VisualizationStatisticalStrength] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
