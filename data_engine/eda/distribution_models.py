"""Structured result models for the richer distribution-analysis layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. These
models *describe* the distribution of a numeric column — they never carry
a transformation of the data and they hold no DataFrame.

Conventions (documented once, here):

* **Standard deviation / variance** — sample estimators, ``ddof=1``
  (matches ``pandas.Series.std`` / ``.var`` and the univariate layer).
  ``None`` when fewer than :data:`MIN_OBS_FOR_STD` finite observations.
* **Skewness** — the *adjusted Fisher-Pearson standardised moment
  coefficient* (``scipy.stats.skew(x, bias=False)``, identical to
  ``pandas.Series.skew``). ``0`` means symmetric. ``None`` for a constant
  column or fewer than :data:`MIN_OBS_FOR_SKEWNESS` finite observations.
* **Kurtosis** — **excess (Fisher) kurtosis**, bias-corrected
  (``scipy.stats.kurtosis(x, fisher=True, bias=False)``, identical to
  ``pandas.Series.kurt``). A normal distribution has kurtosis ``0``.
  ``None`` for a constant column or fewer than
  :data:`MIN_OBS_FOR_KURTOSIS` finite observations.
* **Quantiles** — :data:`DISTRIBUTION_QUANTILES` via ``numpy.quantile``
  (linear interpolation); ``0.0`` is the minimum, ``1.0`` the maximum.
* **Histogram** — a structured bin summary only (edges + counts), no
  rendering. Bin count follows :data:`HISTOGRAM_BIN_RULE` (Sturges:
  ``k = ceil(log2(n)) + 1``), clamped to
  ``[1, MIN(MAX_HISTOGRAM_BINS, ...)]``. Equal-width bins spanning
  ``[min, max]`` of the finite values. A **constant** column has no
  non-zero range, so its histogram is reported ``unavailable`` (never
  infinite / degenerate edges) while its min/max/mean stay valid.

Design rule (identical to the other EDA layers): an **unavailable**
result uses ``None`` plus an explicit ``reason`` / ``status`` — never a
fake ``0`` / ``1`` / ``False``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Rounding for deterministic cross-platform serialization — matches the
# other EDA layers (``statistics._ROUND`` etc.).
DISTRIBUTION_ROUND = 10

# Deterministic cap for the automatic per-column battery.
MAX_DISTRIBUTION_COLUMNS = 50

# Minimum finite observations required for each shape statistic.
MIN_OBS_FOR_STD = 2
MIN_OBS_FOR_SKEWNESS = 3
MIN_OBS_FOR_KURTOSIS = 4

# Fixed quantile probabilities — never caller-configurable. 0.0 / 1.0 are
# the exact min / max.
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Histogram binning: Sturges' rule, ``k = ceil(log2(n)) + 1``, clamped to
# ``[1, MAX_HISTOGRAM_BINS]``. Deterministic given the data.
HISTOGRAM_BIN_RULE = "sturges"
MAX_HISTOGRAM_BINS = 50


class DistributionStatus(str, Enum):
    COMPLETED = "completed"  # the (available) statistics were computed
    UNAVAILABLE = "unavailable"  # nothing could be computed (see `reason`)


class DistributionQuantile(BaseModel):
    quantile: float
    value: float | None


class HistogramBin(BaseModel):
    left_edge: float
    right_edge: float
    count: int


class Histogram(BaseModel):
    """A structured, render-free histogram summary.

    When ``status`` is ``completed`` the bins tile ``[bin_edges[0],
    bin_edges[-1]]`` with equal width and ``len(bin_edges) == n_bins + 1``.
    """

    status: DistributionStatus
    reason: str | None = None
    bin_rule: str = HISTOGRAM_BIN_RULE
    n_bins: int | None = None
    bin_edges: list[float] = Field(default_factory=list)
    bins: list[HistogramBin] = Field(default_factory=list)
    total_count: int | None = Field(
        default=None,
        description="Sum of bin counts; equals the finite-observation count when completed.",
    )


class NumericDistribution(BaseModel):
    """The distribution of one numeric column. See the module docstring for
    the exact statistical conventions."""

    column: str
    status: DistributionStatus
    reason: str | None = Field(
        default=None,
        description="Why the whole result is unavailable; None when status == completed.",
    )

    count: int = Field(description="Non-null observations.")
    missing_count: int
    missing_percentage: float
    unique_count: int = Field(description="Distinct non-null values.")

    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    variance: float | None = None
    skewness: float | None = Field(
        default=None,
        description="Adjusted Fisher-Pearson coefficient (bias-corrected); 0 = symmetric.",
    )
    kurtosis: float | None = Field(
        default=None,
        description="Excess (Fisher) kurtosis, bias-corrected; normal distribution = 0.",
    )

    quantiles: list[DistributionQuantile] = Field(default_factory=list)
    histogram: Histogram

    notes: list[str] = Field(
        default_factory=list,
        description="Per-statistic explanations (e.g. why skewness is unavailable).",
    )


class DistributionAnalysis(BaseModel):
    """The distribution section of an :class:`EDAReport`.

    Numeric columns only, in alphabetical order; every truncation is
    recorded in ``notes``. Additive and defaulted on ``EDAReport`` so
    reports serialised before this layer still validate.
    """

    columns: list[NumericDistribution] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
