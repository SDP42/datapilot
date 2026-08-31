"""Structured result models for the effect-size / association-measure layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. These
models describe *how strong* an observed relationship is — a complement
to the hypothesis tests in ``statistical_models.py``, not a replacement.

Design rule (identical to the statistical layer): an **unavailable**
result uses ``None`` for ``effect_size`` plus an explicit ``reason`` and
``status = unavailable``. It never substitutes a fake ``0.0`` / ``1.0`` /
``False``. A genuinely computed ``0.0`` (e.g. a constant variable carries
no information) is a real ``completed`` result, not an invented one.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Deterministic caps for automatic effect-size selection (module-level).
MAX_CRAMERS_V_PAIRS = 50
MAX_CORRELATION_RATIO_COMBINATIONS = 50
MAX_MUTUAL_INFORMATION_PAIRS = 50

# Minimum valid observations / group counts below which a measure is unavailable.
CRAMERS_V_MIN_CATEGORIES = 2
CORRELATION_RATIO_MIN_OBSERVATIONS = 2
CORRELATION_RATIO_MIN_GROUPS = 2
MUTUAL_INFORMATION_MIN_OBSERVATIONS = 2

# Numeric columns are quantile-binned before the (discrete) MI estimate.
MI_NUMERIC_BINS = 10

# Effect sizes in [0, 1] and MI in nats are rounded to this many places for
# cross-platform deterministic representation (mirrors the statistical layer).
EFFECT_ROUND = 10


class EffectKind(str, Enum):
    CRAMERS_V = "cramers_v"
    CORRELATION_RATIO = "correlation_ratio"
    MUTUAL_INFORMATION = "mutual_information"


class EffectStatus(str, Enum):
    COMPLETED = "completed"  # the measure was computed
    UNAVAILABLE = "unavailable"  # the measure could not be computed (see `reason`)


class EffectSizeResult(BaseModel):
    """One effect-size / association-measure outcome."""

    effect_kind: EffectKind
    measure_name: str = Field(description="Human-readable measure label.")
    columns: list[str] = Field(description="Variables involved, in a deterministic order.")

    status: EffectStatus
    reason: str | None = Field(
        default=None,
        description="Explicit reason the measure is unavailable; None when completed.",
    )

    effect_size: float | None = Field(
        default=None,
        description="The measure value; None if unavailable. Cramér's V / eta in [0, 1]; "
        "mutual information in nats (>= 0).",
    )
    n_observations: int | None = Field(
        default=None, description="Valid observations used; None if unavailable."
    )
    n_groups: int | None = Field(
        default=None, description="Number of category groups (correlation ratio); None otherwise."
    )

    notes: list[str] = Field(default_factory=list)


class EffectSizeAnalysis(BaseModel):
    """The effect-size section of an :class:`EDAReport`.

    Grouped by measure family, mirroring ``StatisticalAnalysis``. Every
    list is deterministically ordered; every truncation is recorded in
    ``notes``.
    """

    cramers_v: list[EffectSizeResult] = Field(default_factory=list)
    correlation_ratio: list[EffectSizeResult] = Field(default_factory=list)
    mutual_information: list[EffectSizeResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
