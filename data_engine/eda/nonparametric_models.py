"""Structured result models for the non-parametric statistical-test layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. These
models describe a *test outcome* — they never carry a transformation of
the data. Mirrors ``statistical_models.py`` / ``effect_models.py``.

Design rule (identical to the other layers): an **unavailable** result
uses ``None`` for the statistic / p-value / significance plus an explicit
``reason`` and ``status = unavailable``. It never substitutes a fake
``0`` / ``1`` / ``False``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .statistical_models import DEFAULT_ALPHA

# Deterministic caps for automatic non-parametric test selection.
MAX_SPEARMAN_PAIRS = 50
MAX_KENDALL_PAIRS = 50
MAX_MANN_WHITNEY_COMBINATIONS = 50
MAX_KRUSKAL_WALLIS_COMBINATIONS = 50

# Minimum valid observations / group sizes below which a test is unavailable.
RANK_CORRELATION_MIN_OBSERVATIONS = 3
MANN_WHITNEY_MIN_GROUP_SIZE = 2
KRUSKAL_WALLIS_MIN_GROUP_SIZE = 2
KRUSKAL_WALLIS_MIN_GROUPS = 2


class NonParametricTestKind(str, Enum):
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    MANN_WHITNEY_U = "mann_whitney_u"
    KRUSKAL_WALLIS = "kruskal_wallis"


class NonParametricTestStatus(str, Enum):
    COMPLETED = "completed"  # the test ran and produced a statistic + p-value
    UNAVAILABLE = "unavailable"  # the test could not be computed (see `reason`)


class NonParametricTestResult(BaseModel):
    """One non-parametric hypothesis-test outcome."""

    test_kind: NonParametricTestKind
    test_name: str = Field(description="Human-readable test label.")
    columns: list[str] = Field(description="Variables involved, in a deterministic order.")

    status: NonParametricTestStatus
    reason: str | None = Field(
        default=None,
        description="Explicit reason the test is unavailable; None when completed.",
    )

    statistic: float | None = Field(
        default=None, description="Test statistic; None if unavailable."
    )
    p_value: float | None = Field(default=None, description="p-value; None if unavailable.")
    degrees_of_freedom: float | None = Field(
        default=None,
        description="Kruskal-Wallis df (k-1); None where the test has no df.",
    )
    n_observations: int | None = Field(
        default=None, description="Valid observations used; None if unavailable."
    )
    n_groups: int | None = Field(
        default=None,
        description="Number of groups (Mann-Whitney / Kruskal-Wallis); None otherwise.",
    )

    alpha: float = Field(default=DEFAULT_ALPHA, description="Significance level used.")
    significant: bool | None = Field(
        default=None,
        description="p_value < alpha; None when the test is unavailable (never a fake False).",
    )

    notes: list[str] = Field(default_factory=list)


class NonParametricAnalysis(BaseModel):
    """The non-parametric section of an :class:`EDAReport`.

    Grouped by test family, mirroring ``StatisticalAnalysis``. Every list
    is deterministically ordered; every truncation is recorded in
    ``notes``.
    """

    alpha: float = DEFAULT_ALPHA
    spearman: list[NonParametricTestResult] = Field(default_factory=list)
    kendall: list[NonParametricTestResult] = Field(default_factory=list)
    mann_whitney_u: list[NonParametricTestResult] = Field(default_factory=list)
    kruskal_wallis: list[NonParametricTestResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
