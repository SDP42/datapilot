"""Structured result models for the statistical-hypothesis-testing layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. These
models describe a *test outcome* — they never carry a transformation of
the data.

Design rule: an **unavailable** result uses ``None`` for the statistic /
p-value / significance plus an explicit ``reason`` and
``status = unavailable``. It never substitutes a fake ``0`` / ``1`` /
``False``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# Fixed default significance level — not caller-configurable per test run
# beyond a single explicit argument; there is no randomised behaviour.
DEFAULT_ALPHA = 0.05

# Deterministic caps for automatic test selection (documented; module-level).
MAX_TTEST_PAIRS = 50
MAX_ANOVA_COMBINATIONS = 50
MAX_CHI_SQUARE_PAIRS = 50

# Minimum valid observations / group sizes below which a test is unavailable.
TTEST_MIN_OBSERVATIONS = 2
ANOVA_MIN_GROUP_SIZE = 2
ANOVA_MIN_GROUPS = 2


class TestKind(str, Enum):
    WELCH_T_TEST = "welch_t_test"
    ONE_WAY_ANOVA = "one_way_anova"
    CHI_SQUARE_INDEPENDENCE = "chi_square_independence"


class TestStatus(str, Enum):
    COMPLETED = "completed"  # the test ran and produced a statistic + p-value
    UNAVAILABLE = "unavailable"  # the test could not be computed (see `reason`)


# These enums are named ``Test*`` for the domain, not for pytest — tell the
# collector to ignore them so importing them into a test module is quiet.
TestKind.__test__ = False  # type: ignore[attr-defined]
TestStatus.__test__ = False  # type: ignore[attr-defined]


class StatisticalTestResult(BaseModel):
    """One hypothesis-test outcome."""

    test_kind: TestKind
    test_name: str = Field(description="Human-readable test label.")
    columns: list[str] = Field(
        description="Variables involved, in a deterministic order.",
    )

    status: TestStatus
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
        description="Welch df (fractional) or chi-square df; None where not applicable.",
    )
    n_observations: int | None = Field(
        default=None, description="Valid observations used; None if unavailable."
    )
    n_groups: int | None = Field(
        default=None, description="Number of groups (one-way ANOVA); None otherwise."
    )

    alpha: float = Field(default=DEFAULT_ALPHA, description="Significance level used.")
    significant: bool | None = Field(
        default=None,
        description="p_value < alpha; None when the test is unavailable (never a fake False).",
    )

    notes: list[str] = Field(default_factory=list)


class StatisticalAnalysis(BaseModel):
    """The statistical section of an :class:`EDAReport`.

    Grouped by test family, mirroring ``BivariateSummary``. Every list is
    deterministically ordered; every truncation is recorded in ``notes``.
    """

    alpha: float = DEFAULT_ALPHA
    t_tests: list[StatisticalTestResult] = Field(default_factory=list)
    anova: list[StatisticalTestResult] = Field(default_factory=list)
    chi_square: list[StatisticalTestResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
