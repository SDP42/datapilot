"""Result model for the paired / one-sided non-parametric test layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. This model
holds only JSON primitives — no SciPy result object, NumPy array, or
DataFrame.

Covers three related-samples tests that the independent-sample
non-parametric foundation (``analyze_nonparametric``) does **not**
provide:

* **Wilcoxon signed-rank** — paired, one- or two-sided;
* **sign test** — paired, binomial test on the signs of the non-zero
  differences;
* **Friedman** — three or more related (repeated-measures) groups.

The existing ``NonParametricTestResult`` / ``analyze_nonparametric`` are
unchanged; this is a separate model and module.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .statistical_models import DEFAULT_ALPHA

PAIRED_NONPARAMETRIC_ROUND = 10

# Wilcoxon / sign test need at least this many usable (non-zero-difference)
# pairs; Friedman needs at least this many complete blocks.
PAIRED_MIN_OBSERVATIONS = 3

# Friedman needs at least this many related groups.
FRIEDMAN_MIN_GROUPS = 3

VALID_ALTERNATIVES = ("two-sided", "greater", "less")


class PairedNonParametricTestName(str, Enum):
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    SIGN_TEST = "sign_test"
    FRIEDMAN = "friedman"


class PairedNonParametricStatus(str, Enum):
    COMPLETED = "completed"  # the test ran and produced a statistic + p-value
    UNAVAILABLE = "unavailable"  # the test could not be computed (see `reason`)


class PairedNonParametricResult(BaseModel):
    """One paired / repeated-measures non-parametric test outcome."""

    test_name: PairedNonParametricTestName
    test_family: str = Field(
        default="paired_nonparametric",
        description="Fixed family label — distinguishes this from analyze_nonparametric.",
    )
    alternative: str | None = Field(
        default=None,
        description="'two-sided' / 'greater' / 'less'; None for the Friedman test.",
    )

    status: PairedNonParametricStatus
    reason: str | None = Field(
        default=None,
        description="Explicit reason the test is unavailable; None when completed.",
    )

    statistic: float | None = Field(
        default=None, description="Test statistic; None if unavailable."
    )
    p_value: float | None = Field(default=None, description="p-value; None if unavailable.")

    n_observations: int | None = Field(
        default=None,
        description=(
            "Usable pairs (Wilcoxon: non-zero differences; sign test: non-zero "
            "differences; Friedman: complete blocks / rows)."
        ),
    )
    n_groups: int | None = Field(
        default=None, description="Number of related groups (Friedman); None otherwise."
    )

    # Sign-test detail (None for the other tests).
    n_positive: int | None = Field(
        default=None, description="Sign test: count of positive non-zero differences."
    )
    n_negative: int | None = Field(
        default=None, description="Sign test: count of negative non-zero differences."
    )
    n_zero: int | None = Field(
        default=None, description="Sign / Wilcoxon: count of excluded zero differences."
    )

    alpha: float = Field(default=DEFAULT_ALPHA, description="Significance level used.")
    significant: bool | None = Field(
        default=None,
        description="p_value < alpha; None when the test is unavailable (never a fake False).",
    )

    notes: list[str] = Field(default_factory=list)
