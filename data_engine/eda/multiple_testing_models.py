"""Result model for the multiple-testing correction layer.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. JSON
primitives only.

The layer takes a collection of **already-computed** p-values and returns
corrected p-values plus rejection decisions. It never recomputes a
p-value and never modifies any existing statistical-test output.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

MULTIPLE_TESTING_ROUND = 10
DEFAULT_CORRECTION_METHOD = "holm"
DEFAULT_CORRECTION_ALPHA = 0.05


class CorrectionMethod(str, Enum):
    BONFERRONI = "bonferroni"  # family-wise error rate
    HOLM = "holm"  # family-wise error rate, step-down
    BENJAMINI_HOCHBERG = "benjamini_hochberg"  # false discovery rate


# Accepted aliases → canonical method.
CORRECTION_METHOD_ALIASES: dict[str, CorrectionMethod] = {
    "bonferroni": CorrectionMethod.BONFERRONI,
    "holm": CorrectionMethod.HOLM,
    "holm-bonferroni": CorrectionMethod.HOLM,
    "benjamini_hochberg": CorrectionMethod.BENJAMINI_HOCHBERG,
    "benjamini-hochberg": CorrectionMethod.BENJAMINI_HOCHBERG,
    "bh": CorrectionMethod.BENJAMINI_HOCHBERG,
    "fdr_bh": CorrectionMethod.BENJAMINI_HOCHBERG,
}

_METHOD_CONTROLS = {
    CorrectionMethod.BONFERRONI: "family-wise error rate (FWER)",
    CorrectionMethod.HOLM: "family-wise error rate (FWER), step-down",
    CorrectionMethod.BENJAMINI_HOCHBERG: "false discovery rate (FDR)",
}


class MultipleTestingStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"  # invalid p-value input (see `reason`)


class MultipleTestingCorrectionResult(BaseModel):
    """Corrected p-values and rejection decisions for one family of tests."""

    method: CorrectionMethod
    controls: str = Field(description="What error rate `method` controls.")
    alpha: float

    n_hypotheses: int
    labels: list[str] | None = Field(
        default=None,
        description="Caller-supplied labels in the original input order; None if unlabelled.",
    )
    p_values: list[float] = Field(
        default_factory=list, description="Original p-values, in the original input order."
    )
    corrected_p_values: list[float] = Field(
        default_factory=list,
        description="Corrected p-values in [0, 1], aligned with `p_values` (input order).",
    )
    rejected: list[bool] = Field(
        default_factory=list,
        description="Reject H0 at `alpha`? Aligned with `p_values` (input order).",
    )
    n_rejected: int = 0

    status: MultipleTestingStatus
    reason: str | None = None
    notes: list[str] = Field(default_factory=list)
