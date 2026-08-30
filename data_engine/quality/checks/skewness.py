"""Check: highly skewed numeric distributions.

Uses the Fisher-Pearson sample skewness (``pandas.Series.skew``). By
Bulmer's rule of thumb ``|skew| > 1`` is "highly skewed". Skew is not an
error — it is a hint that a transform (log, Box-Cox) or a
distribution-robust model may help later.
"""

from __future__ import annotations

import numpy as np

from datapilot.contracts import ColumnType

from ..context import CheckContext
from ..models import FindingType, QualityFinding, SuggestedAction
from ..severity import severity_from_skew
from ..thresholds import SKEW_HIGH_ABS, SKEW_MIN_NON_NULL

CHECK_NAME = "skewness"


def check(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.inferred_type is not ColumnType.NUMERIC:
            continue
        series = ctx.df[col.name].dropna()
        # Skew is only meaningful for a continuous-ish distribution: skip
        # binary / near-constant columns (e.g. 0/1 flags, encoded labels).
        if len(series) < SKEW_MIN_NON_NULL or series.nunique() <= 2:
            continue

        skew = float(series.skew())
        if not np.isfinite(skew) or abs(skew) < SKEW_HIGH_ABS:
            continue

        direction = "right/positive" if skew > 0 else "left/negative"
        findings.append(
            QualityFinding(
                finding_id=f"{FindingType.HIGH_SKEW.value}:{col.name}",
                finding_type=FindingType.HIGH_SKEW,
                severity=severity_from_skew(abs(skew)),
                columns=[col.name],
                observed={
                    "skewness": round(skew, 4),
                    "threshold": SKEW_HIGH_ABS,
                    "direction": direction,
                },
                description=(
                    f"Column '{col.name}' is {direction}-skewed (skewness = {skew:.2f}, "
                    f"threshold |{SKEW_HIGH_ABS}|). A distribution transform may help downstream."
                ),
                recommended_action=SuggestedAction.CONSIDER_DISTRIBUTION_TRANSFORM,
                confidence=None,
            )
        )
    return findings
