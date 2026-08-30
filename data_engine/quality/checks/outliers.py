"""Check: potential outliers in numeric columns, via the IQR rule.

Tukey's fences: a value is a *potential* outlier if it falls outside
``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``. This is a flag for review, not a
verdict that the value is wrong — heavy-tailed but perfectly valid
distributions (income, latency) will legitimately trip it.
"""

from __future__ import annotations

from datapilot.contracts import ColumnType

from ..context import CheckContext
from ..models import FindingType, QualityFinding, SuggestedAction
from ..severity import severity_from_outlier_pct
from ..thresholds import IQR_FENCE_MULTIPLIER, OUTLIER_MIN_NON_NULL

CHECK_NAME = "outliers"


def check(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.inferred_type is not ColumnType.NUMERIC:
            continue
        series = ctx.df[col.name].dropna()
        if len(series) < OUTLIER_MIN_NON_NULL:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue  # >50% of values identical; IQR rule is not informative

        lower = q1 - IQR_FENCE_MULTIPLIER * iqr
        upper = q3 + IQR_FENCE_MULTIPLIER * iqr
        mask = (series < lower) | (series > upper)
        outlier_count = int(mask.sum())
        if outlier_count == 0:
            continue

        pct = ctx.percentage(outlier_count)
        extremes = series[mask]
        findings.append(
            QualityFinding(
                finding_id=f"{FindingType.POTENTIAL_OUTLIERS.value}:{col.name}",
                finding_type=FindingType.POTENTIAL_OUTLIERS,
                severity=severity_from_outlier_pct(pct),
                columns=[col.name],
                affected_rows=outlier_count,
                affected_percentage=pct,
                observed={
                    "method": "iqr",
                    "fence_multiplier": IQR_FENCE_MULTIPLIER,
                    "q1": q1,
                    "q3": q3,
                    "lower_fence": lower,
                    "upper_fence": upper,
                    "min_outlier": float(extremes.min()),
                    "max_outlier": float(extremes.max()),
                },
                description=(
                    f"Column '{col.name}' has {outlier_count} value(s) ({pct:.2f}%) outside the "
                    f"IQR fence [{lower:.4g}, {upper:.4g}]. These are potential anomalies to "
                    "review, not confirmed errors."
                ),
                recommended_action=SuggestedAction.INVESTIGATE_OUTLIERS,
                confidence=None,  # the fence is exact; whether it's an *error* is not
            )
        )
    return findings
