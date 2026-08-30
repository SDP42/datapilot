"""Check: missing values, per column, severity by proportion."""

from __future__ import annotations

from ..context import CheckContext
from ..models import FindingType, QualityFinding, SuggestedAction
from ..severity import severity_from_missing_pct

CHECK_NAME = "missing_values"


def check(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.missing_count == 0:
            continue
        pct = col.missing_percentage
        severity = severity_from_missing_pct(pct)
        fully_missing = col.missing_count == ctx.profile.n_rows
        desc = f"Column '{col.name}' has {col.missing_count} missing value(s) ({pct:.2f}% of rows)."
        if fully_missing:
            desc += " The column is entirely empty."
        findings.append(
            QualityFinding(
                finding_id=f"{FindingType.MISSING_VALUES.value}:{col.name}",
                finding_type=FindingType.MISSING_VALUES,
                severity=severity,
                columns=[col.name],
                affected_rows=col.missing_count,
                affected_percentage=pct,
                observed={
                    "missing_count": col.missing_count,
                    "missing_percentage": pct,
                    "fully_missing": fully_missing,
                },
                description=desc,
                recommended_action=SuggestedAction.HANDLE_MISSING_VALUES,
                confidence=None,  # exact count, not a heuristic
            )
        )
    return findings
