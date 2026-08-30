"""Check: fully-duplicated rows (dataset level)."""

from __future__ import annotations

from ..context import CheckContext
from ..models import FindingType, QualityFinding, SuggestedAction
from ..severity import severity_from_duplicate_pct

CHECK_NAME = "duplicate_rows"


def check(ctx: CheckContext) -> list[QualityFinding]:
    dup_count = ctx.profile.duplicate_row_count
    if dup_count == 0:
        return []
    pct = ctx.percentage(dup_count)
    return [
        QualityFinding(
            finding_id=f"{FindingType.DUPLICATE_ROWS.value}:_dataset_",
            finding_type=FindingType.DUPLICATE_ROWS,
            severity=severity_from_duplicate_pct(pct),
            columns=[],
            affected_rows=dup_count,
            affected_percentage=pct,
            observed={"duplicate_row_count": dup_count, "duplicate_percentage": pct},
            description=(
                f"{dup_count} row(s) ({pct:.2f}%) are exact duplicates of an earlier row "
                "across all columns."
            ),
            recommended_action=SuggestedAction.REMOVE_DUPLICATE_ROWS,
            confidence=None,
        )
    ]
