"""Check: category labels that differ only by case / whitespace.

``["Male", "male", "MALE", " Male "]`` collapses to one normalised key,
so the column probably has one real category recorded four ways. The
engine reports the groups; it never merges them.
"""

from __future__ import annotations

import re

from datapilot.contracts import ColumnType

from ..context import CheckContext
from ..models import FindingType, QualityFinding, Severity, SuggestedAction
from ..thresholds import (
    CATEGORICAL_INCONSISTENCY_CONFIDENCE,
    CATEGORICAL_MAX_DISTINCT,
    CATEGORICAL_MAX_DISTINCT_RATIO,
)

CHECK_NAME = "categorical_consistency"

_WS = re.compile(r"\s+")


def _normalise(value: str) -> str:
    return _WS.sub(" ", value.strip()).casefold()


def _is_category_like(ctx: CheckContext, column: str, distinct: int) -> bool:
    if distinct > CATEGORICAL_MAX_DISTINCT:
        return False
    n = ctx.profile.n_rows
    return not (n and distinct / n > CATEGORICAL_MAX_DISTINCT_RATIO)


def check(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.inferred_type is not ColumnType.CATEGORICAL:
            continue
        if not _is_category_like(ctx, col.name, col.unique_count):
            continue

        series = ctx.df[col.name].dropna().astype(str)
        groups: dict[str, list[str]] = {}
        for raw_value in series.unique():
            groups.setdefault(_normalise(raw_value), [])
            if raw_value not in groups[_normalise(raw_value)]:
                groups[_normalise(raw_value)].append(raw_value)

        collisions = {key: variants for key, variants in groups.items() if len(variants) > 1}
        if not collisions:
            continue

        affected_rows = int(
            series.isin([v for variants in collisions.values() for v in variants]).sum()
        )
        severity = Severity.MEDIUM if len(collisions) > 1 else Severity.LOW
        findings.append(
            QualityFinding(
                finding_id=f"{FindingType.INCONSISTENT_CATEGORIES.value}:{col.name}",
                finding_type=FindingType.INCONSISTENT_CATEGORIES,
                severity=severity,
                columns=[col.name],
                affected_rows=affected_rows,
                affected_percentage=ctx.percentage(affected_rows),
                observed={"variant_groups": collisions},
                description=(
                    f"Column '{col.name}' has {len(collisions)} value(s) that appear under "
                    "multiple spellings differing only by case or whitespace, e.g. "
                    f"{next(iter(collisions.values()))}."
                ),
                recommended_action=SuggestedAction.STANDARDIZE_CATEGORY_VALUES,
                confidence=CATEGORICAL_INCONSISTENCY_CONFIDENCE,
            )
        )
    return findings
