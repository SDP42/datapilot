"""The deterministic cleaning planner.

    QualityReport (+ optional DatasetProfile)
        -> per-finding planning rules
        -> CleaningPlan  (list of CleaningOperation proposals)

Deterministic: no randomness, no LLM. Read-only: it never touches a
DataFrame, and it never mutates the QualityReport it is given.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd

from data_engine.profiling import profile_dataframe
from data_engine.profiling.models import DatasetProfile
from data_engine.quality import analyze_profile
from data_engine.quality.models import FindingType, QualityFinding, QualityReport

from .context import PlanContext
from .models import CleaningOperation, CleaningPlan
from .rules import (
    class_imbalance,
    duplicate_rows,
    high_skew,
    inconsistent_categories,
    missing_values,
    outliers,
    type_mismatch,
)
from .summary import build_plan_summary
from .thresholds import STATUS_ORDER

Rule = Callable[[QualityFinding, PlanContext], list[CleaningOperation]]

RULES: dict[FindingType, Rule] = {
    FindingType.MISSING_VALUES: missing_values.plan,
    FindingType.DUPLICATE_ROWS: duplicate_rows.plan,
    FindingType.POTENTIAL_TYPE_MISMATCH: type_mismatch.plan,
    FindingType.INCONSISTENT_CATEGORIES: inconsistent_categories.plan,
    FindingType.POTENTIAL_OUTLIERS: outliers.plan,
    FindingType.HIGH_SKEW: high_skew.plan,
    FindingType.CLASS_IMBALANCE: class_imbalance.plan,
}


def supported_finding_types() -> tuple[FindingType, ...]:
    return tuple(RULES)


def plan_cleaning(report: QualityReport, *, profile: DatasetProfile | None = None) -> CleaningPlan:
    """Turn a QualityReport into a CleaningPlan of proposals.

    ``profile`` is optional but recommended: without it the planner cannot
    pick type-specific strategies (median vs mode) or verify facts such as
    "strictly positive", and will escalate those operations to
    REVIEW_REQUIRED.
    """
    ctx = PlanContext(report=report, profile=profile)

    operations: list[CleaningOperation] = []
    for finding in report.findings:
        rule = RULES.get(finding.finding_type)
        if rule is None:
            continue
        operations.extend(rule(finding, ctx))

    # Safest first, stable within a status.
    operations.sort(key=lambda op: STATUS_ORDER[op.status])

    return CleaningPlan(
        dataset_id=report.dataset_id,
        generated_at=dt.datetime.now(dt.UTC),
        target_column=report.target_column,
        based_on_quality_engine_version=report.quality_engine_version,
        used_profile=profile is not None,
        source_findings_considered=len(report.findings),
        operations=operations,
        summary=build_plan_summary(operations),
    )


def plan_from_dataframe(
    df: pd.DataFrame,
    *,
    dataset_id: str = "adhoc",
    target_column: str | None = None,
) -> CleaningPlan:
    """Convenience: profile -> quality-analyse -> plan, for one in-memory df.

    ``df`` is treated as read-only throughout.
    """
    profile = profile_dataframe(df, dataset_id=dataset_id)
    report = analyze_profile(df, profile, target_column=target_column)
    return plan_cleaning(report, profile=profile)
