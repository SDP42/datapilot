"""Plan proposals for a ``potential_type_mismatch`` finding.

Two shapes, read from ``finding.observed["looks_like"]``:

* ``"numeric"``  -> propose converting the text column to a numeric dtype
* ``"datetime"`` -> propose converting the text column to datetime

In both cases the plan is explicit that conversion must be *validated*
first and that unparseable values must be surfaced, never silently turned
into missing values.
"""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityFinding

from ..context import PlanContext
from ..models import (
    CleaningOperation,
    OperationCategory,
    OperationStatus,
    OperationType,
)


def _numeric(finding: QualityFinding, column: str) -> CleaningOperation:
    parse_ratio = float(finding.observed.get("parse_ratio", 1.0))
    all_parse = parse_ratio >= 1.0
    status = OperationStatus.RECOMMENDED if all_parse else OperationStatus.REVIEW_REQUIRED
    return CleaningOperation(
        operation_id=f"{OperationType.CONVERT_TEXT_TO_NUMERIC.value}:{column}",
        operation_type=OperationType.CONVERT_TEXT_TO_NUMERIC,
        category=OperationCategory.DATA_TRANSFORMATION,
        status=status,
        status_reason=(
            "Every non-null value parses as a number; converting the stored type is safe and "
            "makes the column usable for statistics and models."
            if all_parse
            else f"Only {parse_ratio:.1%} of non-null values parse as numbers; a human must "
            "decide what the non-numeric values mean before converting."
        ),
        target_columns=[column],
        addresses_finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
        source_finding_id=finding.finding_id,
        problem_summary=f"Column '{column}' holds numbers but is stored as text.",
        proposed_action=(
            f"Convert '{column}' to a numeric dtype AFTER the executor validates that every "
            "non-null value parses. If any value does not parse, abort and report it — do NOT "
            "coerce it to missing."
        ),
        rationale="Text-typed numbers cannot be aggregated, scaled, or fed to most models.",
        assumptions=[
            "The values represent a single numeric quantity in one unit.",
            "Thousands separators / currency symbols, if any, are handled explicitly by the executor.",
        ],
        risks=[
            "Silent coercion of odd values to NaN would hide a real data problem.",
            "Locale differences (',' vs '.' decimal) can misparse values.",
        ],
        parameters={
            "validate_before_apply": True,
            "on_unparseable": "abort_and_report",
            "observed_parse_ratio": round(parse_ratio, 4),
        },
        affected_rows=finding.affected_rows,
        affected_percentage=finding.affected_percentage,
        confidence=parse_ratio,
        requires_train_test_split_awareness=False,
    )


def _datetime(finding: QualityFinding, column: str) -> CleaningOperation:
    return CleaningOperation(
        operation_id=f"{OperationType.CONVERT_TEXT_TO_DATETIME.value}:{column}",
        operation_type=OperationType.CONVERT_TEXT_TO_DATETIME,
        category=OperationCategory.DATA_TRANSFORMATION,
        status=OperationStatus.REVIEW_REQUIRED,
        status_reason=(
            "Date text is ambiguous (day/month order, locale, two-digit years). A human should "
            "confirm the intended format before conversion."
        ),
        target_columns=[column],
        addresses_finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
        source_finding_id=finding.finding_id,
        problem_summary=f"Column '{column}' holds dates/timestamps but is stored as text.",
        proposed_action=(
            f"Convert '{column}' to a datetime dtype using an explicit, confirmed format. The "
            "executor must report any value that fails to parse rather than silently turning it "
            "into NaT (missing)."
        ),
        rationale="A real datetime dtype enables ordering, resampling, and time-based features.",
        assumptions=[
            "A single consistent date format / interpretation applies to the whole column.",
        ],
        risks=[
            "Ambiguous formats (MM/DD vs DD/MM) can silently produce wrong dates.",
            "Coercing unparseable values to NaT hides data-entry errors.",
        ],
        parameters={
            "validate_before_apply": True,
            "on_unparseable": "report_do_not_coerce",
            "format": None,
        },
        affected_rows=finding.affected_rows,
        affected_percentage=finding.affected_percentage,
        confidence=finding.confidence,
        requires_train_test_split_awareness=False,
    )


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    column = finding.columns[0]
    looks_like = finding.observed.get("looks_like")
    if looks_like == "numeric":
        return [_numeric(finding, column)]
    if looks_like == "datetime":
        return [_datetime(finding, column)]
    return []
