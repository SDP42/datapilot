"""Build the :class:`CleaningPlanSummary` from a list of operations."""

from __future__ import annotations

from .models import (
    CleaningOperation,
    CleaningPlanSummary,
    OperationCategory,
    OperationStatus,
    OperationType,
)


def build_plan_summary(operations: list[CleaningOperation]) -> CleaningPlanSummary:
    by_status: dict[OperationStatus, int] = {s: 0 for s in OperationStatus}
    by_type: dict[OperationType, int] = {t: 0 for t in OperationType}
    by_category: dict[OperationCategory, int] = {c: 0 for c in OperationCategory}
    columns: list[str] = []

    for op in operations:
        by_status[op.status] += 1
        by_type[op.operation_type] += 1
        by_category[op.category] += 1
        for col in op.target_columns:
            if col not in columns:
                columns.append(col)

    auto_applicable = sum(
        1
        for op in operations
        if op.status is OperationStatus.RECOMMENDED
        and op.category is OperationCategory.DATA_TRANSFORMATION
    )

    notes: list[str] = []
    if by_status[OperationStatus.REVIEW_REQUIRED]:
        notes.append(
            f"{by_status[OperationStatus.REVIEW_REQUIRED]} operation(s) need human review "
            "before they may be executed."
        )
    if by_status[OperationStatus.NOT_SAFE_TO_AUTOMATE]:
        notes.append(
            f"{by_status[OperationStatus.NOT_SAFE_TO_AUTOMATE]} operation(s) must never be "
            "auto-applied — they need domain context."
        )
    if by_category[OperationCategory.MODELING_RECOMMENDATION]:
        notes.append(
            f"{by_category[OperationCategory.MODELING_RECOMMENDATION]} item(s) are advice for "
            "the ML phase, not data-cleaning transformations."
        )

    return CleaningPlanSummary(
        total_operations=len(operations),
        by_status=by_status,
        by_type=by_type,
        by_category=by_category,
        columns_affected=columns,
        auto_applicable_count=auto_applicable,
        notes=notes,
    )
