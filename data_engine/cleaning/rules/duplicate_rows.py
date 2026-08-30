"""Plan proposal for a ``duplicate_rows`` finding."""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityFinding

from ..context import PlanContext
from ..models import (
    CleaningOperation,
    OperationCategory,
    OperationStatus,
    OperationType,
)


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    return [
        CleaningOperation(
            operation_id=f"{OperationType.REMOVE_EXACT_DUPLICATE_ROWS.value}:_dataset_",
            operation_type=OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
            category=OperationCategory.DATA_TRANSFORMATION,
            status=OperationStatus.RECOMMENDED,
            status_reason=(
                "Rows identical across every column are almost always ingestion or join "
                "artefacts. Removal keeps the first occurrence and is recoverable from the "
                "immutable raw copy."
            ),
            target_columns=[],
            addresses_finding_type=FindingType.DUPLICATE_ROWS,
            source_finding_id=finding.finding_id,
            problem_summary=(
                f"{finding.affected_rows} row(s) are exact duplicates of an earlier row."
            ),
            proposed_action=(
                "Remove rows that are exact, full-row duplicates of an earlier row, keeping the "
                "first occurrence. ONLY exact duplicates across all columns are targeted — "
                "near-duplicates and key-based duplicates are out of scope."
            ),
            rationale="Exact duplicates add no information and bias every statistic toward the repeats.",
            assumptions=[
                "A fully-identical row is a true duplicate, not a legitimate repeated observation.",
                "The first occurrence is the authoritative one.",
            ],
            risks=[
                (
                    "If a real observation can naturally recur with identical values (e.g. two "
                    "identical sensor readings), a genuine row would be dropped — confirm a "
                    "unique key exists or that exact repeats are impossible."
                ),
            ],
            parameters={"scope": "exact_full_row_duplicates", "keep": "first"},
            affected_rows=finding.affected_rows,
            affected_percentage=finding.affected_percentage,
            confidence=0.85,
            requires_train_test_split_awareness=False,
        )
    ]
