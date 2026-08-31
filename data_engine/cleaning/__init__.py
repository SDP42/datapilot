"""Cleaning layer — planning (proposals) and execution (approved changes).

    QualityReport
        -> plan_cleaning(...)        -> CleaningPlan          (proposals, no data touched)
        -> [explicit approval]
        -> execute_cleaning(...)     -> CleaningExecutionReport (+ processed dataset)

Execution performs ONLY explicitly approved operations, on a derived copy,
with atomic per-operation commit, operation-aware validation, train/test
leakage protection, lineage, and a before/after quality comparison. The
raw dataset is never modified.
"""

from __future__ import annotations

from .execution_models import (
    CleaningExecutionReport,
    ColumnStatistics,
    DatasetLineage,
    ExecutionReportStatus,
    ExecutionStatus,
    OperationExecution,
    ProcessedDatasetReference,
    QualityComparison,
    ValidationSummary,
)
from .executor import (
    EXECUTORS,
    CleaningExecutionResult,
    available_executors,
    execute_cleaning,
    execute_dataframe,
)
from .executors.base import ExecutionContext
from .models import (
    CleaningOperation,
    CleaningPlan,
    CleaningPlanSummary,
    ImputationStrategy,
    OperationCategory,
    OperationStatus,
    OperationType,
)
from .planner import (
    RULES,
    plan_cleaning,
    plan_from_dataframe,
    supported_finding_types,
)
from .processed_store import ProcessedDataStore

__all__ = [
    "EXECUTORS",
    "RULES",
    "CleaningExecutionReport",
    "CleaningExecutionResult",
    "CleaningOperation",
    "CleaningPlan",
    "CleaningPlanSummary",
    "ColumnStatistics",
    "DatasetLineage",
    "ExecutionContext",
    "ExecutionReportStatus",
    "ExecutionStatus",
    "ImputationStrategy",
    "OperationCategory",
    "OperationExecution",
    "OperationStatus",
    "OperationType",
    "ProcessedDataStore",
    "ProcessedDatasetReference",
    "QualityComparison",
    "ValidationSummary",
    "available_executors",
    "execute_cleaning",
    "execute_dataframe",
    "plan_cleaning",
    "plan_from_dataframe",
    "supported_finding_types",
]
