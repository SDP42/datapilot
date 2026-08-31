"""Machine-readable models for *executing* an approved CleaningPlan.

Stage 3 of the cleaning layer:

    DETECTION  (data_engine.quality)   -> QualityReport
    PLANNING   (planner.py)            -> CleaningPlan
    EXECUTION  (executor.py, here)     -> CleaningExecutionReport + processed dataset

Every model is Pydantic v2 and JSON-serialisable
(``report.model_dump(mode="json")`` round-trips). None of these models
holds a DataFrame.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_engine.cleaning.models import OperationCategory, OperationStatus, OperationType

EXECUTION_ENGINE_VERSION = "1"


class ExecutionStatus(str, Enum):
    """Outcome of a single operation."""

    SUCCESS = "success"  # transformation applied and validated
    SKIPPED = "skipped"  # not approved, or non-transforming by design
    FAILED = "failed"  # rejected or failed validation; data unchanged
    ABORTED = "aborted"  # safety contract stopped it mid-flight; data unchanged


class ExecutionReportStatus(str, Enum):
    COMPLETED = "completed"  # every attempted operation succeeded
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    NOTHING_EXECUTED = "nothing_executed"  # no operation was eligible/approved


class ColumnStatistics(BaseModel):
    """A compact, auditable snapshot of one column — never the raw values."""

    column: str
    dtype: str
    count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    median: float | None = None


class OperationExecution(BaseModel):
    """Full record of what happened to one CleaningOperation."""

    execution_id: str
    operation_id: str
    operation_type: OperationType
    operation_category: OperationCategory
    plan_status: OperationStatus = Field(
        description="The safety status the operation had in the plan."
    )
    approved: bool

    status: ExecutionStatus
    message: str
    source_finding_id: str
    target_columns: list[str] = Field(default_factory=list)

    rows_before: int | None = None
    rows_after: int | None = None
    columns_before: int | None = None
    columns_after: int | None = None
    affected_rows: int | None = None
    values_changed: int | None = None
    columns_added: list[str] = Field(default_factory=list)
    columns_removed: list[str] = Field(default_factory=list)

    before_statistics: dict[str, ColumnStatistics] = Field(default_factory=dict)
    after_statistics: dict[str, ColumnStatistics] = Field(default_factory=dict)

    fit_details: dict[str, Any] = Field(
        default_factory=dict,
        description="For leakage-aware ops: strategy, fit_on, fit_rows, fit_value.",
    )
    parameters_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Effective parameters + op-specific outputs (parse_ratio, mapping, ...).",
    )

    validation_passed: bool = True
    validation_messages: list[str] = Field(default_factory=list)
    error: str | None = None

    started_at: _dt.datetime
    completed_at: _dt.datetime


class LineageStep(BaseModel):
    index: int
    operation_id: str
    operation_type: OperationType
    source_finding_id: str
    status: ExecutionStatus
    summary: str


class DatasetLineage(BaseModel):
    """How the processed dataset was produced from the raw dataset."""

    raw_dataset_id: str
    raw_sha256: str | None = None
    plan_fingerprint: str
    planner_version: str
    quality_engine_version: str
    steps: list[LineageStep] = Field(default_factory=list)
    processed_dataset_id: str | None = None
    processed_sha256: str | None = None


class ProcessedDatasetReference(BaseModel):
    """An immutable pointer to a processed (cleaned) dataset version."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(description="Identity of this processed version.")
    parent_dataset_id: str = Field(description="The raw dataset it derives from.")
    execution_id: str
    plan_fingerprint: str
    path: Path
    source_format: str = "csv"
    size_bytes: int = Field(ge=0)
    sha256: str
    n_rows: int
    n_columns: int
    created_at: _dt.datetime


class QualityComparison(BaseModel):
    """Did the cleaning actually improve the dataset?"""

    before: dict[str, Any]
    after: dict[str, Any]
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    operations_validated: int
    all_passed: bool
    failures: list[str] = Field(default_factory=list)


class CleaningExecutionReport(BaseModel):
    """The complete, machine-readable record of one execution run."""

    dataset_id: str
    execution_engine_version: str = EXECUTION_ENGINE_VERSION
    execution_id: str
    generated_at: _dt.datetime

    plan_fingerprint: str
    planner_version: str
    based_on_quality_engine_version: str
    target_column: str | None = None

    status: ExecutionReportStatus
    approved_operation_ids: list[str] = Field(default_factory=list)
    auto_execute_recommended: bool = False

    operations_attempted: int
    operations_succeeded: int
    operations_skipped: int
    operations_failed: int

    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int

    operations: list[OperationExecution] = Field(default_factory=list)

    output_dataset_reference: ProcessedDatasetReference | None = None

    validation_summary: ValidationSummary
    lineage: DatasetLineage

    before_quality_summary: dict[str, Any] = Field(default_factory=dict)
    after_quality_summary: dict[str, Any] = Field(default_factory=dict)
    quality_comparison: QualityComparison | None = None
