"""First-class, typed models for a concrete dataset *version* and its
provenance.

Phase 3 (Validation & Data Lineage) makes dataset versions auditable
objects rather than loose metadata scattered across the execution report.

Reuses, does not duplicate:

* ``datapilot.contracts.DatasetReference``          — the immutable raw file
* ``data_engine.cleaning.ProcessedDatasetReference`` — a processed file pointer
* ``data_engine.cleaning.CleaningExecutionReport``   — how a processed version was produced

A ``DatasetVersion`` links those together and adds a schema snapshot, a
quality snapshot, parent/child lineage, and a stable identity.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from data_engine.cleaning.execution_models import CleaningExecutionReport
from datapilot.contracts import DatasetReference

VERSION_MODEL_VERSION = "1"

RAW_VERSION_SUFFIX = "raw"


class DatasetVersionKind(str, Enum):
    RAW = "raw"  # the ingested, immutable original
    PROCESSED = "processed"  # produced by the cleaning executor from a parent version


class DatasetVersionStatus(str, Enum):
    REGISTERED = "registered"  # recorded in the version store
    SUPERSEDED = "superseded"  # a newer version replaced it (set manually, never silently)
    INVALID = "invalid"  # failed a later integrity check


class ColumnSchema(BaseModel):
    name: str
    dtype: str


class SchemaSnapshot(BaseModel):
    """The column layout of a dataset version at registration time."""

    column_order: list[str]
    columns: list[ColumnSchema]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> SchemaSnapshot:
        cols = [ColumnSchema(name=str(c), dtype=str(df[c].dtype)) for c in df.columns]
        return cls(column_order=[str(c) for c in df.columns], columns=cols)


class QualitySnapshot(BaseModel):
    """A compact copy of the quality picture for this version, if known."""

    score: float
    total_findings: int
    has_critical: bool
    findings_by_type: dict[str, int] = Field(default_factory=dict)
    missing_cells: int | None = Field(
        default=None, description="Total missing cells, when the count is available."
    )

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any] | None) -> QualitySnapshot | None:
        """Build from an execution report's before/after quality summary dict."""
        if not summary:
            return None
        raw_missing = summary.get("total_missing_cells")
        raw_findings: Mapping[str, Any] = summary.get("findings_by_type") or {}
        return cls(
            score=float(summary.get("score", 0.0)),
            total_findings=int(summary.get("total_findings", 0)),
            has_critical=bool(summary.get("has_critical", False)),
            findings_by_type={str(k): int(v) for k, v in raw_findings.items()},
            missing_cells=None if raw_missing is None else int(raw_missing),
        )


class DatasetVersion(BaseModel):
    """One concrete, auditable version of a dataset."""

    version_model_version: str = VERSION_MODEL_VERSION

    dataset_version_id: str = Field(
        description="Stable identity. '<dataset_id>:raw' or '<dataset_id>:exec-<execution_id>'."
    )
    dataset_id: str = Field(description="The dataset family id (the raw dataset's id).")
    version_number: int = Field(
        ge=0, description="Monotonic index within the family (store-assigned)."
    )
    parent_version_id: str | None = Field(
        default=None, description="The version this one derives from (None for a raw version)."
    )

    kind: DatasetVersionKind
    status: DatasetVersionStatus = DatasetVersionStatus.REGISTERED

    # source / raw identity
    raw_dataset_id: str
    raw_sha256: str | None = None

    created_at: _dt.datetime
    created_by: str = Field(
        description="Producer, e.g. 'data_engine.ingestion' or 'cleaning.executor:1'."
    )

    # file / reference information
    path: Path
    source_format: str = "csv"
    size_bytes: int = Field(ge=0)
    sha256: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)

    schema_snapshot: SchemaSnapshot
    quality: QualitySnapshot | None = None

    # lineage / execution reference
    execution_id: str | None = None
    plan_fingerprint: str | None = None
    lineage_step_count: int | None = None
    applied_operation_ids: list[str] = Field(
        default_factory=list, description="operation_ids of steps that were successfully applied."
    )

    @model_validator(mode="after")
    def _check_identity(self) -> DatasetVersion:
        expected_prefix = f"{self.dataset_id}:"
        if not self.dataset_version_id.startswith(expected_prefix):
            raise ValueError(
                f"dataset_version_id {self.dataset_version_id!r} must start with "
                f"'{expected_prefix}'"
            )
        if self.kind is DatasetVersionKind.RAW:
            if self.parent_version_id is not None:
                raise ValueError("a raw version cannot have a parent_version_id")
            if self.dataset_version_id != f"{self.dataset_id}:{RAW_VERSION_SUFFIX}":
                raise ValueError("a raw version id must be '<dataset_id>:raw'")
        else:
            if self.parent_version_id is None:
                raise ValueError("a processed version must have a parent_version_id")
            if self.execution_id is None:
                raise ValueError("a processed version must have an execution_id")
        if self.column_count != len(self.schema_snapshot.column_order):
            raise ValueError("column_count does not match the schema snapshot")
        return self

    # ---- factories -------------------------------------------------------

    @classmethod
    def raw_version_id(cls, dataset_id: str) -> str:
        return f"{dataset_id}:{RAW_VERSION_SUFFIX}"

    @classmethod
    def from_raw(
        cls,
        reference: DatasetReference,
        df: pd.DataFrame,
        *,
        version_number: int = 0,
        created_by: str = "data_engine.ingestion",
        quality: QualitySnapshot | None = None,
    ) -> DatasetVersion:
        return cls(
            dataset_version_id=cls.raw_version_id(reference.dataset_id),
            dataset_id=reference.dataset_id,
            version_number=version_number,
            parent_version_id=None,
            kind=DatasetVersionKind.RAW,
            raw_dataset_id=reference.dataset_id,
            raw_sha256=reference.sha256,
            created_at=reference.created_at,
            created_by=created_by,
            path=reference.raw_path,
            source_format=reference.source_format.value,
            size_bytes=reference.size_bytes,
            sha256=reference.sha256,
            row_count=len(df),
            column_count=int(df.shape[1]),
            schema_snapshot=SchemaSnapshot.from_dataframe(df),
            quality=quality,
        )

    @classmethod
    def from_execution_report(
        cls,
        report: CleaningExecutionReport,
        *,
        parent_version_id: str,
        schema_source: pd.DataFrame,
        version_number: int,
        created_by: str = "data_engine.cleaning.executor",
    ) -> DatasetVersion:
        ref = report.output_dataset_reference
        if ref is None:
            raise ValueError(
                "execution report has no output_dataset_reference; nothing to register"
            )

        applied = [op.operation_id for op in report.operations if op.status.value == "success"]
        quality = QualitySnapshot.from_summary(report.after_quality_summary)

        return cls(
            dataset_version_id=ref.dataset_id,
            dataset_id=ref.parent_dataset_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            kind=DatasetVersionKind.PROCESSED,
            raw_dataset_id=report.lineage.raw_dataset_id,
            raw_sha256=report.lineage.raw_sha256,
            created_at=ref.created_at,
            created_by=f"{created_by}:{report.execution_engine_version}",
            path=ref.path,
            source_format=ref.source_format,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
            row_count=ref.n_rows,
            column_count=ref.n_columns,
            schema_snapshot=SchemaSnapshot.from_dataframe(schema_source),
            quality=quality,
            execution_id=report.execution_id,
            plan_fingerprint=report.plan_fingerprint,
            lineage_step_count=len(report.lineage.steps),
            applied_operation_ids=applied,
        )
