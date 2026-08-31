"""Lineage validation — makes provenance *checkable*, not just descriptive.

``validate_lineage`` inspects a :class:`CleaningExecutionReport` (and,
where available, the raw file, the processed file, and the parent/child
:class:`DatasetVersion` records) and returns a
:class:`LineageValidationResult`. It never mutates anything and never
silently repairs an inconsistency — a problem is reported as an error.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from data_engine.cleaning.execution_models import (
    CleaningExecutionReport,
    ExecutionStatus,
    OperationExecution,
)
from data_engine.cleaning.models import OperationCategory
from data_engine.ingestion.raw_store import sha256_of_file
from datapilot.contracts import DatasetReference

from .version_models import DatasetVersion

_TRANSFORMING = {
    OperationCategory.DATA_TRANSFORMATION,
}


class LineageValidationResult(BaseModel):
    valid: bool
    checks_run: int
    errors: list[str] = Field(default_factory=list)

    def raise_for_status(self) -> None:
        if not self.valid:
            raise LineageValidationError("; ".join(self.errors))


class LineageValidationError(Exception):
    """Raised by ``LineageValidationResult.raise_for_status`` on an invalid lineage."""


def _op_index(report: CleaningExecutionReport) -> dict[str, OperationExecution]:
    return {op.operation_id: op for op in report.operations}


def validate_lineage(
    report: CleaningExecutionReport,
    *,
    raw_reference: DatasetReference | None = None,
    processed_path: Path | None = None,
    parent_version: DatasetVersion | None = None,
    child_version: DatasetVersion | None = None,
) -> LineageValidationResult:
    errors: list[str] = []
    checks = 0
    ops = _op_index(report)
    lineage = report.lineage

    # 1. raw dataset identity matches the execution input
    checks += 1
    if report.dataset_id != lineage.raw_dataset_id:
        errors.append(
            f"raw identity mismatch: report.dataset_id={report.dataset_id!r} vs "
            f"lineage.raw_dataset_id={lineage.raw_dataset_id!r}"
        )
    if raw_reference is not None and raw_reference.dataset_id != report.dataset_id:
        errors.append(
            f"raw_reference.dataset_id={raw_reference.dataset_id!r} does not match the report"
        )

    # 2. raw sha256 matches the actual raw file where available
    checks += 1
    if raw_reference is not None:
        if lineage.raw_sha256 is not None and lineage.raw_sha256 != raw_reference.sha256:
            errors.append("lineage.raw_sha256 does not match raw_reference.sha256")
        if raw_reference.raw_path.exists():
            actual = sha256_of_file(raw_reference.raw_path)
            if actual != raw_reference.sha256:
                errors.append("raw file on disk does not match raw_reference.sha256")
            if lineage.raw_sha256 is not None and actual != lineage.raw_sha256:
                errors.append("raw file on disk does not match lineage.raw_sha256")

    # 3. plan fingerprint matches the execution report
    checks += 1
    if report.plan_fingerprint != lineage.plan_fingerprint:
        errors.append(
            f"plan fingerprint mismatch: report={report.plan_fingerprint!r} vs "
            f"lineage={lineage.plan_fingerprint!r}"
        )

    # 4/5. every lineage step references a real operation and preserves its finding
    checks += 1
    if len(lineage.steps) != len(report.operations):
        errors.append(
            f"lineage has {len(lineage.steps)} steps but the report has "
            f"{len(report.operations)} operations"
        )
    for step in lineage.steps:
        op = ops.get(step.operation_id)
        if op is None:
            errors.append(
                f"lineage step {step.index} references unknown operation {step.operation_id!r}"
            )
            continue
        if step.operation_type != op.operation_type:
            errors.append(f"lineage step {step.index}: operation_type disagrees with the record")
        if step.status != op.status:
            errors.append(
                f"lineage step {step.index}: status {step.status.value!r} disagrees with the "
                f"record status {op.status.value!r}"
            )
        if step.source_finding_id != op.source_finding_id:
            errors.append(f"lineage step {step.index}: source_finding_id disagrees with the record")
        if op.operation_category in _TRANSFORMING and not step.source_finding_id:
            errors.append(
                f"lineage step {step.index}: transforming operation has no source_finding_id"
            )

    # 6. processed dataset sha256 matches the stored processed file
    checks += 1
    ref = report.output_dataset_reference
    if ref is not None:
        if lineage.processed_sha256 is not None and lineage.processed_sha256 != ref.sha256:
            errors.append("lineage.processed_sha256 does not match output_dataset_reference.sha256")
        check_path = processed_path or ref.path
        if check_path is not None and Path(check_path).exists():
            actual = sha256_of_file(Path(check_path))
            if actual != ref.sha256:
                errors.append(
                    "processed file on disk does not match output_dataset_reference.sha256"
                )
            if lineage.processed_sha256 is not None and actual != lineage.processed_sha256:
                errors.append("processed file on disk does not match lineage.processed_sha256")

    # 7. failed/skipped operations cannot be recorded as successfully applied
    checks += 1
    success_steps = [s for s in lineage.steps if s.status is ExecutionStatus.SUCCESS]
    for step in success_steps:
        op = ops.get(step.operation_id)
        if op is not None and op.status is not ExecutionStatus.SUCCESS:
            errors.append(
                f"lineage step {step.index} claims SUCCESS but the operation record is "
                f"{op.status.value!r}"
            )
    declared_success = report.operations_succeeded
    counted_success = sum(1 for op in report.operations if op.status is ExecutionStatus.SUCCESS)
    if declared_success != counted_success:
        errors.append(
            f"operations_succeeded={declared_success} but {counted_success} records are SUCCESS"
        )

    # 8. parent / child dataset-version relationships
    checks += 1
    if child_version is not None:
        if ref is not None and child_version.dataset_version_id != ref.dataset_id:
            errors.append(
                "child_version.dataset_version_id does not match output_dataset_reference.dataset_id"
            )
        if child_version.execution_id != report.execution_id:
            errors.append("child_version.execution_id does not match report.execution_id")
        if child_version.dataset_id != report.dataset_id:
            errors.append("child_version belongs to a different dataset than the report")
        if child_version.plan_fingerprint not in (None, report.plan_fingerprint):
            errors.append("child_version.plan_fingerprint does not match the report")
    if parent_version is not None and child_version is not None:
        if child_version.parent_version_id != parent_version.dataset_version_id:
            errors.append("child_version does not reference the given parent_version")
        if child_version.dataset_id != parent_version.dataset_id:
            errors.append("child and parent versions are in different dataset families")
        if child_version.raw_dataset_id != parent_version.raw_dataset_id:
            errors.append("child and parent versions disagree on the raw dataset identity")
        if parent_version.dataset_version_id == child_version.dataset_version_id:
            errors.append("a version cannot be its own parent")

    return LineageValidationResult(valid=not errors, checks_run=checks, errors=errors)


def assert_roundtrip_consistent(
    report: CleaningExecutionReport, **kwargs: object
) -> LineageValidationResult:
    """Serialise the report to JSON, reload it, and re-validate its lineage.

    Guarantees the lineage is not just valid in memory but survives a full
    JSON round-trip without losing information.
    """
    reloaded = CleaningExecutionReport.model_validate_json(report.model_dump_json())
    before = validate_lineage(report, **kwargs)  # type: ignore[arg-type]
    after = validate_lineage(reloaded, **kwargs)  # type: ignore[arg-type]
    if before.model_dump() != after.model_dump():
        return LineageValidationResult(
            valid=False,
            checks_run=after.checks_run,
            errors=[*after.errors, "lineage validation result changed after JSON round-trip"],
        )
    return after
