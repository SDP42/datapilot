"""Version-store consistency checks for a whole dataset family.

Validates every registered version for one ``dataset_id`` together:
record parseability, duplicate/conflicting identities, raw-identity
agreement, structural lineage (reusing :class:`LineageGraph`), and
per-version file integrity.

Reports **all** discovered errors — it does not stop at the first — and
never repairs anything.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from data_engine.cleaning.execution_models import CleaningExecutionReport, ExecutionStatus
from datapilot.contracts import DatasetReference

from .integrity import VersionIntegrityResult, verify_version_integrity
from .lineage_graph import LineageGraph, LineageGraphError
from .lineage_validation import LineageValidationResult, validate_lineage
from .version_models import DatasetVersion, DatasetVersionKind
from .version_store import DatasetVersionStore, VersionStoreError


class FamilyConsistencyResult(BaseModel):
    dataset_id: str
    valid: bool
    version_count: int
    checks_run: int
    errors: list[str] = Field(default_factory=list)
    integrity: dict[str, VersionIntegrityResult] = Field(default_factory=dict)

    def raise_for_status(self) -> None:
        if not self.valid:
            raise VersionStoreError(f"{self.dataset_id}: " + "; ".join(self.errors))


def _load_records(
    store: DatasetVersionStore, dataset_id: str
) -> tuple[list[DatasetVersion], list[str]]:
    versions: list[DatasetVersion] = []
    errors: list[str] = []
    for path in store.iter_version_files(dataset_id):
        try:
            versions.append(DatasetVersion.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValidationError, ValueError) as exc:
            errors.append(f"version record {path.name!r} is corrupted or unparseable: {exc}")
    return versions, errors


def check_family_consistency(
    store: DatasetVersionStore, dataset_id: str, *, verify_files: bool = True
) -> FamilyConsistencyResult:
    versions, errors = _load_records(store, dataset_id)
    checks = 1  # record parseability

    # duplicate / conflicting identities
    checks += 1
    by_id: dict[str, list[DatasetVersion]] = {}
    for version in versions:
        by_id.setdefault(version.dataset_version_id, []).append(version)
    for vid, records in by_id.items():
        if len(records) > 1:
            errors.append(f"duplicate / conflicting records for version id {vid!r}")

    unique = [records[0] for records in by_id.values()]

    # raw-identity agreement
    checks += 1
    for version in unique:
        if version.raw_dataset_id != dataset_id:
            errors.append(
                f"{version.dataset_version_id}: raw_dataset_id {version.raw_dataset_id!r} != "
                f"family {dataset_id!r}"
            )
        if not version.dataset_version_id.startswith(f"{dataset_id}:"):
            errors.append(
                f"{version.dataset_version_id!r} does not belong to family {dataset_id!r}"
            )
    raw_ids = {v.raw_dataset_id for v in unique}
    if len(raw_ids) > 1:
        errors.append(f"family disagrees on raw dataset identity: {sorted(raw_ids)}")

    # structural lineage: collect what we can, then use LineageGraph as a backstop
    checks += 1
    ids = {v.dataset_version_id for v in unique}
    roots = [v for v in unique if v.parent_version_id is None]
    for version in unique:
        parent_id = version.parent_version_id
        if parent_id is None:
            continue
        if parent_id == version.dataset_version_id:
            errors.append(f"{version.dataset_version_id}: version is its own parent")
        elif parent_id not in ids:
            if parent_id.split(":", 1)[0] != dataset_id:
                errors.append(
                    f"{version.dataset_version_id}: parent {parent_id!r} is from a foreign family"
                )
            else:
                errors.append(
                    f"{version.dataset_version_id}: parent {parent_id!r} is not registered"
                )
    if len(roots) == 0 and unique:
        errors.append("family has no root version")
    elif len(roots) > 1:
        errors.append(f"family has multiple roots: {sorted(r.dataset_version_id for r in roots)}")
    elif roots and roots[0].kind is not DatasetVersionKind.RAW:
        errors.append(f"root version {roots[0].dataset_version_id!r} is not of kind 'raw'")

    checks += 1
    try:
        LineageGraph(unique)
    except LineageGraphError as exc:
        message = f"lineage graph: {exc}"
        if message not in errors:
            errors.append(message)

    # per-version file integrity
    if verify_files:
        checks += 1
        integrity: dict[str, VersionIntegrityResult] = {}
        for version in sorted(unique, key=lambda v: v.dataset_version_id):
            result = verify_version_integrity(version)
            integrity[version.dataset_version_id] = result
            if not result.valid:
                errors.append(
                    f"{version.dataset_version_id}: file integrity failed "
                    f"({'; '.join(result.errors)})"
                )
    else:
        integrity = {}

    return FamilyConsistencyResult(
        dataset_id=dataset_id,
        valid=not errors,
        version_count=len(unique),
        checks_run=checks,
        errors=errors,
        integrity=integrity,
    )


def check_version_lineage_binding(
    processed_version: DatasetVersion,
    report: CleaningExecutionReport,
    *,
    parent_version: DatasetVersion | None = None,
    raw_reference: DatasetReference | None = None,
) -> LineageValidationResult:
    """Verify a registered processed version against its execution report.

    Reuses :func:`validate_lineage` and then checks the extra
    registered-version <-> report relationships:

    * processed version id <-> processed dataset reference / execution id
    * plan fingerprint <-> report
    * lineage_step_count <-> report lineage steps
    * applied_operation_ids <-> successful execution records
    * row / column / sha counts <-> processed dataset reference
    """
    base = validate_lineage(
        report,
        raw_reference=raw_reference,
        parent_version=parent_version,
        child_version=processed_version,
    )
    errors = list(base.errors)
    checks = base.checks_run

    checks += 1
    ref = report.output_dataset_reference
    if ref is None:
        errors.append("execution report has no output_dataset_reference")
        return LineageValidationResult(valid=False, checks_run=checks, errors=errors)

    if processed_version.dataset_version_id != ref.dataset_id:
        errors.append("processed version id does not match the processed dataset reference id")
    if processed_version.execution_id != report.execution_id:
        errors.append("processed version execution_id does not match the execution report")
    if (
        processed_version.execution_id is not None
        and processed_version.dataset_version_id
        != f"{processed_version.dataset_id}:exec-{processed_version.execution_id}"
    ):
        errors.append("processed version id does not encode its execution_id")

    checks += 1
    if processed_version.plan_fingerprint is None:
        errors.append("registered processed version has no plan_fingerprint")
    elif processed_version.plan_fingerprint != report.plan_fingerprint:
        errors.append("registered processed version plan_fingerprint does not match the report")

    checks += 1
    if (
        processed_version.lineage_step_count is not None
        and processed_version.lineage_step_count != len(report.lineage.steps)
    ):
        errors.append(
            f"registered lineage_step_count {processed_version.lineage_step_count} != "
            f"{len(report.lineage.steps)} report lineage steps"
        )

    checks += 1
    successful = sorted(
        op.operation_id for op in report.operations if op.status is ExecutionStatus.SUCCESS
    )
    if sorted(processed_version.applied_operation_ids) != successful:
        errors.append(
            "registered applied_operation_ids do not match the successful execution records"
        )

    checks += 1
    if processed_version.sha256 != ref.sha256:
        errors.append("registered processed sha256 does not match the processed dataset reference")
    if processed_version.row_count != ref.n_rows:
        errors.append("registered row_count does not match the processed dataset reference")
    if processed_version.column_count != ref.n_columns:
        errors.append("registered column_count does not match the processed dataset reference")

    if parent_version is not None:
        checks += 1
        if parent_version.raw_dataset_id != processed_version.raw_dataset_id:
            errors.append(
                "parent version raw dataset identity does not match the processed version"
            )
        if parent_version.dataset_version_id != processed_version.parent_version_id:
            errors.append("processed version does not reference the given parent version")

    return LineageValidationResult(valid=not errors, checks_run=checks, errors=errors)
