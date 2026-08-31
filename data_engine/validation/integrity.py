"""Version integrity verification.

Verifies a :class:`DatasetVersion` against the actual filesystem state:
the referenced file exists, is readable, and matches the recorded size
and SHA-256; and the version metadata is internally consistent and still
parseable.

Never modifies the dataset file. Never repairs metadata. Returns a
structured :class:`VersionIntegrityResult`; ``raise_for_status()`` raises
the existing :class:`VersionIntegrityError`.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field, ValidationError

from data_engine.ingestion.raw_store import sha256_of_file

from .version_models import DatasetVersion, DatasetVersionKind
from .version_store import DatasetVersionStore, VersionIntegrityError


class VersionIntegrityResult(BaseModel):
    dataset_version_id: str
    valid: bool
    checks_run: int
    errors: list[str] = Field(default_factory=list)

    def raise_for_status(self) -> None:
        if not self.valid:
            raise VersionIntegrityError(f"{self.dataset_version_id}: " + "; ".join(self.errors))


def _check_identity(version: DatasetVersion, errors: list[str]) -> None:
    if not version.dataset_version_id.startswith(f"{version.dataset_id}:"):
        errors.append("dataset_version_id is not prefixed with dataset_id")
    if version.raw_dataset_id != version.dataset_id:
        errors.append(
            f"raw_dataset_id {version.raw_dataset_id!r} != dataset_id {version.dataset_id!r}"
        )
    if version.column_count != len(version.schema_snapshot.column_order):
        errors.append("column_count does not match schema_snapshot.column_order")
    if len(version.schema_snapshot.columns) != len(version.schema_snapshot.column_order):
        errors.append("schema_snapshot columns / column_order length mismatch")

    if version.kind is DatasetVersionKind.RAW:
        if version.parent_version_id is not None:
            errors.append("raw version has a parent_version_id")
        if version.dataset_version_id != DatasetVersion.raw_version_id(version.dataset_id):
            errors.append("raw version id is not '<dataset_id>:raw'")
    else:
        if version.parent_version_id is None:
            errors.append("processed version has no parent_version_id")
        if version.execution_id is None:
            errors.append("processed version has no execution_id")
        else:
            expected = f"{version.dataset_id}:exec-{version.execution_id}"
            if version.dataset_version_id != expected:
                errors.append(
                    f"processed version id {version.dataset_version_id!r} does not encode its "
                    f"execution_id (expected {expected!r})"
                )


def verify_version_integrity(version: DatasetVersion) -> VersionIntegrityResult:
    """Verify one version against the filesystem. Works for raw and processed."""
    errors: list[str] = []
    checks = 0

    # metadata still parses / round-trips
    checks += 1
    try:
        DatasetVersion.model_validate_json(version.model_dump_json())
    except ValidationError as exc:  # pragma: no cover - a valid object always round-trips
        errors.append(f"version metadata does not round-trip: {exc}")

    # identity internally consistent
    checks += 1
    _check_identity(version, errors)

    # file exists / readable / size / hash
    path = version.path
    checks += 1
    if not path.exists() or not path.is_file():
        errors.append(f"referenced file is missing: {path}")
        return VersionIntegrityResult(
            dataset_version_id=version.dataset_version_id,
            valid=not errors,
            checks_run=checks,
            errors=errors,
        )

    checks += 1
    if not os.access(path, os.R_OK):
        errors.append(f"referenced file is not readable: {path}")
        return VersionIntegrityResult(
            dataset_version_id=version.dataset_version_id,
            valid=not errors,
            checks_run=checks,
            errors=errors,
        )

    checks += 1
    actual_size = path.stat().st_size
    if actual_size != version.size_bytes:
        errors.append(f"file size mismatch: recorded {version.size_bytes}, actual {actual_size}")

    checks += 1
    actual_sha = sha256_of_file(path)
    if actual_sha != version.sha256:
        errors.append(f"file SHA-256 mismatch: recorded {version.sha256}, actual {actual_sha}")

    return VersionIntegrityResult(
        dataset_version_id=version.dataset_version_id,
        valid=not errors,
        checks_run=checks,
        errors=errors,
    )


def verify_registered_version(
    store: DatasetVersionStore, dataset_version_id: str
) -> VersionIntegrityResult:
    """Load a registered version record and verify its integrity.

    A corrupted / unparseable record file is reported (not raised).
    """
    path = store.version_file_path(dataset_version_id)
    if not path.exists():
        return VersionIntegrityResult(
            dataset_version_id=dataset_version_id,
            valid=False,
            checks_run=1,
            errors=[f"no registered record file at {path}"],
        )
    try:
        version = DatasetVersion.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as exc:
        return VersionIntegrityResult(
            dataset_version_id=dataset_version_id,
            valid=False,
            checks_run=1,
            errors=[f"version record is corrupted or unparseable: {exc}"],
        )
    if version.dataset_version_id != dataset_version_id:
        return VersionIntegrityResult(
            dataset_version_id=dataset_version_id,
            valid=False,
            checks_run=1,
            errors=[
                (
                    f"record file {path.name} holds id {version.dataset_version_id!r}, "
                    f"expected {dataset_version_id!r}"
                )
            ],
        )
    return verify_version_integrity(version)
