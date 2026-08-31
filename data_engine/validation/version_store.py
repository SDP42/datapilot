"""A deterministic, filesystem-based dataset-version registry.

Design mirrors ``data_engine.cleaning.ProcessedDataStore``: one directory
per dataset family, one read-only JSON file per version, no database.

    data/versions/<dataset_id>/
        raw.json
        exec-<execution_id>.json
        ...

Guarantees:

* never touches the raw or processed CSV files
* never silently overwrites a registered version
* rejects duplicate and conflicting registrations
* preserves parent -> child relationships
* verifies the referenced file's sha256 where the file is reachable
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_engine.cleaning.execution_models import CleaningExecutionReport
from data_engine.ingestion.raw_store import sha256_of_file
from datapilot import paths
from datapilot.contracts import DatasetReference

from .version_models import (
    RAW_VERSION_SUFFIX,
    DatasetVersion,
    DatasetVersionKind,
    DatasetVersionStatus,
    QualitySnapshot,
)

_READ_ONLY = 0o444


class VersionStoreError(Exception):
    """Base class for version-store failures."""


class DuplicateVersionError(VersionStoreError):
    """The exact version is already registered."""


class ConflictingVersionError(VersionStoreError):
    """A different version is already registered under this identity, or a
    parent/child relationship is invalid."""


class VersionNotFoundError(VersionStoreError):
    """No version with the requested id is registered."""


class VersionIntegrityError(VersionStoreError):
    """The referenced file does not match the recorded hash / metadata."""


def _file_stem(version: DatasetVersion) -> str:
    if version.kind is DatasetVersionKind.RAW:
        return RAW_VERSION_SUFFIX
    return f"exec-{version.execution_id}"


class DatasetVersionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @classmethod
    def default(cls) -> DatasetVersionStore:
        return cls(paths.DATA_VERSIONS_DIR)

    # ---- paths ----------------------------------------------------------

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / dataset_id

    def _version_path(self, version: DatasetVersion) -> Path:
        return self.dataset_dir(version.dataset_id) / f"{_file_stem(version)}.json"

    def version_file_path(self, dataset_version_id: str) -> Path:
        """The on-disk record path for a version id, computed without parsing.

        ``<dataset_id>:raw`` -> ``<root>/<dataset_id>/raw.json``;
        ``<dataset_id>:exec-<id>`` -> ``<root>/<dataset_id>/exec-<id>.json``.
        """
        dataset_id, sep, stem = dataset_version_id.partition(":")
        if not sep or not stem:
            raise VersionStoreError(f"malformed dataset_version_id {dataset_version_id!r}")
        return self.dataset_dir(dataset_id) / f"{stem}.json"

    def iter_version_files(self, dataset_id: str) -> list[Path]:
        """Every version-record file for a family, sorted, without parsing."""
        directory = self.dataset_dir(dataset_id)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.json"))

    # ---- read ----------------------------------------------------------

    def get(self, dataset_version_id: str) -> DatasetVersion:
        dataset_id = dataset_version_id.split(":", 1)[0]
        directory = self.dataset_dir(dataset_id)
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                version = DatasetVersion.model_validate_json(path.read_text(encoding="utf-8"))
                if version.dataset_version_id == dataset_version_id:
                    return version
        raise VersionNotFoundError(f"no registered version {dataset_version_id!r}")

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        directory = self.dataset_dir(dataset_id)
        if not directory.is_dir():
            return []
        versions = [
            DatasetVersion.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(directory.glob("*.json"))
        ]
        return sorted(versions, key=lambda v: v.version_number)

    def children(self, dataset_version_id: str) -> list[DatasetVersion]:
        dataset_id = dataset_version_id.split(":", 1)[0]
        return [
            v for v in self.list_versions(dataset_id) if v.parent_version_id == dataset_version_id
        ]

    def exists(self, dataset_version_id: str) -> bool:
        try:
            self.get(dataset_version_id)
        except VersionNotFoundError:
            return False
        return True

    # ---- write ----------------------------------------------------------

    def register(self, version: DatasetVersion) -> DatasetVersion:
        """Record a version. Rejects duplicates, conflicts, and bad parents."""
        self._validate_integrity(version)
        self._validate_parent(version)

        path = self._version_path(version)
        if path.exists():
            existing = DatasetVersion.model_validate_json(path.read_text(encoding="utf-8"))
            if (
                existing.dataset_version_id == version.dataset_version_id
                and existing.sha256 == version.sha256
                and existing.parent_version_id == version.parent_version_id
            ):
                raise DuplicateVersionError(
                    f"version {version.dataset_version_id!r} is already registered"
                )
            raise ConflictingVersionError(
                f"a different version is already registered at {path.name!r} for dataset "
                f"{version.dataset_id!r}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(version.model_dump_json(indent=2), encoding="utf-8")
        path.chmod(_READ_ONLY)
        return version

    def _validate_integrity(self, version: DatasetVersion) -> None:
        if version.path.exists():
            actual = sha256_of_file(version.path)
            if actual != version.sha256:
                raise VersionIntegrityError(
                    f"sha256 mismatch for {version.path}: recorded {version.sha256}, "
                    f"actual {actual}"
                )

    def _validate_parent(self, version: DatasetVersion) -> None:
        if version.parent_version_id is None:
            return
        try:
            parent = self.get(version.parent_version_id)
        except VersionNotFoundError as exc:
            raise ConflictingVersionError(
                f"parent version {version.parent_version_id!r} is not registered"
            ) from exc
        if parent.dataset_id != version.dataset_id:
            raise ConflictingVersionError(
                "child version belongs to a different dataset family than its parent"
            )
        if parent.raw_dataset_id != version.raw_dataset_id:
            raise ConflictingVersionError(
                "child version and parent disagree on the raw dataset identity"
            )
        if parent.dataset_version_id == version.dataset_version_id:
            raise ConflictingVersionError("a version cannot be its own parent")

    # ---- convenience --------------------------------------------------

    def _next_version_number(self, dataset_id: str) -> int:
        existing = self.list_versions(dataset_id)
        return (max(v.version_number for v in existing) + 1) if existing else 0

    def register_raw(
        self,
        reference: DatasetReference,
        df: pd.DataFrame,
        *,
        created_by: str = "data_engine.ingestion",
        quality: QualitySnapshot | None = None,
    ) -> DatasetVersion:
        version = DatasetVersion.from_raw(
            reference,
            df,
            version_number=self._next_version_number(reference.dataset_id),
            created_by=created_by,
            quality=quality,
        )
        return self.register(version)

    def register_from_execution(
        self,
        report: CleaningExecutionReport,
        *,
        parent_version_id: str,
        cleaned_df: pd.DataFrame | None = None,
        created_by: str = "data_engine.cleaning.executor",
    ) -> DatasetVersion:
        ref = report.output_dataset_reference
        if ref is None:
            raise ValueError(
                "execution report has no output_dataset_reference; nothing to register"
            )

        schema_source = cleaned_df
        if schema_source is None:
            if not ref.path.exists():
                raise VersionIntegrityError(
                    f"processed file {ref.path} is missing and no cleaned_df was supplied"
                )
            schema_source = pd.read_csv(ref.path)

        version = DatasetVersion.from_execution_report(
            report,
            parent_version_id=parent_version_id,
            schema_source=schema_source,
            version_number=self._next_version_number(ref.parent_dataset_id),
            created_by=created_by,
        )
        return self.register(version)

    def mark_status(self, dataset_version_id: str, status: DatasetVersionStatus) -> DatasetVersion:
        """Explicit status change (never silent). Rewrites the single record."""
        version = self.get(dataset_version_id)
        updated = DatasetVersion.model_validate(
            {**version.model_dump(mode="json"), "status": status.value}
        )
        path = self._version_path(updated)
        if path.exists():
            path.chmod(0o644)
            path.unlink()
        path.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
        path.chmod(_READ_ONLY)
        return updated
