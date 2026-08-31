"""Validation & Data Lineage (Phase 3).

Makes dataset versions and their provenance first-class, auditable
objects:

* :class:`DatasetVersion` — a typed, JSON-serialisable dataset version
  linking the raw / processed references, a schema snapshot, a quality
  snapshot, and parent/child lineage.
* :class:`DatasetVersionStore` — a deterministic, filesystem-based
  registry (no database). Never overwrites; rejects duplicates and
  conflicts; verifies file hashes.
* :func:`validate_lineage` — checks that an execution report's lineage is
  internally consistent and matches the real files / version records.
  Fails clearly; never silently repairs.

This layer is additive: it does not change ingestion, profiling, quality,
the cleaning planner, or the cleaning executor.
"""

from __future__ import annotations

from .lineage_validation import (
    LineageValidationError,
    LineageValidationResult,
    assert_roundtrip_consistent,
    validate_lineage,
)
from .version_models import (
    ColumnSchema,
    DatasetVersion,
    DatasetVersionKind,
    DatasetVersionStatus,
    QualitySnapshot,
    SchemaSnapshot,
)
from .version_store import (
    ConflictingVersionError,
    DatasetVersionStore,
    DuplicateVersionError,
    VersionIntegrityError,
    VersionNotFoundError,
    VersionStoreError,
)

__all__ = [
    "ColumnSchema",
    "ConflictingVersionError",
    "DatasetVersion",
    "DatasetVersionKind",
    "DatasetVersionStatus",
    "DatasetVersionStore",
    "DuplicateVersionError",
    "LineageValidationError",
    "LineageValidationResult",
    "QualitySnapshot",
    "SchemaSnapshot",
    "VersionIntegrityError",
    "VersionNotFoundError",
    "VersionStoreError",
    "assert_roundtrip_consistent",
    "validate_lineage",
]
