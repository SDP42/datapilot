"""Validation & Data Lineage (Phase 3).

Makes dataset versions and their provenance first-class, auditable
objects, and adds deterministic navigation and comparison on top:

* :class:`DatasetVersion` / :class:`DatasetVersionStore` — typed dataset
  versions and a filesystem-only registry (no database). Never
  overwrites; rejects duplicates and conflicts; verifies file hashes.
* :func:`validate_lineage` — checks an execution report's provenance
  against the real files and version records. Fails clearly; never
  silently repairs.
* :class:`LineageGraph` — an in-memory, read-only DAG over one dataset
  family's registered versions (parent / children / ancestors /
  descendants / root / path). Raises on missing parents, cross-family
  parents, self-parents, multiple roots, and cycles.
* :func:`execute_and_register_cleaning` — an *opt-in* wrapper that runs
  ``execute_cleaning`` unchanged and then registers the versions and
  validates the lineage. The default cleaning flow is untouched.
* :func:`diff_versions` — deterministic metadata / schema / quality /
  content comparison of two registered versions in the same family.

Filesystem-only. Additive: it does not change ingestion, profiling,
quality, the cleaning planner, or the cleaning executor.
"""

from __future__ import annotations

from .auto_register import (
    AutoRegistrationError,
    RegisteredCleaningResult,
    execute_and_register_cleaning,
)
from .lineage_graph import LineageGraph, LineageGraphError
from .lineage_validation import (
    LineageValidationError,
    LineageValidationResult,
    assert_roundtrip_consistent,
    validate_lineage,
)
from .version_diff import (
    ContentDiff,
    DtypeChange,
    FieldChange,
    LineageRelationship,
    MetadataDiff,
    QualityDiff,
    SchemaDiff,
    VersionDiff,
    VersionDiffError,
    diff_registered_versions,
    diff_versions,
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
    "AutoRegistrationError",
    "ColumnSchema",
    "ConflictingVersionError",
    "ContentDiff",
    "DatasetVersion",
    "DatasetVersionKind",
    "DatasetVersionStatus",
    "DatasetVersionStore",
    "DtypeChange",
    "DuplicateVersionError",
    "FieldChange",
    "LineageGraph",
    "LineageGraphError",
    "LineageRelationship",
    "LineageValidationError",
    "LineageValidationResult",
    "MetadataDiff",
    "QualityDiff",
    "QualitySnapshot",
    "RegisteredCleaningResult",
    "SchemaDiff",
    "SchemaSnapshot",
    "VersionDiff",
    "VersionDiffError",
    "VersionIntegrityError",
    "VersionNotFoundError",
    "VersionStoreError",
    "assert_roundtrip_consistent",
    "diff_registered_versions",
    "diff_versions",
    "execute_and_register_cleaning",
    "validate_lineage",
]
