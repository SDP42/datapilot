"""Deterministic cross-version diffing for two registered DatasetVersions.

Two layers:

* **metadata + schema + quality** — computed from the ``DatasetVersion``
  records alone.
* **content** — computed only when both dataset files are present and
  readable. When a file is missing this is reported explicitly as
  *unavailable* — never silently skipped, never assumed identical.

Comparing versions from different ``dataset_id`` families is rejected.
All output is Pydantic v2, JSON-serialisable, and deterministically
ordered.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from .lineage_graph import LineageGraph
from .version_models import DatasetVersion

DIFF_MODEL_VERSION = "1"

Scalar = str | int | float | bool | None


class VersionDiffError(Exception):
    """The two versions cannot be compared (e.g. different dataset families)."""


class LineageRelationship(str, Enum):
    SAME_VERSION = "same_version"
    ANCESTOR_TO_DESCENDANT = "ancestor_to_descendant"  # `from` is an ancestor of `to`
    DESCENDANT_TO_ANCESTOR = "descendant_to_ancestor"  # `from` is a descendant of `to`
    SIBLINGS = "siblings"  # same family, neither is an ancestor of the other
    UNKNOWN_SAME_FAMILY = "unknown_same_family"  # same family, no graph supplied to resolve


class FieldChange(BaseModel):
    field: str
    before: Scalar
    after: Scalar

    @property
    def changed(self) -> bool:
        return self.before != self.after


class DtypeChange(BaseModel):
    column: str
    before: str
    after: str


class MetadataDiff(BaseModel):
    from_version_id: str
    to_version_id: str
    changes: list[FieldChange] = Field(default_factory=list)

    @property
    def changed_fields(self) -> list[str]:
        return [c.field for c in self.changes if c.changed]


class SchemaDiff(BaseModel):
    added_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
    dtype_changes: list[DtypeChange] = Field(default_factory=list)
    column_order_changed: bool = False
    common_columns: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.added_columns
            or self.removed_columns
            or self.dtype_changes
            or self.column_order_changed
        )


class QualityDiff(BaseModel):
    available: bool
    score_before: float | None = None
    score_after: float | None = None
    total_findings_before: int | None = None
    total_findings_after: int | None = None
    has_critical_before: bool | None = None
    has_critical_after: bool | None = None
    missing_cells_before: int | None = None
    missing_cells_after: int | None = None
    findings_by_type_changes: list[FieldChange] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)


class ContentDiff(BaseModel):
    available: bool
    unavailable_reason: str | None = None
    row_count_before: int | None = None
    row_count_after: int | None = None
    duplicate_rows_before: int | None = None
    duplicate_rows_after: int | None = None
    missing_cells_before: int | None = None
    missing_cells_after: int | None = None
    identical_content: bool | None = None


class VersionDiff(BaseModel):
    diff_model_version: str = DIFF_MODEL_VERSION
    dataset_id: str
    from_version_id: str
    to_version_id: str
    lineage_relationship: LineageRelationship
    ancestor_version_id: str | None = None
    descendant_version_id: str | None = None
    metadata: MetadataDiff
    schema_diff: SchemaDiff
    quality: QualityDiff
    content: ContentDiff


# ---- builders --------------------------------------------------------


def _metadata_diff(a: DatasetVersion, b: DatasetVersion) -> MetadataDiff:
    fields: list[tuple[str, Scalar, Scalar]] = [
        ("parent_version_id", a.parent_version_id, b.parent_version_id),
        ("kind", a.kind.value, b.kind.value),
        ("status", a.status.value, b.status.value),
        ("row_count", a.row_count, b.row_count),
        ("column_count", a.column_count, b.column_count),
        ("size_bytes", a.size_bytes, b.size_bytes),
        ("sha256", a.sha256, b.sha256),
        ("version_number", a.version_number, b.version_number),
    ]
    return MetadataDiff(
        from_version_id=a.dataset_version_id,
        to_version_id=b.dataset_version_id,
        changes=[FieldChange(field=f, before=x, after=y) for f, x, y in fields],
    )


def _schema_diff(a: DatasetVersion, b: DatasetVersion) -> SchemaDiff:
    a_cols = {c.name: c.dtype for c in a.schema_snapshot.columns}
    b_cols = {c.name: c.dtype for c in b.schema_snapshot.columns}
    added = sorted(set(b_cols) - set(a_cols))
    removed = sorted(set(a_cols) - set(b_cols))
    common = sorted(set(a_cols) & set(b_cols))
    dtype_changes = [
        DtypeChange(column=name, before=a_cols[name], after=b_cols[name])
        for name in common
        if a_cols[name] != b_cols[name]
    ]
    order_changed = [c for c in a.schema_snapshot.column_order if c in b_cols] != [
        c for c in b.schema_snapshot.column_order if c in a_cols
    ]
    return SchemaDiff(
        added_columns=added,
        removed_columns=removed,
        dtype_changes=dtype_changes,
        column_order_changed=order_changed,
        common_columns=common,
    )


def _quality_diff(a: DatasetVersion, b: DatasetVersion) -> QualityDiff:
    qa, qb = a.quality, b.quality
    if qa is None and qb is None:
        return QualityDiff(available=False)

    improvements: list[str] = []
    regressions: list[str] = []
    changes: list[FieldChange] = []

    sa = qa.score if qa else None
    sb = qb.score if qb else None
    if sa is not None and sb is not None and sa != sb:
        (improvements if sb > sa else regressions).append(f"quality_score: {sa} -> {sb}")

    ta = qa.total_findings if qa else None
    tb = qb.total_findings if qb else None
    if ta is not None and tb is not None and ta != tb:
        (improvements if tb < ta else regressions).append(f"total_findings: {ta} -> {tb}")

    ma = qa.missing_cells if qa else None
    mb = qb.missing_cells if qb else None
    if ma is not None and mb is not None and ma != mb:
        (improvements if mb < ma else regressions).append(f"missing_cells: {ma} -> {mb}")

    types = sorted(set(qa.findings_by_type if qa else {}) | set(qb.findings_by_type if qb else {}))
    for finding_type in types:
        before = qa.findings_by_type.get(finding_type, 0) if qa else 0
        after = qb.findings_by_type.get(finding_type, 0) if qb else 0
        if before != after:
            changes.append(FieldChange(field=finding_type, before=before, after=after))
            (improvements if after < before else regressions).append(
                f"{finding_type}: {before} -> {after}"
            )

    return QualityDiff(
        available=True,
        score_before=sa,
        score_after=sb,
        total_findings_before=ta,
        total_findings_after=tb,
        has_critical_before=qa.has_critical if qa else None,
        has_critical_after=qb.has_critical if qb else None,
        missing_cells_before=ma,
        missing_cells_after=mb,
        findings_by_type_changes=changes,
        improvements=improvements,
        regressions=regressions,
    )


def _content_diff(a: DatasetVersion, b: DatasetVersion) -> ContentDiff:
    for label, version in (("from", a), ("to", b)):
        path = Path(version.path)
        if not path.exists() or not path.is_file():
            return ContentDiff(
                available=False,
                unavailable_reason=(
                    f"the {label} version's data file is not available: {version.path}"
                ),
            )

    df_a = pd.read_csv(a.path)
    df_b = pd.read_csv(b.path)
    identical = df_a.equals(df_b)
    return ContentDiff(
        available=True,
        row_count_before=len(df_a),
        row_count_after=len(df_b),
        duplicate_rows_before=int(df_a.duplicated().sum()),
        duplicate_rows_after=int(df_b.duplicated().sum()),
        missing_cells_before=int(df_a.isna().sum().sum()),
        missing_cells_after=int(df_b.isna().sum().sum()),
        identical_content=bool(identical),
    )


def _relationship(
    a: DatasetVersion, b: DatasetVersion, graph: LineageGraph | None
) -> tuple[LineageRelationship, str | None, str | None]:
    if a.dataset_version_id == b.dataset_version_id:
        return LineageRelationship.SAME_VERSION, None, None
    if graph is not None:
        if graph.is_ancestor(a.dataset_version_id, b.dataset_version_id):
            return (
                LineageRelationship.ANCESTOR_TO_DESCENDANT,
                a.dataset_version_id,
                b.dataset_version_id,
            )
        if graph.is_ancestor(b.dataset_version_id, a.dataset_version_id):
            return (
                LineageRelationship.DESCENDANT_TO_ANCESTOR,
                b.dataset_version_id,
                a.dataset_version_id,
            )
        return LineageRelationship.SIBLINGS, None, None
    # No graph: only a direct parent link can be proven.
    if b.parent_version_id == a.dataset_version_id:
        return (
            LineageRelationship.ANCESTOR_TO_DESCENDANT,
            a.dataset_version_id,
            b.dataset_version_id,
        )
    if a.parent_version_id == b.dataset_version_id:
        return (
            LineageRelationship.DESCENDANT_TO_ANCESTOR,
            b.dataset_version_id,
            a.dataset_version_id,
        )
    return LineageRelationship.UNKNOWN_SAME_FAMILY, None, None


def diff_versions(
    from_version: DatasetVersion,
    to_version: DatasetVersion,
    *,
    graph: LineageGraph | None = None,
) -> VersionDiff:
    """Compare two registered dataset versions. Same family only."""
    if from_version.dataset_id != to_version.dataset_id:
        raise VersionDiffError(
            f"cannot compare versions from different dataset families: "
            f"{from_version.dataset_id!r} vs {to_version.dataset_id!r}"
        )

    relationship, ancestor, descendant = _relationship(from_version, to_version, graph)
    return VersionDiff(
        dataset_id=from_version.dataset_id,
        from_version_id=from_version.dataset_version_id,
        to_version_id=to_version.dataset_version_id,
        lineage_relationship=relationship,
        ancestor_version_id=ancestor,
        descendant_version_id=descendant,
        metadata=_metadata_diff(from_version, to_version),
        schema_diff=_schema_diff(from_version, to_version),
        quality=_quality_diff(from_version, to_version),
        content=_content_diff(from_version, to_version),
    )


def diff_registered_versions(
    store_or_graph: LineageGraph,
    from_version_id: str,
    to_version_id: str,
) -> VersionDiff:
    """Convenience: resolve both ids through a :class:`LineageGraph` and diff."""
    return diff_versions(
        store_or_graph.get(from_version_id),
        store_or_graph.get(to_version_id),
        graph=store_or_graph,
    )
