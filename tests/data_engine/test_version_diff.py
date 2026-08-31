"""Phase 3 — deterministic cross-version diffing."""

from __future__ import annotations

import pytest

from data_engine.validation import (
    LineageGraph,
    VersionDiff,
    VersionDiffError,
    diff_registered_versions,
    diff_versions,
)
from data_engine.validation.version_diff import LineageRelationship
from data_engine.validation.version_models import DatasetVersionKind, QualitySnapshot


def _v(make_version, **kw):
    return make_version(**kw)


def test_identical_version_comparison_is_deterministic(make_version):
    v = _v(
        make_version,
        dataset_id="ds",
        version_id="ds:raw",
        kind=DatasetVersionKind.RAW,
        parent_version_id=None,
    )
    d1 = diff_versions(v, v).model_dump()
    d2 = diff_versions(v, v).model_dump()
    assert d1 == d2
    assert diff_versions(v, v).lineage_relationship is LineageRelationship.SAME_VERSION


def _raw_and_child(make_version, **child_kw):
    raw = _v(
        make_version,
        dataset_id="ds",
        version_id="ds:raw",
        kind=DatasetVersionKind.RAW,
        parent_version_id=None,
        version_number=0,
        columns=[("a", "object"), ("b", "int64")],
        row_count=5,
    )
    defaults: dict = {
        "dataset_id": "ds",
        "version_id": "ds:exec-A",
        "kind": DatasetVersionKind.PROCESSED,
        "parent_version_id": "ds:raw",
        "execution_id": "A",
        "version_number": 1,
        "columns": [("a", "object"), ("b", "int64")],
        "row_count": 5,
    }
    defaults.update(child_kw)
    return raw, _v(make_version, **defaults)


def test_row_count_change_is_detected(make_version):
    raw, child = _raw_and_child(make_version, row_count=3)
    diff = diff_versions(raw, child)
    changes = {c.field: (c.before, c.after) for c in diff.metadata.changes}
    assert changes["row_count"] == (5, 3)


def test_added_column_is_detected(make_version):
    raw, child = _raw_and_child(
        make_version, columns=[("a", "object"), ("b", "int64"), ("c", "float64")]
    )
    assert diff_versions(raw, child).schema_diff.added_columns == ["c"]


def test_removed_column_is_detected(make_version):
    raw, child = _raw_and_child(make_version, columns=[("a", "object")])
    assert diff_versions(raw, child).schema_diff.removed_columns == ["b"]


def test_dtype_change_is_detected(make_version):
    raw, child = _raw_and_child(make_version, columns=[("a", "object"), ("b", "float64")])
    changes = diff_versions(raw, child).schema_diff.dtype_changes
    assert [(c.column, c.before, c.after) for c in changes] == [("b", "int64", "float64")]


def test_column_order_change_is_detected(make_version):
    raw, child = _raw_and_child(make_version, columns=[("b", "int64"), ("a", "object")])
    assert diff_versions(raw, child).schema_diff.column_order_changed is True


def test_quality_score_and_finding_changes_are_detected(make_version):
    raw, child = _raw_and_child(make_version)
    raw.quality = QualitySnapshot(
        score=46.0,
        total_findings=3,
        has_critical=False,
        findings_by_type={"missing_values": 1, "duplicate_rows": 1},
        missing_cells=3,
    )
    child.quality = QualitySnapshot(
        score=88.0,
        total_findings=1,
        has_critical=False,
        findings_by_type={"potential_outliers": 1},
        missing_cells=0,
    )
    q = diff_versions(raw, child).quality
    assert q.available
    assert q.score_before == 46.0 and q.score_after == 88.0
    assert q.total_findings_before == 3 and q.total_findings_after == 1
    assert q.missing_cells_before == 3 and q.missing_cells_after == 0
    assert "quality_score: 46.0 -> 88.0" in q.improvements
    assert any("missing_values" in i for i in q.improvements)
    assert any("potential_outliers" in r for r in q.regressions)


def test_unrelated_dataset_families_are_rejected(make_version):
    a = _v(
        make_version,
        dataset_id="ds-A",
        version_id="ds-A:raw",
        kind=DatasetVersionKind.RAW,
        parent_version_id=None,
    )
    b = _v(
        make_version,
        dataset_id="ds-B",
        version_id="ds-B:raw",
        kind=DatasetVersionKind.RAW,
        parent_version_id=None,
    )
    with pytest.raises(VersionDiffError, match="different dataset families"):
        diff_versions(a, b)


def test_missing_files_are_reported_explicitly(make_version):
    raw, child = _raw_and_child(make_version)
    content = diff_versions(raw, child).content
    assert content.available is False
    assert content.unavailable_reason is not None
    assert content.identical_content is None  # never silently "identical"


def test_ancestor_descendant_relationship_is_reported(make_version):
    raw, child = _raw_and_child(make_version)
    diff = diff_versions(raw, child)
    assert diff.lineage_relationship is LineageRelationship.ANCESTOR_TO_DESCENDANT
    assert diff.ancestor_version_id == "ds:raw"
    assert diff.descendant_version_id == "ds:exec-A"
    # reverse direction
    rev = diff_versions(child, raw)
    assert rev.lineage_relationship is LineageRelationship.DESCENDANT_TO_ANCESTOR


def test_diff_json_round_trip(make_version):
    raw, child = _raw_and_child(make_version)
    diff = diff_versions(raw, child)
    restored = VersionDiff.model_validate_json(diff.model_dump_json())
    assert restored == diff


def test_content_diff_uses_real_files(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    graph = LineageGraph.from_store(p.version_store, p.reference.dataset_id)
    diff = diff_registered_versions(graph, raw_v.dataset_version_id, proc_v.dataset_version_id)
    assert diff.content.available is True
    assert diff.content.row_count_before > diff.content.row_count_after  # duplicate removed
    assert diff.content.missing_cells_before >= diff.content.missing_cells_after
    assert diff.lineage_relationship is LineageRelationship.ANCESTOR_TO_DESCENDANT
