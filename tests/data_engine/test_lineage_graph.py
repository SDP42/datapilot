"""Phase 3 — the lineage graph / DAG navigation layer."""

from __future__ import annotations

import pytest

from data_engine.validation import LineageGraph, LineageGraphError
from data_engine.validation.version_models import DatasetVersionKind


def _chain(make_version):
    """raw -> exec-A -> exec-B ; raw -> exec-C"""
    ds = "ds-fam"
    raw = make_version(
        ds, f"{ds}:raw", kind=DatasetVersionKind.RAW, parent_version_id=None, version_number=0
    )
    a = make_version(
        ds,
        f"{ds}:exec-A",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=raw.dataset_version_id,
        execution_id="A",
        version_number=1,
    )
    b = make_version(
        ds,
        f"{ds}:exec-B",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=a.dataset_version_id,
        execution_id="B",
        version_number=2,
    )
    c = make_version(
        ds,
        f"{ds}:exec-C",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=raw.dataset_version_id,
        execution_id="C",
        version_number=3,
    )
    return raw, a, b, c


def test_raw_version_resolves_as_root(make_version):
    raw, a, b, c = _chain(make_version)
    g = LineageGraph([b, c, a, raw])  # unordered input
    assert g.root(b.dataset_version_id).dataset_version_id == raw.dataset_version_id
    assert g.root(c.dataset_version_id).dataset_version_id == raw.dataset_version_id


def test_direct_parent_lookup(make_version):
    raw, a, b, c = _chain(make_version)
    g = LineageGraph([raw, a, b, c])
    assert g.parent(b.dataset_version_id).dataset_version_id == a.dataset_version_id
    assert g.parent(raw.dataset_version_id) is None


def test_direct_children_lookup(make_version):
    raw, a, b, c = _chain(make_version)
    g = LineageGraph([raw, a, b, c])
    assert [v.dataset_version_id for v in g.children(raw.dataset_version_id)] == [
        a.dataset_version_id,
        c.dataset_version_id,
    ]
    assert g.children(b.dataset_version_id) == []


def test_ancestor_traversal(make_version):
    raw, a, b, c = _chain(make_version)
    g = LineageGraph([raw, a, b, c])
    assert [v.dataset_version_id for v in g.ancestors(b.dataset_version_id)] == [
        a.dataset_version_id,
        raw.dataset_version_id,
    ]


def test_descendant_traversal(make_version):
    raw, a, b, c = _chain(make_version)
    g = LineageGraph([raw, a, b, c])
    assert [v.dataset_version_id for v in g.descendants(raw.dataset_version_id)] == [
        a.dataset_version_id,
        c.dataset_version_id,
        b.dataset_version_id,
    ]


def test_raw_to_processed_path_is_deterministic(make_version):
    raw, a, b, c = _chain(make_version)
    g1 = LineageGraph([raw, a, b, c])
    g2 = LineageGraph([c, b, a, raw])
    path1 = [v.dataset_version_id for v in g1.path_to(b.dataset_version_id)]
    path2 = [v.dataset_version_id for v in g2.path_to(b.dataset_version_id)]
    assert path1 == path2 == [raw.dataset_version_id, a.dataset_version_id, b.dataset_version_id]


def test_unrelated_families_are_rejected(make_version):
    raw_a = make_version("ds-A", "ds-A:raw", kind=DatasetVersionKind.RAW, parent_version_id=None)
    raw_b = make_version("ds-B", "ds-B:raw", kind=DatasetVersionKind.RAW, parent_version_id=None)
    with pytest.raises(LineageGraphError, match="single dataset family"):
        LineageGraph([raw_a, raw_b])
    g = LineageGraph([raw_a])
    assert g.same_family("ds-A:raw", "ds-B:raw") is False


def test_missing_parent_is_detected(make_version):
    orphan = make_version(
        "ds-x",
        "ds-x:exec-Z",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id="ds-x:raw",
        execution_id="Z",
    )
    with pytest.raises(LineageGraphError, match="not in the lineage"):
        LineageGraph([orphan])


def test_self_parent_is_detected(make_version):
    v = make_version(
        "ds-x",
        "ds-x:exec-S",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id="ds-x:exec-S",
        execution_id="S",
    )
    with pytest.raises(LineageGraphError, match="its own parent"):
        LineageGraph([v])


def test_lineage_cycle_is_detected(make_version):
    ds = "ds-cyc"
    raw = make_version(ds, f"{ds}:raw", kind=DatasetVersionKind.RAW, parent_version_id=None)
    p = make_version(
        ds,
        f"{ds}:exec-P",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=f"{ds}:exec-Q",
        execution_id="P",
    )
    q = make_version(
        ds,
        f"{ds}:exec-Q",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=f"{ds}:exec-P",
        execution_id="Q",
    )
    with pytest.raises(LineageGraphError):
        LineageGraph([raw, p, q])


def test_traversal_cannot_loop_forever(make_version):
    ds = "ds-cyc2"
    p = make_version(
        ds,
        f"{ds}:exec-P",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=f"{ds}:exec-Q",
        execution_id="P",
    )
    q = make_version(
        ds,
        f"{ds}:exec-Q",
        kind=DatasetVersionKind.PROCESSED,
        parent_version_id=f"{ds}:exec-P",
        execution_id="Q",
    )
    g = LineageGraph([p, q], _validate=False)  # bypass construction check
    with pytest.raises(LineageGraphError, match="cycle"):
        g.ancestors(p.dataset_version_id)
    with pytest.raises(LineageGraphError, match="cycle"):
        g.descendants(p.dataset_version_id)


def test_from_store_and_real_pipeline(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    g = LineageGraph.from_store(p.version_store, p.reference.dataset_id)
    assert g.root(proc_v.dataset_version_id).dataset_version_id == raw_v.dataset_version_id
    assert g.same_family(raw_v.dataset_version_id, proc_v.dataset_version_id) is True
