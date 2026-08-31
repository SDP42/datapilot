"""Phase 3 — version integrity verification, family consistency, and the
strengthened version <-> lineage binding check.

Central behaviour under test: corruption / staleness / tampering is
DETECTED and REPORTED, and nothing is ever silently repaired.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from data_engine.cleaning import (
    ExecutionContext,
    OperationType,
    ProcessedDataStore,
    execute_cleaning,
    plan_from_dataframe,
)
from data_engine.ingestion import RawDataStore, ingest_dataset
from data_engine.validation import (
    DatasetVersion,
    DatasetVersionStore,
    VersionIntegrityError,
    check_family_consistency,
    check_version_lineage_binding,
    execute_and_register_cleaning,
    verify_registered_version,
    verify_version_integrity,
)

_CSV = (
    "age,city,score\n"
    "34,London,9.5\n"
    "34,London,9.5\n"  # exact duplicate
    ",Paris,7.1\n"
    "29,London,\n"
    "41,Berlin,8.8\n"
    "52,Paris,6.0\n"
    "37,London,7.7\n"
    "45,Berlin,8.1\n"
)


def _rewrite_record(path: Path, mutate) -> None:
    path.chmod(0o644)
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    path.chmod(0o444)


def _tamper_file(path: Path, append: bytes = b"99,Tampered,1.0\n") -> None:
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + append)
    path.chmod(0o444)


def _truncate_file(path: Path, drop: int = 1) -> None:
    path.chmod(0o644)
    path.write_bytes(path.read_bytes()[:-drop])
    path.chmod(0o444)


@dataclasses.dataclass
class Registered:
    reference: object
    df: pd.DataFrame
    store: DatasetVersionStore
    result: object  # RegisteredCleaningResult


@dataclasses.dataclass
class Family:
    ref: object
    store: DatasetVersionStore
    raw: DatasetVersion
    a: DatasetVersion
    b: DatasetVersion


def _register(tmp_path) -> Registered:
    src = tmp_path / "c.csv"
    src.write_text(_CSV, encoding="utf-8")
    ref = ingest_dataset(src, raw_store=RawDataStore(tmp_path / "raw"))
    df = pd.read_csv(ref.raw_path)
    plan = plan_from_dataframe(df, dataset_id=ref.dataset_id)
    approved = [
        o.operation_id
        for o in plan.operations
        if o.operation_type
        in (OperationType.REMOVE_EXACT_DUPLICATE_ROWS, OperationType.IMPUTE_MISSING_NUMERIC)
    ]
    store = DatasetVersionStore(tmp_path / "versions")
    result = execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    return Registered(ref, df, store, result)


def _build_three_level_family(tmp_path):
    """raw -> exec-A -> exec-B (exec-B registered with parent = exec-A)."""
    src = tmp_path / "c.csv"
    src.write_text(_CSV, encoding="utf-8")
    ref = ingest_dataset(src, raw_store=RawDataStore(tmp_path / "raw"))
    df = pd.read_csv(ref.raw_path)
    plan = plan_from_dataframe(df, dataset_id=ref.dataset_id)
    dedup = next(
        o.operation_id
        for o in plan.operations
        if o.operation_type is OperationType.REMOVE_EXACT_DUPLICATE_ROWS
    )
    impute = [
        o.operation_id
        for o in plan.operations
        if o.operation_type is OperationType.IMPUTE_MISSING_NUMERIC
    ]
    store = DatasetVersionStore(tmp_path / "versions")
    pstore = ProcessedDataStore(tmp_path / "processed")
    ctx = ExecutionContext(allow_full_data_fit=True)

    raw_v = store.register_raw(ref, df)
    report_a = execute_cleaning(
        ref, plan, approved_operation_ids=[dedup], context=ctx, processed_store=pstore
    )
    ver_a = store.register_from_execution(report_a, parent_version_id=raw_v.dataset_version_id)
    report_b = execute_cleaning(
        ref, plan, approved_operation_ids=[dedup, *impute], context=ctx, processed_store=pstore
    )
    ver_b = store.register_from_execution(report_b, parent_version_id=ver_a.dataset_version_id)
    return Family(ref, store, raw_v, ver_a, ver_b)


# ======================================================================
# Task 4 — stale / tampered-file detection
# ======================================================================


def test_detects_modified_processed_csv(tmp_path):
    r = _register(tmp_path)
    pid = r.result.processed_version.dataset_version_id
    _tamper_file(Path(r.result.processed_version.path))
    res = verify_registered_version(r.store, pid)
    assert not res.valid
    assert any("SHA-256 mismatch" in e for e in res.errors)


def test_detects_modified_raw_csv(tmp_path):
    r = _register(tmp_path)
    rid = r.result.raw_version.dataset_version_id
    _tamper_file(Path(r.result.raw_version.path))
    res = verify_registered_version(r.store, rid)
    assert not res.valid
    assert any("SHA-256 mismatch" in e for e in res.errors)


def test_detects_changed_file_size(tmp_path):
    r = _register(tmp_path)
    pid = r.result.processed_version.dataset_version_id
    _truncate_file(Path(r.result.processed_version.path), drop=3)
    res = verify_registered_version(r.store, pid)
    assert not res.valid
    assert any("size mismatch" in e for e in res.errors)


def test_detects_missing_processed_file(tmp_path):
    r = _register(tmp_path)
    p = Path(r.result.processed_version.path)
    p.chmod(0o644)
    p.unlink()
    res = verify_registered_version(r.store, r.result.processed_version.dataset_version_id)
    assert not res.valid
    assert any("missing" in e for e in res.errors)


def test_detects_missing_raw_file(tmp_path):
    r = _register(tmp_path)
    p = Path(r.result.raw_version.path)
    p.chmod(0o644)
    p.unlink()
    res = verify_registered_version(r.store, r.result.raw_version.dataset_version_id)
    assert not res.valid
    assert any("missing" in e for e in res.errors)


def test_detects_corrupted_version_json(tmp_path):
    r = _register(tmp_path)
    pid = r.result.processed_version.dataset_version_id
    record = r.store.version_file_path(pid)
    record.chmod(0o644)
    record.write_text("{ this is not valid json", encoding="utf-8")
    record.chmod(0o444)
    res = verify_registered_version(r.store, pid)
    assert not res.valid
    assert any("corrupted or unparseable" in e for e in res.errors)


def test_detects_mismatched_stored_sha256(tmp_path):
    r = _register(tmp_path)
    pid = r.result.processed_version.dataset_version_id
    record = r.store.version_file_path(pid)
    _rewrite_record(record, lambda d: d.__setitem__("sha256", "f" * 64))
    res = verify_registered_version(r.store, pid)
    assert not res.valid
    assert any("SHA-256 mismatch" in e for e in res.errors)


def test_integrity_check_never_repairs(tmp_path):
    r = _register(tmp_path)
    pid = r.result.processed_version.dataset_version_id
    record = r.store.version_file_path(pid)
    data_file = Path(r.result.processed_version.path)

    _tamper_file(data_file)
    record_before = record.read_bytes()
    data_before = data_file.read_bytes()

    res = verify_registered_version(r.store, pid)
    assert not res.valid
    with pytest.raises(VersionIntegrityError):
        res.raise_for_status()

    assert record.read_bytes() == record_before  # record untouched
    assert data_file.read_bytes() == data_before  # data untouched


def test_verify_version_integrity_accepts_a_healthy_version(tmp_path):
    r = _register(tmp_path)
    assert verify_version_integrity(r.result.raw_version).valid
    assert verify_version_integrity(r.result.processed_version).valid


# ======================================================================
# Task 5 — full-family validation, one corruption at a time
# ======================================================================


def test_valid_three_level_family_validates(tmp_path):
    fam = _build_three_level_family(tmp_path)
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert result.valid, result.errors
    assert result.version_count == 3
    assert set(result.integrity) == {
        fam.raw.dataset_version_id,
        fam.a.dataset_version_id,
        fam.b.dataset_version_id,
    }
    assert all(v.valid for v in result.integrity.values())


def test_broken_parent_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    record = fam.store.version_file_path(fam.b.dataset_version_id)
    _rewrite_record(
        record, lambda d: d.__setitem__("parent_version_id", f"{fam.ref.dataset_id}:exec-ghost")
    )
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("not registered" in e for e in result.errors)


def test_foreign_parent_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    record = fam.store.version_file_path(fam.b.dataset_version_id)
    _rewrite_record(record, lambda d: d.__setitem__("parent_version_id", "ds-OTHER:raw"))
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("foreign family" in e for e in result.errors)


def test_cycle_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    # exec-A <-> exec-B mutual parents, raw still a root
    _rewrite_record(
        fam.store.version_file_path(fam.a.dataset_version_id),
        lambda d: d.__setitem__("parent_version_id", fam.b.dataset_version_id),
    )
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("cycle" in e.lower() for e in result.errors)


def test_wrong_raw_sha_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    _rewrite_record(
        fam.store.version_file_path(fam.raw.dataset_version_id),
        lambda d: d.__setitem__("sha256", "a" * 64),
    )
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("SHA-256 mismatch" in e for e in result.errors)


def test_wrong_processed_sha_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    _rewrite_record(
        fam.store.version_file_path(fam.b.dataset_version_id),
        lambda d: d.__setitem__("sha256", "b" * 64),
    )
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("SHA-256 mismatch" in e for e in result.errors)


def test_invalid_version_metadata_detected(tmp_path):
    fam = _build_three_level_family(tmp_path)
    _rewrite_record(
        fam.store.version_file_path(fam.a.dataset_version_id),
        lambda d: d.__setitem__("column_count", 999),
    )
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("corrupted or unparseable" in e for e in result.errors)


def test_family_check_reports_all_errors_not_just_the_first(tmp_path):
    fam = _build_three_level_family(tmp_path)
    _rewrite_record(
        fam.store.version_file_path(fam.b.dataset_version_id),
        lambda d: d.__setitem__("parent_version_id", "ds-OTHER:raw"),
    )
    _tamper_file(Path(fam.raw.path))
    result = check_family_consistency(fam.store, fam.ref.dataset_id)
    assert not result.valid
    assert any("foreign family" in e for e in result.errors)
    assert any("SHA-256 mismatch" in e for e in result.errors)
    assert len(result.errors) >= 2


# ======================================================================
# Task 3 — version <-> lineage binding
# ======================================================================


def test_version_lineage_binding_valid(tmp_path):
    r = _register(tmp_path)
    result = check_version_lineage_binding(
        r.result.processed_version,
        r.result.execution_report,
        parent_version=r.result.raw_version,
        raw_reference=r.reference,
    )
    assert result.valid, result.errors


def test_binding_detects_inconsistent_execution_id(tmp_path):
    r = _register(tmp_path)
    tampered = r.result.processed_version.model_copy(update={"execution_id": "WRONG"})
    result = check_version_lineage_binding(tampered, r.result.execution_report)
    assert not result.valid
    assert any("execution_id" in e for e in result.errors)


def test_binding_detects_inconsistent_plan_fingerprint(tmp_path):
    r = _register(tmp_path)
    tampered = r.result.processed_version.model_copy(update={"plan_fingerprint": "deadbeef"})
    result = check_version_lineage_binding(tampered, r.result.execution_report)
    assert not result.valid
    assert any("plan_fingerprint" in e for e in result.errors)


def test_binding_detects_wrong_applied_operation_ids(tmp_path):
    r = _register(tmp_path)
    tampered = r.result.processed_version.model_copy(update={"applied_operation_ids": ["bogus:op"]})
    result = check_version_lineage_binding(tampered, r.result.execution_report)
    assert not result.valid
    assert any("applied_operation_ids" in e for e in result.errors)


def test_binding_detects_wrong_lineage_step_count(tmp_path):
    r = _register(tmp_path)
    tampered = r.result.processed_version.model_copy(update={"lineage_step_count": 999})
    result = check_version_lineage_binding(tampered, r.result.execution_report)
    assert not result.valid
    assert any("lineage_step_count" in e for e in result.errors)


def test_binding_detects_wrong_processed_sha(tmp_path):
    r = _register(tmp_path)
    tampered = r.result.processed_version.model_copy(update={"sha256": "c" * 64})
    result = check_version_lineage_binding(tampered, r.result.execution_report)
    assert not result.valid
    assert any("sha256" in e or "SHA" in e for e in result.errors)


def test_integrity_and_binding_results_json_round_trip(tmp_path):
    r = _register(tmp_path)
    integ = verify_registered_version(r.store, r.result.processed_version.dataset_version_id)
    from data_engine.validation import VersionIntegrityResult

    assert VersionIntegrityResult.model_validate_json(integ.model_dump_json()) == integ

    fam = check_family_consistency(r.store, r.reference.dataset_id)
    from data_engine.validation import FamilyConsistencyResult

    assert FamilyConsistencyResult.model_validate_json(fam.model_dump_json()) == fam


def test_datasetversion_still_registers_normally(tmp_path):
    """Backward-compatibility smoke: the existing register path is unchanged."""
    r = _register(tmp_path)
    fetched = r.store.get(r.result.processed_version.dataset_version_id)
    assert isinstance(fetched, DatasetVersion)
    assert fetched == r.result.processed_version
