"""Phase 3 — DatasetVersion model, version store, and lineage validation."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest
from pydantic import ValidationError

from data_engine.cleaning import (
    ExecutionContext,
    OperationType,
    ProcessedDataStore,
    execute_cleaning,
    plan_from_dataframe,
)
from data_engine.cleaning.execution_models import ExecutionStatus
from data_engine.ingestion import RawDataStore, ingest_dataset
from data_engine.ingestion.raw_store import sha256_of_file
from data_engine.validation import (
    ConflictingVersionError,
    DatasetVersion,
    DatasetVersionKind,
    DatasetVersionStore,
    DuplicateVersionError,
    LineageValidationError,
    SchemaSnapshot,
    VersionIntegrityError,
    assert_roundtrip_consistent,
    validate_lineage,
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
    "500,Berlin,8.0\n"  # outlier-ish
)


@dataclasses.dataclass
class Pipeline:
    reference: object
    df: pd.DataFrame
    report: object
    cleaned: pd.DataFrame
    version_store: DatasetVersionStore


def _pipeline(tmp_path, *, register: bool = True) -> Pipeline:
    src = tmp_path / "customers.csv"
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
    report = execute_cleaning(
        ref,
        plan,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    cleaned = pd.read_csv(report.output_dataset_reference.path)
    store = DatasetVersionStore(tmp_path / "versions")
    return Pipeline(ref, df, report, cleaned, store)


# --- 1. DatasetVersion JSON serialisation & round-trip ------------------


def test_dataset_version_json_roundtrip(tmp_path):
    p = _pipeline(tmp_path)
    raw_v = DatasetVersion.from_raw(p.reference, p.df)
    proc_v = DatasetVersion.from_execution_report(
        p.report,
        parent_version_id=raw_v.dataset_version_id,
        schema_source=p.cleaned,
        version_number=1,
    )
    for version in (raw_v, proc_v):
        restored = DatasetVersion.model_validate_json(version.model_dump_json())
        assert restored == version
        assert restored.model_dump(mode="json") == version.model_dump(mode="json")


# --- 2. rejects invalid required metadata ------------------------------


def _valid_kwargs() -> dict:
    return {
        "dataset_version_id": "ds-x:raw",
        "dataset_id": "ds-x",
        "version_number": 0,
        "kind": DatasetVersionKind.RAW,
        "raw_dataset_id": "ds-x",
        "created_at": "2026-01-01T00:00:00Z",
        "created_by": "test",
        "path": "/tmp/x.csv",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "row_count": 1,
        "column_count": 1,
        "schema_snapshot": SchemaSnapshot(
            column_order=["a"], columns=[{"name": "a", "dtype": "int64"}]
        ),
    }


def test_dataset_version_rejects_invalid_metadata():
    # id not prefixed with dataset_id
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate({**_valid_kwargs(), "dataset_version_id": "other:raw"})
    # raw version with a parent
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate({**_valid_kwargs(), "parent_version_id": "ds-x:raw"})
    # processed version without execution_id
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate(
            {
                **_valid_kwargs(),
                "dataset_version_id": "ds-x:exec-abc",
                "kind": DatasetVersionKind.PROCESSED,
                "parent_version_id": "ds-x:raw",
            }
        )
    # column_count disagrees with the schema snapshot
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate({**_valid_kwargs(), "column_count": 5})
    # negative row_count
    with pytest.raises(ValidationError):
        DatasetVersion.model_validate({**_valid_kwargs(), "row_count": -1})


# --- 3. registering a version succeeds ---------------------------------


def test_registering_a_version_succeeds(tmp_path):
    p = _pipeline(tmp_path)
    version = p.version_store.register_raw(p.reference, p.df)
    assert version.kind is DatasetVersionKind.RAW
    assert version.version_number == 0
    stored = p.version_store.dataset_dir(p.reference.dataset_id) / "raw.json"
    assert stored.exists()
    assert stored.stat().st_mode & 0o222 == 0  # read-only record


# --- 4. duplicate registration is rejected ----------------------------


def test_duplicate_version_registration_is_rejected(tmp_path):
    p = _pipeline(tmp_path)
    p.version_store.register_raw(p.reference, p.df)
    with pytest.raises(DuplicateVersionError):
        p.version_store.register_raw(p.reference, p.df)


# --- 5. retrieving a registered version works ------------------------


def test_retrieving_a_registered_version_works(tmp_path):
    p = _pipeline(tmp_path)
    registered = p.version_store.register_raw(p.reference, p.df)
    fetched = p.version_store.get(registered.dataset_version_id)
    assert fetched == registered


# --- 6. listing preserves parent/child relationships ---------------


def test_listing_versions_preserves_parent_child(tmp_path):
    p = _pipeline(tmp_path)
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    versions = p.version_store.list_versions(p.reference.dataset_id)
    assert [v.version_number for v in versions] == [0, 1]
    assert versions[0].dataset_version_id == raw_v.dataset_version_id
    assert versions[1].parent_version_id == raw_v.dataset_version_id
    assert [c.dataset_version_id for c in p.version_store.children(raw_v.dataset_version_id)] == [
        proc_v.dataset_version_id
    ]


# --- 7. raw dataset remains immutable ------------------------------


def test_raw_dataset_remains_immutable(tmp_path):
    p = _pipeline(tmp_path)
    raw_bytes = p.reference.raw_path.read_bytes()
    raw_mode = p.reference.raw_path.stat().st_mode
    raw_v = p.version_store.register_raw(p.reference, p.df)
    p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    assert p.reference.raw_path.read_bytes() == raw_bytes
    assert p.reference.raw_path.stat().st_mode == raw_mode


# --- 8. processed SHA256 matches the stored file ------------------


def test_processed_sha256_matches_stored_file(tmp_path):
    p = _pipeline(tmp_path)
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    assert proc_v.sha256 == sha256_of_file(proc_v.path)

    tampered = proc_v.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(VersionIntegrityError):
        DatasetVersionStore(tmp_path / "versions2")._validate_integrity(tampered)


# --- 9. inconsistent lineage is rejected --------------------------


def test_inconsistent_lineage_is_rejected(tmp_path):
    p = _pipeline(tmp_path)
    assert validate_lineage(p.report, raw_reference=p.reference).valid

    broken = p.report.model_copy(deep=True)
    broken.lineage.raw_dataset_id = "ds-unrelated"
    result = validate_lineage(broken, raw_reference=p.reference)
    assert not result.valid
    assert any("raw identity mismatch" in e for e in result.errors)
    with pytest.raises(LineageValidationError):
        result.raise_for_status()


# --- 10. failed/skipped ops cannot be successful lineage steps -----


def test_failed_ops_cannot_appear_as_successful_lineage_steps(tmp_path):
    p = _pipeline(tmp_path)
    broken = p.report.model_copy(deep=True)
    idx = next(
        i for i, s in enumerate(broken.lineage.steps) if s.status is not ExecutionStatus.SUCCESS
    )
    broken.lineage.steps[idx].status = ExecutionStatus.SUCCESS
    result = validate_lineage(broken)
    assert not result.valid
    assert any(
        "claims SUCCESS" in e or "disagrees with the record status" in e for e in result.errors
    )


def test_operations_succeeded_count_must_match_records(tmp_path):
    p = _pipeline(tmp_path)
    broken = p.report.model_copy(deep=True)
    broken.operations_succeeded += 5
    result = validate_lineage(broken)
    assert not result.valid
    assert any("operations_succeeded" in e for e in result.errors)


# --- 11. plan fingerprint mismatch is detected -------------------


def test_plan_fingerprint_mismatch_is_detected(tmp_path):
    p = _pipeline(tmp_path)
    broken = p.report.model_copy(deep=True)
    broken.plan_fingerprint = "deadbeefdeadbeef"
    result = validate_lineage(broken)
    assert not result.valid
    assert any("plan fingerprint mismatch" in e for e in result.errors)


# --- 12. lineage survives JSON round-trip -----------------------


def test_lineage_survives_json_roundtrip(tmp_path):
    p = _pipeline(tmp_path)
    result = assert_roundtrip_consistent(p.report, raw_reference=p.reference)
    assert result.valid

    broken = p.report.model_copy(deep=True)
    broken.plan_fingerprint = "0000"
    broken_result = assert_roundtrip_consistent(broken)
    assert not broken_result.valid


# --- 13. a child cannot reference an unrelated parent ----------


def test_child_version_cannot_reference_unrelated_parent(tmp_path):
    p = _pipeline(tmp_path)

    # a foreign dataset family, registered in the same store
    other_src = tmp_path / "other.csv"
    other_src.write_text("x\n1\n2\n", encoding="utf-8")
    other_ref = ingest_dataset(other_src, raw_store=RawDataStore(tmp_path / "raw"))
    other_df = pd.read_csv(other_ref.raw_path)
    foreign_raw = p.version_store.register_raw(other_ref, other_df)

    # processed version of p's dataset, but pointing at the foreign raw as parent
    bad_child = DatasetVersion.from_execution_report(
        p.report,
        parent_version_id=foreign_raw.dataset_version_id,
        schema_source=p.cleaned,
        version_number=1,
    )
    with pytest.raises(ConflictingVersionError):
        p.version_store.register(bad_child)

    # and an unregistered parent is also rejected
    orphan = DatasetVersion.from_execution_report(
        p.report,
        parent_version_id=f"{p.reference.dataset_id}:raw",
        schema_source=p.cleaned,
        version_number=1,
    )
    with pytest.raises(ConflictingVersionError):
        p.version_store.register(orphan)  # parent not registered yet


def test_lineage_validation_full_happy_path(tmp_path):
    p = _pipeline(tmp_path)
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    result = validate_lineage(
        p.report,
        raw_reference=p.reference,
        parent_version=raw_v,
        child_version=proc_v,
    )
    assert result.valid, result.errors
    assert result.checks_run == 7
