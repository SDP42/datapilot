"""Phase 3 — optional opt-in auto-registration around execute_cleaning."""

from __future__ import annotations

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
    AutoRegistrationError,
    DatasetVersionStore,
    RegisteredCleaningResult,
    execute_and_register_cleaning,
    validate_lineage,
)


def _prepare(tmp_path):
    src = tmp_path / "c.csv"
    src.write_text(
        "age,city\n34,London\n34,London\n,Paris\n41,Berlin\n50,London\n29,Paris\n", encoding="utf-8"
    )
    ref = ingest_dataset(src, raw_store=RawDataStore(tmp_path / "raw"))
    df = pd.read_csv(ref.raw_path)
    plan = plan_from_dataframe(df, dataset_id=ref.dataset_id)
    approved = [
        o.operation_id
        for o in plan.operations
        if o.operation_type
        in (OperationType.REMOVE_EXACT_DUPLICATE_ROWS, OperationType.IMPUTE_MISSING_NUMERIC)
    ]
    return ref, df, plan, approved


def test_execute_cleaning_default_behaviour_is_unchanged(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    report = execute_cleaning(
        ref,
        plan,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    # nothing auto-registered
    store = DatasetVersionStore(tmp_path / "versions")
    assert store.list_versions(ref.dataset_id) == []
    assert report.output_dataset_reference is not None


def test_opt_in_registration_creates_processed_version(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")
    result = execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    assert isinstance(result, RegisteredCleaningResult)
    assert result.processed_version.kind.value == "processed"
    assert {v.dataset_version_id for v in store.list_versions(ref.dataset_id)} == {
        result.raw_version.dataset_version_id,
        result.processed_version.dataset_version_id,
    }


def test_registered_version_has_correct_parent_execution_and_hash(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")
    result = execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    proc = result.processed_version
    assert proc.parent_version_id == result.raw_version.dataset_version_id
    assert proc.execution_id == result.execution_report.execution_id
    assert proc.plan_fingerprint == result.execution_report.plan_fingerprint
    assert proc.sha256 == result.execution_report.output_dataset_reference.sha256
    assert proc.dataset_version_id == result.execution_report.output_dataset_reference.dataset_id


def test_auto_registration_passes_validate_lineage(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")
    result = execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    assert result.lineage_validation.valid
    revalidated = validate_lineage(
        result.execution_report,
        raw_reference=ref,
        parent_version=result.raw_version,
        child_version=result.processed_version,
    )
    assert revalidated.valid


def test_registration_failure_is_surfaced(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")

    # Pre-register a CONFLICTING raw record (different sha) at the raw identity.
    from data_engine.validation import DatasetVersion
    from data_engine.validation.version_models import DatasetVersionKind, SchemaSnapshot

    bogus_raw = DatasetVersion(
        dataset_version_id=DatasetVersion.raw_version_id(ref.dataset_id),
        dataset_id=ref.dataset_id,
        version_number=0,
        parent_version_id=None,
        kind=DatasetVersionKind.RAW,
        raw_dataset_id=ref.dataset_id,
        created_at=ref.created_at,
        created_by="test",
        path="/nonexistent/raw.csv",
        size_bytes=1,
        sha256="a" * 64,
        row_count=1,
        column_count=1,
        schema_snapshot=SchemaSnapshot(
            column_order=["x"], columns=[{"name": "x", "dtype": "int64"}]
        ),
    )
    store.register(bogus_raw)

    with pytest.raises(AutoRegistrationError):
        execute_and_register_cleaning(
            ref,
            plan,
            version_store=store,
            approved_operation_ids=approved,
            context=ExecutionContext(allow_full_data_fit=True),
            processed_store=ProcessedDataStore(tmp_path / "processed"),
        )


def test_raw_dataset_remains_byte_for_byte_unchanged(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    raw_bytes = ref.raw_path.read_bytes()
    raw_mode = ref.raw_path.stat().st_mode
    store = DatasetVersionStore(tmp_path / "versions")
    execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    assert ref.raw_path.read_bytes() == raw_bytes
    assert ref.raw_path.stat().st_mode == raw_mode


def test_opt_in_registration_is_idempotent(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")
    kwargs = {
        "version_store": store,
        "approved_operation_ids": approved,
        "context": ExecutionContext(allow_full_data_fit=True),
        "processed_store": ProcessedDataStore(tmp_path / "processed"),
    }
    first = execute_and_register_cleaning(ref, plan, **kwargs)
    second = execute_and_register_cleaning(ref, plan, **kwargs)
    assert first.processed_version.dataset_version_id == second.processed_version.dataset_version_id
    assert len(store.list_versions(ref.dataset_id)) == 2


def test_result_json_round_trips(tmp_path):
    ref, _df, plan, approved = _prepare(tmp_path)
    store = DatasetVersionStore(tmp_path / "versions")
    result = execute_and_register_cleaning(
        ref,
        plan,
        version_store=store,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    restored = RegisteredCleaningResult.model_validate_json(result.model_dump_json())
    assert restored.processed_version == result.processed_version
    assert restored.execution_report.execution_id == result.execution_report.execution_id
