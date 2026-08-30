"""Ingestion: success path, raw preservation, metadata, failure cases."""

from __future__ import annotations

import os

import pytest

from data_engine.ingestion import (
    InvalidCSVError,
    RawDataStore,
    SourceFileNotFoundError,
    UnsupportedFormatError,
    ingest_dataset,
)
from datapilot.contracts import DatasetFormat, DatasetReference


def test_successful_csv_ingestion(sample_csv_path, raw_store):
    ref = ingest_dataset(sample_csv_path, raw_store=raw_store)

    assert isinstance(ref, DatasetReference)
    assert ref.source_format is DatasetFormat.CSV
    assert ref.dataset_id.startswith("ds-")
    assert ref.original_filename == "customers.csv"
    assert ref.raw_path.exists()
    assert ref.size_bytes == ref.raw_path.stat().st_size
    assert len(ref.sha256) == 64


def test_raw_copy_is_preserved_byte_for_byte(sample_csv_path, raw_store):
    original_bytes = sample_csv_path.read_bytes()
    ref = ingest_dataset(sample_csv_path, raw_store=raw_store)
    assert ref.raw_path.read_bytes() == original_bytes


def test_raw_copy_is_read_only(sample_csv_path, raw_store):
    ref = ingest_dataset(sample_csv_path, raw_store=raw_store)
    mode = ref.raw_path.stat().st_mode & 0o777
    assert mode & 0o222 == 0  # no write bits for anyone


def test_original_source_file_is_untouched(sample_csv_path, raw_store):
    before = sample_csv_path.read_bytes()
    ingest_dataset(sample_csv_path, raw_store=raw_store)
    assert sample_csv_path.read_bytes() == before
    assert os.access(sample_csv_path, os.W_OK)


def test_reference_sidecar_is_written(sample_csv_path, raw_store):
    ref = ingest_dataset(sample_csv_path, raw_store=raw_store)
    sidecar = raw_store.dataset_dir(ref.dataset_id) / "reference.json"
    assert sidecar.exists()
    assert ref.dataset_id in sidecar.read_text()


def test_unique_id_per_ingestion(sample_csv_path, raw_store):
    a = ingest_dataset(sample_csv_path, raw_store=raw_store)
    b = ingest_dataset(sample_csv_path, raw_store=raw_store)
    assert a.dataset_id != b.dataset_id


def test_missing_file_raises(tmp_path, raw_store):
    with pytest.raises(SourceFileNotFoundError):
        ingest_dataset(tmp_path / "nope.csv", raw_store=raw_store)


def test_unsupported_extension_raises(tmp_path, raw_store):
    xlsx = tmp_path / "data.xlsx"
    xlsx.write_bytes(b"not really excel")
    with pytest.raises(UnsupportedFormatError):
        ingest_dataset(xlsx, raw_store=raw_store)


def test_invalid_csv_raises(tmp_path, raw_store):
    bad = tmp_path / "broken.csv"
    bad.write_text('a,b,c\n1,2\n"unterminated,3,4\n', encoding="utf-8")
    with pytest.raises(InvalidCSVError):
        ingest_dataset(bad, raw_store=raw_store)


def test_empty_csv_raises(tmp_path, raw_store):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(InvalidCSVError):
        ingest_dataset(empty, raw_store=raw_store)


def test_failed_validation_stores_nothing(tmp_path):
    store = RawDataStore(tmp_path / "store")
    missing = tmp_path / "gone.csv"
    with pytest.raises(SourceFileNotFoundError):
        ingest_dataset(missing, raw_store=store)
    assert not store.root.exists() or not any(store.root.iterdir())
