"""The end-to-end data-engine contract:

raw file -> ingest_dataset -> DatasetReference -> profile_dataset -> DatasetProfile
"""

from __future__ import annotations

from data_engine.ingestion import ingest_dataset
from data_engine.profiling import load_dataframe, profile_dataset


def test_ingest_then_profile(sample_csv_path, raw_store):
    reference = ingest_dataset(sample_csv_path, raw_store=raw_store)
    profile = profile_dataset(reference)

    # The profile is tied back to the same dataset.
    assert profile.dataset_id == reference.dataset_id
    assert profile.n_rows == 6
    assert profile.n_columns == 5
    assert profile.duplicate_row_count == 1
    assert "age" in profile.numeric_columns
    assert "signup_date" in profile.datetime_columns


def test_profiling_leaves_raw_copy_read_only_and_unchanged(sample_csv_path, raw_store):
    reference = ingest_dataset(sample_csv_path, raw_store=raw_store)
    digest_before = reference.raw_path.read_bytes()

    load_dataframe(reference)
    profile_dataset(reference)

    assert reference.raw_path.read_bytes() == digest_before
    assert reference.raw_path.stat().st_mode & 0o222 == 0
