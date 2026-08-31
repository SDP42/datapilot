"""Shared fixtures for data_engine tests.

All datasets are tiny and created on the fly inside a temp directory —
nothing is written to the repo's ``data/raw/``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import textwrap
from collections.abc import Callable
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
from data_engine.validation import DatasetVersion, DatasetVersionStore
from data_engine.validation.version_models import DatasetVersionKind, SchemaSnapshot

# A small, deliberately messy CSV: missing values, a duplicate row,
# numeric / categorical / date-like columns.
SAMPLE_CSV = textwrap.dedent(
    """\
    id,age,city,signup_date,score
    1,34,London,2021-01-05,9.5
    2,,Paris,2021-02-11,7.1
    3,29,London,2021-03-01,
    4,41,Berlin,2021-03-01,8.8
    4,41,Berlin,2021-03-01,8.8
    5,52,,2021-04-20,6.0
    """
)


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "customers.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


@pytest.fixture
def raw_store(tmp_path: Path) -> RawDataStore:
    return RawDataStore(tmp_path / "raw_store")


# --- Phase 3 lineage-graph / auto-register / diff helpers -------------

_LINEAGE_CSV = (
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


@dataclasses.dataclass
class LineagePipeline:
    reference: object
    df: pd.DataFrame
    report: object
    cleaned: pd.DataFrame
    version_store: DatasetVersionStore


@pytest.fixture
def lineage_pipeline(tmp_path: Path) -> LineagePipeline:
    src = tmp_path / "customers.csv"
    src.write_text(_LINEAGE_CSV, encoding="utf-8")
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
    return LineagePipeline(ref, df, report, cleaned, DatasetVersionStore(tmp_path / "versions"))


@pytest.fixture
def make_version() -> Callable[..., DatasetVersion]:
    def _make(
        dataset_id: str,
        version_id: str,
        *,
        kind: DatasetVersionKind,
        parent_version_id: str | None,
        execution_id: str | None = None,
        row_count: int = 3,
        columns: list[tuple[str, str]] | None = None,
        sha256: str = "0" * 64,
        version_number: int = 0,
    ) -> DatasetVersion:
        cols = columns or [("a", "int64"), ("b", "object")]
        return DatasetVersion(
            dataset_version_id=version_id,
            dataset_id=dataset_id,
            version_number=version_number,
            parent_version_id=parent_version_id,
            kind=kind,
            raw_dataset_id=dataset_id,
            created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            created_by="test",
            path=f"/nonexistent/{version_id.replace(':', '_')}.csv",
            size_bytes=1,
            sha256=sha256,
            row_count=row_count,
            column_count=len(cols),
            schema_snapshot=SchemaSnapshot(
                column_order=[c for c, _ in cols],
                columns=[{"name": c, "dtype": d} for c, d in cols],
            ),
            execution_id=execution_id,
        )

    return _make
