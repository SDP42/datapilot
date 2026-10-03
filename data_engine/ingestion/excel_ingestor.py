"""Excel (.xlsx)-specific ingestion logic — Phase 14.10.

Mirrors `csv_ingestor.py` exactly: validate, fail fast on an unparseable
file, preserve an immutable raw copy, build the `DatasetReference`
handoff object. It does **not** clean, normalise, or transform anything.

Only the **first sheet** of a workbook is ingested — `DatasetReference`
models one dataset as one rectangular table, the same assumption every
other ingestion path in this codebase already makes; a multi-sheet
workbook has no single well-defined "the dataset" otherwise. Legacy
`.xls` is deliberately not accepted — see `pyproject.toml`'s own
reasoning on the `openpyxl` dependency for why.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
import zipfile
from pathlib import Path

import pandas as pd

from datapilot.contracts import DatasetFormat, DatasetReference

from .errors import (
    InvalidExcelError,
    SourceFileNotFoundError,
    SourceFileNotReadableError,
    UnsupportedFormatError,
)
from .raw_store import RawDataStore, sha256_of_file

SUPPORTED_SUFFIXES = {".xlsx"}


def _new_dataset_id() -> str:
    return f"ds-{uuid.uuid4().hex}"


def _validate_source(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise SourceFileNotFoundError(f"No such file: {path}")
    if not os.access(path, os.R_OK):
        raise SourceFileNotReadableError(f"File is not readable: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise UnsupportedFormatError(
            f"Unsupported extension {path.suffix!r}; ingestion currently accepts: "
            f"{sorted(SUPPORTED_SUFFIXES)}"
        )


def _assert_parseable_excel(path: Path) -> None:
    try:
        frame = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        raise InvalidExcelError(f"File is not a valid .xlsx workbook: {path} ({exc})") from exc
    if frame.empty:
        raise InvalidExcelError(f"The first sheet of {path} has no rows")


def ingest_excel(source: Path | str, *, raw_store: RawDataStore | None = None) -> DatasetReference:
    """Ingest a single .xlsx workbook's first sheet and return a `DatasetReference`.

    The raw file is copied into `raw_store` (defaulting to `data/raw/`)
    as a read-only copy, exactly as `ingest_csv` does — the original
    file passed by the caller is left completely untouched.
    """
    path = Path(source).expanduser().resolve()
    _validate_source(path)
    _assert_parseable_excel(path)

    store = raw_store or RawDataStore.default()
    dataset_id = _new_dataset_id()

    stored_path = store.store(path, dataset_id=dataset_id, original_filename=path.name)

    reference = DatasetReference(
        dataset_id=dataset_id,
        original_filename=path.name,
        source_format=DatasetFormat.XLSX,
        raw_path=stored_path,
        size_bytes=stored_path.stat().st_size,
        sha256=sha256_of_file(stored_path),
        created_at=dt.datetime.now(dt.UTC),
    )
    store.write_reference_sidecar(dataset_id, reference.model_dump_json(indent=2))
    return reference
