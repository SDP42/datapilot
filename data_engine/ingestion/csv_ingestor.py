"""CSV-specific ingestion logic.

Responsibilities:

1. Validate the source file (exists, readable, right extension, parseable).
2. Hand the file to the :class:`RawDataStore` for immutable preservation.
3. Build the :class:`DatasetReference` handoff object.

It does **not** clean, normalise, or transform anything. The only reason
this module reads the CSV at all is to *fail fast* on a file that is not
valid CSV — the parsed frame is discarded.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from pathlib import Path

import pandas as pd

from datapilot.contracts import DatasetFormat, DatasetReference

from .errors import (
    InvalidCSVError,
    SourceFileNotFoundError,
    SourceFileNotReadableError,
    UnsupportedFormatError,
)
from .raw_store import RawDataStore, sha256_of_file

SUPPORTED_SUFFIXES = {".csv"}


def _new_dataset_id() -> str:
    """Opaque, collision-safe identifier for one ingested dataset."""
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


def _assert_parseable_csv(path: Path) -> None:
    try:
        # nrows is not passed: a file that only breaks partway through is
        # still an invalid CSV we want to reject at ingestion time.
        pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise InvalidCSVError(f"CSV file is empty: {path}") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InvalidCSVError(f"File is not valid CSV: {path} ({exc})") from exc


def ingest_csv(source: Path | str, *, raw_store: RawDataStore | None = None) -> DatasetReference:
    """Ingest a single CSV file and return a :class:`DatasetReference`.

    The raw file is copied into ``raw_store`` (defaulting to
    ``data/raw/``) as a read-only copy. The original file passed by the
    caller is left completely untouched.
    """
    path = Path(source).expanduser().resolve()
    _validate_source(path)
    _assert_parseable_csv(path)

    store = raw_store or RawDataStore.default()
    dataset_id = _new_dataset_id()

    stored_path = store.store(path, dataset_id=dataset_id, original_filename=path.name)

    reference = DatasetReference(
        dataset_id=dataset_id,
        original_filename=path.name,
        source_format=DatasetFormat.CSV,
        raw_path=stored_path,
        size_bytes=stored_path.stat().st_size,
        sha256=sha256_of_file(stored_path),
        created_at=dt.datetime.now(dt.UTC),
    )
    store.write_reference_sidecar(dataset_id, reference.model_dump_json(indent=2))
    return reference
