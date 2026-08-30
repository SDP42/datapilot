"""Dataset ingestion: validate a source file, preserve an immutable raw
copy, and return a structured :class:`~datapilot.contracts.DatasetReference`.

Public entrypoint::

    from data_engine.ingestion import ingest_dataset
    reference = ingest_dataset("customers.csv")

Only CSV is supported today. ``ingest_dataset`` dispatches on the file
extension so new formats can be added without changing callers.
"""

from __future__ import annotations

from pathlib import Path

from datapilot.contracts import DatasetReference

from .csv_ingestor import SUPPORTED_SUFFIXES, ingest_csv
from .errors import (
    IngestionError,
    InvalidCSVError,
    SourceFileNotFoundError,
    SourceFileNotReadableError,
    UnsupportedFormatError,
)
from .raw_store import RawDataStore

__all__ = [
    "IngestionError",
    "InvalidCSVError",
    "RawDataStore",
    "SourceFileNotFoundError",
    "SourceFileNotReadableError",
    "UnsupportedFormatError",
    "ingest_csv",
    "ingest_dataset",
]


def ingest_dataset(
    source: Path | str, *, raw_store: RawDataStore | None = None
) -> DatasetReference:
    """Ingest ``source`` by dispatching on its file extension."""
    suffix = Path(source).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return ingest_csv(source, raw_store=raw_store)
    raise UnsupportedFormatError(
        f"Unsupported extension {suffix!r}; ingestion currently accepts: "
        f"{sorted(SUPPORTED_SUFFIXES)}"
    )
