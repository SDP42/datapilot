"""Dataset ingestion: validate a source file, preserve an immutable raw
copy, and return a structured :class:`~datapilot.contracts.DatasetReference`.

Public entrypoint::

    from data_engine.ingestion import ingest_dataset
    reference = ingest_dataset("customers.csv")
    reference = ingest_dataset("customers.xlsx")

CSV and Excel (.xlsx, first sheet only) are supported. ``ingest_dataset``
dispatches on the file extension so new formats can be added without
changing callers.
"""

from __future__ import annotations

from pathlib import Path

from datapilot.contracts import DatasetReference

from .csv_ingestor import SUPPORTED_SUFFIXES as _CSV_SUFFIXES
from .csv_ingestor import ingest_csv
from .errors import (
    IngestionError,
    InvalidCSVError,
    InvalidExcelError,
    SourceFileNotFoundError,
    SourceFileNotReadableError,
    UnsupportedFormatError,
)
from .excel_ingestor import SUPPORTED_SUFFIXES as _EXCEL_SUFFIXES
from .excel_ingestor import ingest_excel
from .raw_store import RawDataStore

SUPPORTED_SUFFIXES = _CSV_SUFFIXES | _EXCEL_SUFFIXES

__all__ = [
    "SUPPORTED_SUFFIXES",
    "IngestionError",
    "InvalidCSVError",
    "InvalidExcelError",
    "RawDataStore",
    "SourceFileNotFoundError",
    "SourceFileNotReadableError",
    "UnsupportedFormatError",
    "ingest_csv",
    "ingest_dataset",
    "ingest_excel",
]


def ingest_dataset(
    source: Path | str, *, raw_store: RawDataStore | None = None
) -> DatasetReference:
    """Ingest ``source`` by dispatching on its file extension."""
    suffix = Path(source).suffix.lower()
    if suffix in _CSV_SUFFIXES:
        return ingest_csv(source, raw_store=raw_store)
    if suffix in _EXCEL_SUFFIXES:
        return ingest_excel(source, raw_store=raw_store)
    raise UnsupportedFormatError(
        f"Unsupported extension {suffix!r}; ingestion currently accepts: "
        f"{sorted(SUPPORTED_SUFFIXES)}"
    )
