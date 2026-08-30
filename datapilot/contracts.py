"""Shared, cross-engine data contracts.

These types are the *handoff objects* between engines. They live in the
shared core (not inside any single engine) so that, e.g., ``profiling``
can accept whatever ``ingestion`` produced without the two packages
importing each other.

Design rules for everything in this module:

* Plain data, no behaviour that touches datasets.
* Fully serialisable to JSON (machine-readable for the future data-quality
  engine, API, frontend, and AI engine).
* Immutable where it represents a fact that already happened.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DatasetFormat(str, Enum):
    """Source file formats DataPilot can ingest.

    Only ``CSV`` is supported today; other members are added in the phase
    that implements their reader.
    """

    CSV = "csv"


class ColumnType(str, Enum):
    """Deterministically inferred logical type of a column.

    This is a *best-effort read* of the data as it currently sits. It is
    never used to coerce or transform values — mismatches between the
    stored dtype and the apparent type are a data-quality concern handled
    in Phase 2.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class DatasetReference(BaseModel):
    """An immutable pointer to an ingested raw dataset.

    Produced by ``data_engine.ingestion``; consumed by
    ``data_engine.profiling`` (and later stages). It describes the *file*
    that was ingested — never its contents. Content-level facts (row
    counts, column types, ...) are the profiler's responsibility.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(description="Unique identifier assigned at ingestion time.")
    original_filename: str = Field(description="Name of the file as the caller supplied it.")
    source_format: DatasetFormat
    raw_path: Path = Field(description="Absolute path to the preserved, read-only raw copy.")
    size_bytes: int = Field(ge=0, description="Size of the raw copy in bytes.")
    sha256: str = Field(description="SHA-256 of the raw bytes, for integrity verification.")
    created_at: _dt.datetime = Field(description="UTC timestamp of ingestion.")
