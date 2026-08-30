"""Structured, machine-readable result models for dataset profiling.

Every field is JSON-serialisable so the data-quality engine, API,
frontend, and AI engine can consume a profile without parsing prose.
Get a plain dict with ``profile.model_dump(mode="json")`` or a JSON
string with ``profile.model_dump_json()``.

These models describe the data; they never hold a reference to the
DataFrame itself.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, Field

from datapilot.contracts import ColumnType

PROFILER_VERSION = "1"


class NumericColumnStats(BaseModel):
    """Basic descriptive statistics for a numeric column (non-null values)."""

    count: int
    mean: float | None
    std: float | None
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None


class CategoricalColumnStats(BaseModel):
    """Basic frequency statistics for a categorical / text column."""

    distinct_count: int
    most_frequent: str | None
    most_frequent_count: int | None
    sample_values: list[str] = Field(
        default_factory=list,
        description="Up to 5 distinct values, as strings, for a quick glance.",
    )


class DatetimeColumnStats(BaseModel):
    """Range of a datetime-like column, as ISO-8601 strings."""

    minimum: str | None
    maximum: str | None


class ColumnProfile(BaseModel):
    """Profile of a single column."""

    name: str
    pandas_dtype: str = Field(description="The literal pandas dtype, e.g. 'int64'.")
    inferred_type: ColumnType
    missing_count: int
    missing_percentage: float
    unique_count: int = Field(description="Distinct non-null values.")
    numeric_stats: NumericColumnStats | None = None
    categorical_stats: CategoricalColumnStats | None = None
    datetime_stats: DatetimeColumnStats | None = None


class DatasetProfile(BaseModel):
    """The complete structured profile of one ingested dataset."""

    dataset_id: str
    profiler_version: str = PROFILER_VERSION
    generated_at: _dt.datetime

    n_rows: int
    n_columns: int
    column_names: list[str]
    duplicate_row_count: int

    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]

    columns: list[ColumnProfile]
