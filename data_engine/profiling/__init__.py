"""Dataset profiling: turn an ingested dataset into a structured,
machine-readable :class:`DatasetProfile`.

Typical use::

    from data_engine.ingestion import ingest_dataset
    from data_engine.profiling import profile_dataset

    reference = ingest_dataset("customers.csv")
    profile = profile_dataset(reference)
    payload = profile.model_dump(mode="json")   # hand to quality engine / API / UI

Profiling is strictly read-only: it never cleans, imputes, de-duplicates,
or otherwise changes the data. It only *describes* it.
"""

from __future__ import annotations

from .loader import load_dataframe
from .models import (
    CategoricalColumnStats,
    ColumnProfile,
    DatasetProfile,
    DatetimeColumnStats,
    NumericColumnStats,
)
from .profiler import profile_dataframe, profile_dataset
from .type_inference import infer_column_type

__all__ = [
    "CategoricalColumnStats",
    "ColumnProfile",
    "DatasetProfile",
    "DatetimeColumnStats",
    "NumericColumnStats",
    "infer_column_type",
    "load_dataframe",
    "profile_dataframe",
    "profile_dataset",
]
