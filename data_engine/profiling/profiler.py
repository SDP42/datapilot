"""Deterministic dataset profiling.

Two entrypoints:

* :func:`profile_dataframe` — pure function, ``DataFrame -> DatasetProfile``.
* :func:`profile_dataset` — the contract-level call,
  ``DatasetReference -> DatasetProfile`` (loads, then profiles).

No LLM, no randomness, no mutation. The input DataFrame is only read.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

from datapilot.contracts import ColumnType, DatasetReference

from .loader import load_dataframe
from .models import (
    CategoricalColumnStats,
    ColumnProfile,
    DatasetProfile,
    DatetimeColumnStats,
    NumericColumnStats,
)
from .type_inference import infer_column_type

_SAMPLE_VALUES = 5


def _clean_float(value: Any) -> float | None:
    """Convert a numpy/pandas scalar to a JSON-safe float or ``None``."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _numeric_stats(series: pd.Series) -> NumericColumnStats:
    non_null = series.dropna()
    quantiles = non_null.quantile([0.25, 0.5, 0.75]) if not non_null.empty else {}
    return NumericColumnStats(
        count=int(non_null.count()),
        mean=_clean_float(non_null.mean()) if not non_null.empty else None,
        std=_clean_float(non_null.std()) if len(non_null) > 1 else None,
        minimum=_clean_float(non_null.min()) if not non_null.empty else None,
        q25=_clean_float(quantiles.get(0.25)) if len(quantiles) else None,
        median=_clean_float(quantiles.get(0.5)) if len(quantiles) else None,
        q75=_clean_float(quantiles.get(0.75)) if len(quantiles) else None,
        maximum=_clean_float(non_null.max()) if not non_null.empty else None,
    )


def _categorical_stats(series: pd.Series) -> CategoricalColumnStats:
    non_null = series.dropna()
    counts = non_null.value_counts()
    most_frequent = str(counts.index[0]) if not counts.empty else None
    most_frequent_count = int(counts.iloc[0]) if not counts.empty else None
    sample = [str(v) for v in counts.index[:_SAMPLE_VALUES]]
    return CategoricalColumnStats(
        distinct_count=int(non_null.nunique()),
        most_frequent=most_frequent,
        most_frequent_count=most_frequent_count,
        sample_values=sample,
    )


def _datetime_stats(series: pd.Series) -> DatetimeColumnStats:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = parsed.dropna()
    return DatetimeColumnStats(
        minimum=non_null.min().isoformat() if not non_null.empty else None,
        maximum=non_null.max().isoformat() if not non_null.empty else None,
    )


def _profile_column(series: pd.Series, n_rows: int) -> ColumnProfile:
    inferred = infer_column_type(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(100.0 * missing_count / n_rows, 4) if n_rows else 0.0

    profile = ColumnProfile(
        name=str(series.name),
        pandas_dtype=str(series.dtype),
        inferred_type=inferred,
        missing_count=missing_count,
        missing_percentage=missing_pct,
        unique_count=int(series.nunique(dropna=True)),
    )

    if inferred is ColumnType.NUMERIC:
        profile.numeric_stats = _numeric_stats(series)
    elif inferred is ColumnType.DATETIME:
        profile.datetime_stats = _datetime_stats(series)
    elif inferred in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
        profile.categorical_stats = _categorical_stats(series.astype("object"))

    return profile


def profile_dataframe(df: pd.DataFrame, *, dataset_id: str) -> DatasetProfile:
    """Profile an in-memory DataFrame. Pure function; ``df`` is not modified."""
    n_rows = len(df)
    columns = [_profile_column(df[col], n_rows) for col in df.columns]

    by_type: dict[ColumnType, list[str]] = {
        t: [] for t in (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.DATETIME)
    }
    for col in columns:
        if col.inferred_type in by_type:
            by_type[col.inferred_type].append(col.name)

    return DatasetProfile(
        dataset_id=dataset_id,
        generated_at=dt.datetime.now(dt.UTC),
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        column_names=[str(c) for c in df.columns],
        duplicate_row_count=int(df.duplicated().sum()),
        numeric_columns=by_type[ColumnType.NUMERIC],
        categorical_columns=by_type[ColumnType.CATEGORICAL],
        datetime_columns=by_type[ColumnType.DATETIME],
        columns=columns,
    )


def profile_dataset(reference: DatasetReference) -> DatasetProfile:
    """Contract-level entrypoint: load the referenced raw dataset, then profile it."""
    df = load_dataframe(reference)
    return profile_dataframe(df, dataset_id=reference.dataset_id)
