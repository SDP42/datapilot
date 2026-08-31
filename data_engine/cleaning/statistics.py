"""Compact before/after column statistics for the execution report.

Deliberately a *summary* — never a second copy of the data.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from .execution_models import ColumnStatistics


def column_statistics(df: pd.DataFrame, column: str) -> ColumnStatistics:
    series = df[column]
    n = len(df)
    missing = int(series.isna().sum())
    stats = ColumnStatistics(
        column=column,
        dtype=str(series.dtype),
        count=int(series.notna().sum()),
        missing_count=missing,
        missing_percentage=round(100.0 * missing / n, 4) if n else 0.0,
        unique_count=int(series.nunique(dropna=True)),
    )
    if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        non_null = series.dropna()
        if not non_null.empty:
            stats.minimum = float(non_null.min())
            stats.maximum = float(non_null.max())
            stats.mean = float(non_null.mean())
            stats.median = float(non_null.median())
    return stats


def total_missing_cells(df: pd.DataFrame) -> int:
    return int(df.isna().sum().sum())
