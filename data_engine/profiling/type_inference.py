"""Deterministic, read-only column-type inference.

The goal is a *useful label* for each column, not a transformation. We
never coerce values. If a column is stored as text but looks like dates,
we label it ``DATETIME`` so downstream stages know to look closer — but
the stored dtype and the raw values are left exactly as they are.

Rules (in order):

1. pandas boolean dtype            -> BOOLEAN
2. pandas numeric dtype            -> NUMERIC
3. pandas datetime dtype           -> DATETIME
4. text/object/category column that parses as dates for >= 90% of a
   sample, and clearly looks date-like (contains separators/letters)
                                   -> DATETIME
5. anything else non-empty text    -> CATEGORICAL
6. entirely empty column           -> UNKNOWN
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from datapilot.contracts import ColumnType

_DATETIME_SAMPLE_SIZE = 1000
_DATETIME_MIN_MATCH_RATIO = 0.9
_DATE_LIKE_CHARS = set("-/:")


def _looks_date_like(values: pd.Series) -> bool:
    """Heuristic guard so pure-number strings aren't read as years."""
    as_str = values.astype(str)
    has_separator = as_str.str.contains(r"[-/:]", regex=True).mean() >= _DATETIME_MIN_MATCH_RATIO
    has_alpha = as_str.str.contains(r"[A-Za-z]", regex=True).mean() >= _DATETIME_MIN_MATCH_RATIO
    return bool(has_separator or has_alpha)


def _parses_as_datetime(values: pd.Series) -> bool:
    sample = values.dropna()
    if sample.empty:
        return False
    if len(sample) > _DATETIME_SAMPLE_SIZE:
        sample = sample.head(_DATETIME_SAMPLE_SIZE)
    if not _looks_date_like(sample):
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= _DATETIME_MIN_MATCH_RATIO)


def infer_column_type(series: pd.Series) -> ColumnType:
    if series.dropna().empty:
        return ColumnType.UNKNOWN
    if ptypes.is_bool_dtype(series):
        return ColumnType.BOOLEAN
    if ptypes.is_numeric_dtype(series):
        return ColumnType.NUMERIC
    if ptypes.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    is_categorical_dtype = isinstance(series.dtype, pd.CategoricalDtype)
    is_text = ptypes.is_object_dtype(series) or ptypes.is_string_dtype(series)
    if is_text or is_categorical_dtype:
        if is_text and _parses_as_datetime(series):
            return ColumnType.DATETIME
        return ColumnType.CATEGORICAL

    return ColumnType.UNKNOWN
