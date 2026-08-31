"""Deterministic univariate EDA.

Read-only: ``df`` is never modified. Every statistic that cannot be
computed is represented as ``None`` — never invented, never imputed.
"""

from __future__ import annotations

import math

import pandas as pd
from pandas.api import types as ptypes

from .models import (
    DEFAULT_TOP_N,
    FIXED_QUANTILES,
    CategoricalColumnAnalysis,
    ColumnMissingness,
    DatetimeColumnAnalysis,
    EDAColumnKind,
    MissingnessAnalysis,
    NumericColumnAnalysis,
    QuantileValue,
    TopValue,
    UnivariateAnalysis,
)


def _clean_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def classify_column(series: pd.Series) -> EDAColumnKind:
    """Classify a column by its actual pandas dtype (not a heuristic)."""
    if ptypes.is_datetime64_any_dtype(series):
        return EDAColumnKind.DATETIME
    if ptypes.is_bool_dtype(series):
        return EDAColumnKind.CATEGORICAL
    if ptypes.is_numeric_dtype(series):
        return EDAColumnKind.NUMERIC
    return EDAColumnKind.CATEGORICAL


def classify_columns(df: pd.DataFrame) -> dict[str, EDAColumnKind]:
    return {str(col): classify_column(df[col]) for col in df.columns}


def _missing(series: pd.Series, n_rows: int) -> tuple[int, int, float]:
    missing = int(series.isna().sum())
    count = int(series.notna().sum())
    pct = round(100.0 * missing / n_rows, 6) if n_rows else 0.0
    return count, missing, pct


def analyze_numeric_column(series: pd.Series, n_rows: int) -> NumericColumnAnalysis:
    count, missing, pct = _missing(series, n_rows)
    non_null = series.dropna()
    quantile_values = (
        {q: _clean_float(non_null.quantile(q)) for q in FIXED_QUANTILES}
        if not non_null.empty
        else {q: None for q in FIXED_QUANTILES}
    )
    return NumericColumnAnalysis(
        column=str(series.name),
        count=count,
        missing_count=missing,
        missing_percentage=pct,
        mean=_clean_float(non_null.mean()) if not non_null.empty else None,
        median=_clean_float(non_null.median()) if not non_null.empty else None,
        std=_clean_float(non_null.std()) if len(non_null) > 1 else None,
        minimum=_clean_float(non_null.min()) if not non_null.empty else None,
        maximum=_clean_float(non_null.max()) if not non_null.empty else None,
        quantiles=[QuantileValue(quantile=q, value=v) for q, v in quantile_values.items()],
    )


def analyze_categorical_column(
    series: pd.Series, n_rows: int, *, top_n: int = DEFAULT_TOP_N
) -> CategoricalColumnAnalysis:
    count, missing, pct = _missing(series, n_rows)
    non_null = series.dropna()
    counts = non_null.value_counts()
    # deterministic ordering: highest count first, ties broken by string form
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    top = [
        TopValue(
            value=str(value),
            count=int(freq),
            frequency=round(int(freq) / count, 6) if count else 0.0,
        )
        for value, freq in ordered[:top_n]
    ]
    unique = int(non_null.nunique())
    return CategoricalColumnAnalysis(
        column=str(series.name),
        count=count,
        missing_count=missing,
        missing_percentage=pct,
        unique_count=unique,
        cardinality_ratio=round(unique / n_rows, 6) if n_rows else None,
        top_values=top,
    )


def analyze_datetime_column(series: pd.Series, n_rows: int) -> DatetimeColumnAnalysis:
    count, missing, pct = _missing(series, n_rows)
    non_null = series.dropna()
    return DatetimeColumnAnalysis(
        column=str(series.name),
        count=count,
        missing_count=missing,
        missing_percentage=pct,
        minimum=non_null.min().isoformat() if not non_null.empty else None,
        maximum=non_null.max().isoformat() if not non_null.empty else None,
        unique_count=int(non_null.nunique()),
    )


def analyze_missingness(df: pd.DataFrame) -> MissingnessAnalysis:
    n_rows = len(df)
    total_cells = n_rows * int(df.shape[1])
    per_column = [
        ColumnMissingness(
            column=str(col),
            missing_count=int(df[col].isna().sum()),
            missing_percentage=round(100.0 * int(df[col].isna().sum()) / n_rows, 6)
            if n_rows
            else 0.0,
        )
        for col in df.columns
    ]
    total_missing = sum(c.missing_count for c in per_column)
    return MissingnessAnalysis(
        total_cells=total_cells,
        total_missing_cells=total_missing,
        missing_percentage=round(100.0 * total_missing / total_cells, 6) if total_cells else 0.0,
        columns=per_column,
    )


def analyze_univariate(df: pd.DataFrame, *, top_n: int = DEFAULT_TOP_N) -> UnivariateAnalysis:
    """Classify every column and produce its univariate summary. ``df`` unchanged."""
    n_rows = len(df)
    kinds = classify_columns(df)

    numeric: list[NumericColumnAnalysis] = []
    categorical: list[CategoricalColumnAnalysis] = []
    datetime_: list[DatetimeColumnAnalysis] = []
    for col in df.columns:  # preserve dataframe column order
        kind = kinds[str(col)]
        if kind is EDAColumnKind.NUMERIC:
            numeric.append(analyze_numeric_column(df[col], n_rows))
        elif kind is EDAColumnKind.DATETIME:
            datetime_.append(analyze_datetime_column(df[col], n_rows))
        else:
            categorical.append(analyze_categorical_column(df[col], n_rows, top_n=top_n))

    return UnivariateAnalysis(
        numeric=numeric,
        categorical=categorical,
        datetime=datetime_,
        missingness=analyze_missingness(df),
    )
