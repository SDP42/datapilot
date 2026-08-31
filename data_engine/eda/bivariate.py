"""Deterministic, intentionally-small bivariate EDA.

Implemented for this increment:

* numeric  <-> numeric      : Pearson correlation (with paired-obs count)
* categorical <-> numeric   : grouped count / mean / median
* categorical <-> categorical: contingency-table counts

No hypothesis testing, no p-values, no association measures — those are
later increments. Read-only: ``df`` is never modified.
"""

from __future__ import annotations

import math

import pandas as pd

from .models import (
    MAX_BIVARIATE_CARDINALITY,
    MAX_CONTINGENCY_ROWS,
    MAX_GROUPED_CATEGORIES,
    MAX_NUMERIC_PAIRS,
    BivariateSummary,
    CategoricalContingency,
    CategoricalNumericSummary,
    CategoryNumericGroup,
    ContingencyRow,
    EDAColumnKind,
    NumericPairCorrelation,
)
from .univariate import _clean_float, classify_columns


def _numeric_correlations(
    df: pd.DataFrame, columns: list[str], notes: list[str]
) -> list[NumericPairCorrelation]:
    pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i + 1 :]]
    if len(pairs) > MAX_NUMERIC_PAIRS:
        notes.append(
            f"numeric-numeric: {len(pairs)} pairs exceed the cap of {MAX_NUMERIC_PAIRS}; "
            "kept the first (sorted by column name)"
        )
        pairs = pairs[:MAX_NUMERIC_PAIRS]

    results: list[NumericPairCorrelation] = []
    for a, b in pairs:
        paired = df[[a, b]].dropna()
        n = len(paired)
        correlation: float | None = None
        if n >= 2 and paired[a].nunique() > 1 and paired[b].nunique() > 1:
            value = _clean_float(paired[a].corr(paired[b]))
            correlation = None if value is None or math.isnan(value) else round(value, 10)
        results.append(
            NumericPairCorrelation(
                column_a=a, column_b=b, n_observations=n, correlation=correlation
            )
        )
    return results


def _categorical_numeric(
    df: pd.DataFrame, cat_columns: list[str], num_columns: list[str], notes: list[str]
) -> list[CategoricalNumericSummary]:
    results: list[CategoricalNumericSummary] = []
    for cat in cat_columns:
        for num in num_columns:
            grouped = df[[cat, num]].dropna(subset=[cat])
            categories = sorted(str(c) for c in grouped[cat].astype("object").unique())
            truncated = len(categories) > MAX_GROUPED_CATEGORIES
            if truncated:
                notes.append(
                    f"categorical-numeric ({cat} x {num}): {len(categories)} categories exceed "
                    f"the cap of {MAX_GROUPED_CATEGORIES}; kept the first (sorted)"
                )
                categories = categories[:MAX_GROUPED_CATEGORIES]

            groups: list[CategoryNumericGroup] = []
            for category in categories:
                values = grouped.loc[grouped[cat].astype("object").astype(str) == category, num]
                non_null = values.dropna()
                groups.append(
                    CategoryNumericGroup(
                        category=category,
                        count=len(non_null),
                        mean=_clean_float(non_null.mean()) if not non_null.empty else None,
                        median=_clean_float(non_null.median()) if not non_null.empty else None,
                    )
                )
            results.append(
                CategoricalNumericSummary(
                    categorical_column=cat,
                    numeric_column=num,
                    groups=groups,
                    truncated=truncated,
                )
            )
    return results


def _categorical_categorical(
    df: pd.DataFrame, cat_columns: list[str], notes: list[str]
) -> list[CategoricalContingency]:
    results: list[CategoricalContingency] = []
    for i, a in enumerate(cat_columns):
        for b in cat_columns[i + 1 :]:
            pair = df[[a, b]].dropna()
            counts: dict[tuple[str, str], int] = {}
            for va, vb in zip(pair[a].astype(str), pair[b].astype(str), strict=True):
                counts[(va, vb)] = counts.get((va, vb), 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: kv[0])
            truncated = len(ordered) > MAX_CONTINGENCY_ROWS
            if truncated:
                notes.append(
                    f"categorical-categorical ({a} x {b}): {len(ordered)} contingency rows exceed "
                    f"the cap of {MAX_CONTINGENCY_ROWS}; kept the first (sorted)"
                )
                ordered = ordered[:MAX_CONTINGENCY_ROWS]
            results.append(
                CategoricalContingency(
                    column_a=a,
                    column_b=b,
                    rows=[
                        ContingencyRow(category_a=ca, category_b=cb, count=n)
                        for (ca, cb), n in ordered
                    ],
                    truncated=truncated,
                )
            )
    return results


def analyze_bivariate(df: pd.DataFrame) -> BivariateSummary:
    """Deterministic basic relationships. ``df`` is not modified."""
    kinds = classify_columns(df)
    numeric_cols = sorted(c for c, k in kinds.items() if k is EDAColumnKind.NUMERIC)

    categorical_cols: list[str] = []
    notes: list[str] = []
    for col in sorted(c for c, k in kinds.items() if k is EDAColumnKind.CATEGORICAL):
        if int(df[col].dropna().nunique()) <= MAX_BIVARIATE_CARDINALITY:
            categorical_cols.append(col)
        else:
            notes.append(
                f"'{col}' skipped in bivariate analysis: cardinality exceeds "
                f"{MAX_BIVARIATE_CARDINALITY}"
            )

    return BivariateSummary(
        numeric_correlations=_numeric_correlations(df, numeric_cols, notes),
        categorical_numeric=_categorical_numeric(df, categorical_cols, numeric_cols, notes),
        categorical_categorical=_categorical_categorical(df, categorical_cols, notes),
        notes=notes,
    )
