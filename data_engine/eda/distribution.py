"""Deterministic richer distribution analysis for numeric columns.

Goes further than the univariate numeric summary: variance, skewness,
excess kurtosis, a full 0.00-1.00 quantile set, and a structured
histogram (bin edges + counts, no rendering) that the future
visualization layer can consume directly.

Observational and read-only: ``df`` is never modified, no imputation, no
row repair. Rows are dropped only where a statistic inherently needs
valid / finite observations, and the kept count is always reported.
Degenerate inputs (no valid observations, a constant column, too few
observations for a shape statistic, non-finite values) yield ``None``
plus an explicit reason — never a crash, never an invented number.

See :mod:`data_engine.eda.distribution_models` for the exact statistical
conventions (skew / kurtosis definitions, the histogram bin rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .distribution_models import (
    DISTRIBUTION_QUANTILES,
    DISTRIBUTION_ROUND,
    MAX_DISTRIBUTION_COLUMNS,
    MAX_HISTOGRAM_BINS,
    MIN_OBS_FOR_KURTOSIS,
    MIN_OBS_FOR_SKEWNESS,
    MIN_OBS_FOR_STD,
    DistributionAnalysis,
    DistributionQuantile,
    DistributionStatus,
    Histogram,
    HistogramBin,
    NumericDistribution,
)
from .models import EDAColumnKind
from .univariate import _clean_float, classify_columns

_ROUND = DISTRIBUTION_ROUND


def _num(value: object) -> float | None:
    """Clean (NaN/inf -> None) and round for deterministic serialization."""
    cleaned = _clean_float(value)
    if cleaned is None or not np.isfinite(cleaned):
        return None
    return round(cleaned, _ROUND)


def _empty_quantiles() -> list[DistributionQuantile]:
    return [DistributionQuantile(quantile=q, value=None) for q in DISTRIBUTION_QUANTILES]


def sturges_bin_count(n: int) -> int:
    """Deterministic histogram bin count — Sturges' rule ``ceil(log2(n)) + 1``,
    clamped to ``[1, MAX_HISTOGRAM_BINS]``.

    The single source of truth for the bin count, shared by this layer
    and the visualization layer so the two never diverge.
    """
    if n <= 1:
        return 1
    k = int(np.ceil(np.log2(n))) + 1
    return max(1, min(k, MAX_HISTOGRAM_BINS))


def _histogram(values: np.ndarray) -> Histogram:
    """Structured equal-width histogram over the finite ``values``.

    Bin count: Sturges' rule ``ceil(log2(n)) + 1``, clamped to
    ``[1, MAX_HISTOGRAM_BINS]``. A constant column has no range and is
    reported ``unavailable`` rather than producing degenerate edges.
    """
    n = int(values.size)
    if n == 0:
        return Histogram(status=DistributionStatus.UNAVAILABLE, reason="no finite observations")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        return Histogram(
            status=DistributionStatus.UNAVAILABLE,
            reason="constant column: a histogram needs a non-zero value range",
        )

    k = sturges_bin_count(n)

    counts, edges = np.histogram(values, bins=k)
    bins = [
        HistogramBin(
            left_edge=round(float(edges[i]), _ROUND),
            right_edge=round(float(edges[i + 1]), _ROUND),
            count=int(counts[i]),
        )
        for i in range(k)
    ]
    return Histogram(
        status=DistributionStatus.COMPLETED,
        n_bins=k,
        bin_edges=[round(float(e), _ROUND) for e in edges],
        bins=bins,
        total_count=int(counts.sum()),
    )


def analyze_numeric_distribution(series: pd.Series, n_rows: int) -> NumericDistribution:
    """Distribution of one numeric column. ``series`` / its frame unchanged."""
    name = str(series.name)
    missing = int(series.isna().sum())
    missing_pct = round(100.0 * missing / n_rows, 6) if n_rows else 0.0
    non_null = series.dropna()
    count = int(non_null.size)
    unique_count = int(non_null.nunique())

    if count == 0:
        return NumericDistribution(
            column=name,
            status=DistributionStatus.UNAVAILABLE,
            reason="no valid (non-null) observations",
            count=0,
            missing_count=missing,
            missing_percentage=missing_pct,
            unique_count=0,
            quantiles=_empty_quantiles(),
            histogram=Histogram(
                status=DistributionStatus.UNAVAILABLE, reason="no valid observations"
            ),
        )

    values = non_null.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    notes: list[str] = []
    non_finite = int(values.size - finite.size)
    if non_finite:
        notes.append(f"{non_finite} non-finite value(s) excluded from distribution statistics")

    if finite.size == 0:
        return NumericDistribution(
            column=name,
            status=DistributionStatus.UNAVAILABLE,
            reason="no finite observations",
            count=count,
            missing_count=missing,
            missing_percentage=missing_pct,
            unique_count=unique_count,
            quantiles=_empty_quantiles(),
            histogram=Histogram(
                status=DistributionStatus.UNAVAILABLE, reason="no finite observations"
            ),
            notes=notes,
        )

    nf = int(finite.size)
    is_constant = bool(np.min(finite) == np.max(finite))

    std = variance = None
    if nf >= MIN_OBS_FOR_STD:
        variance = _num(np.var(finite, ddof=1))
        std = _num(np.std(finite, ddof=1))
    else:
        notes.append(f"std/variance unavailable: fewer than {MIN_OBS_FOR_STD} finite observations")

    skewness = None
    if is_constant:
        notes.append("skewness unavailable: constant column (undefined)")
    elif nf < MIN_OBS_FOR_SKEWNESS:
        notes.append(f"skewness unavailable: fewer than {MIN_OBS_FOR_SKEWNESS} finite observations")
    else:
        skewness = _num(stats.skew(finite, bias=False))

    kurtosis = None
    if is_constant:
        notes.append("kurtosis unavailable: constant column (undefined)")
    elif nf < MIN_OBS_FOR_KURTOSIS:
        notes.append(f"kurtosis unavailable: fewer than {MIN_OBS_FOR_KURTOSIS} finite observations")
    else:
        kurtosis = _num(stats.kurtosis(finite, fisher=True, bias=False))

    quantiles = [
        DistributionQuantile(quantile=q, value=_num(np.quantile(finite, q)))
        for q in DISTRIBUTION_QUANTILES
    ]

    histogram = _histogram(finite)
    if histogram.status is DistributionStatus.UNAVAILABLE and is_constant:
        notes.append("histogram unavailable: constant column")

    return NumericDistribution(
        column=name,
        status=DistributionStatus.COMPLETED,
        count=count,
        missing_count=missing,
        missing_percentage=missing_pct,
        unique_count=unique_count,
        minimum=_num(np.min(finite)),
        maximum=_num(np.max(finite)),
        mean=_num(np.mean(finite)),
        median=_num(np.median(finite)),
        std=std,
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
        quantiles=quantiles,
        histogram=histogram,
        notes=notes,
    )


def analyze_distribution(df: pd.DataFrame) -> DistributionAnalysis:
    """Run the bounded, deterministic per-column distribution battery.

    Numeric columns only (classified by pandas dtype), in alphabetical
    order, capped at :data:`MAX_DISTRIBUTION_COLUMNS`. Row order does not
    affect the result. ``df`` is not modified.
    """
    notes: list[str] = []
    kinds = classify_columns(df)
    numeric_cols = sorted(c for c, k in kinds.items() if k is EDAColumnKind.NUMERIC)
    if len(numeric_cols) > MAX_DISTRIBUTION_COLUMNS:
        notes.append(
            f"{len(numeric_cols)} numeric columns exceed the cap of "
            f"{MAX_DISTRIBUTION_COLUMNS}; kept the first (sorted by column name)"
        )
        numeric_cols = numeric_cols[:MAX_DISTRIBUTION_COLUMNS]

    n_rows = len(df)
    columns = [analyze_numeric_distribution(df[col], n_rows) for col in numeric_cols]
    return DistributionAnalysis(columns=columns, notes=notes)
