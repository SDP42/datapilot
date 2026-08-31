"""Phase 4 — deterministic richer distribution analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data_engine.eda import (
    DistributionAnalysis,
    DistributionStatus,
    EDAReport,
    analyze_dataframe,
    analyze_distribution,
)
from data_engine.eda.distribution_models import MAX_HISTOGRAM_BINS


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "b_norm": rng.normal(size=60),
            "a_skew": np.concatenate([np.arange(50, dtype=float), [500.0] * 10]),
            "const": [3.0] * 60,
            "with_missing": [float(i) if i % 3 else None for i in range(60)],
            "cat": ["x", "y"] * 30,
        }
    )


def _col(result: DistributionAnalysis, name: str):
    return next(c for c in result.columns if c.column == name)


# --- structure / determinism ----------------------------------------


def test_only_numeric_columns_sorted_alphabetically(df):
    result = analyze_distribution(df)
    assert [c.column for c in result.columns] == [
        "a_skew",
        "b_norm",
        "const",
        "with_missing",
    ]


def test_row_order_invariant(df):
    a = analyze_distribution(df)
    b = analyze_distribution(df.sample(frac=1.0, random_state=1).reset_index(drop=True))
    assert a.model_dump() == b.model_dump()


def test_read_only(df):
    before = df.copy(deep=True)
    analyze_distribution(df)
    pd.testing.assert_frame_equal(df, before)


def test_json_round_trip(df):
    result = analyze_distribution(df)
    dumped = result.model_dump_json()
    assert DistributionAnalysis.model_validate_json(dumped).model_dump() == result.model_dump()


# --- statistic correctness -----------------------------------------


def test_moments_match_scipy_and_pandas(df):
    c = _col(analyze_distribution(df), "a_skew")
    x = df["a_skew"].to_numpy(dtype=float)
    assert c.count == 60
    assert c.missing_count == 0
    assert c.minimum == pytest.approx(0.0)
    assert c.maximum == pytest.approx(500.0)
    assert c.mean == pytest.approx(float(np.mean(x)))
    assert c.median == pytest.approx(float(np.median(x)))
    assert c.variance == pytest.approx(float(np.var(x, ddof=1)))
    assert c.std == pytest.approx(float(np.std(x, ddof=1)))
    assert c.skewness == pytest.approx(float(stats.skew(x, bias=False)))
    assert c.kurtosis == pytest.approx(float(stats.kurtosis(x, fisher=True, bias=False)))
    assert c.skewness == pytest.approx(float(pd.Series(x).skew()))
    assert c.kurtosis == pytest.approx(float(pd.Series(x).kurt()))
    assert c.unique_count == 51


def test_quantiles_are_zero_to_one_and_match_numpy(df):
    c = _col(analyze_distribution(df), "b_norm")
    probs = [q.quantile for q in c.quantiles]
    assert probs == [0.0, 0.25, 0.5, 0.75, 1.0]
    x = df["b_norm"].to_numpy(dtype=float)
    assert c.quantiles[0].value == pytest.approx(float(np.min(x)))
    assert c.quantiles[-1].value == pytest.approx(float(np.max(x)))
    assert c.quantiles[1].value == pytest.approx(float(np.quantile(x, 0.25)))


# --- histogram -----------------------------------------------------


def test_histogram_sturges_bins_and_counts(df):
    c = _col(analyze_distribution(df), "b_norm")
    h = c.histogram
    assert h.status is DistributionStatus.COMPLETED
    assert h.bin_rule == "sturges"
    assert h.n_bins == int(np.ceil(np.log2(60))) + 1
    assert len(h.bin_edges) == h.n_bins + 1
    assert len(h.bins) == h.n_bins
    assert sum(b.count for b in h.bins) == c.count == h.total_count
    x = df["b_norm"].to_numpy(dtype=float)
    assert h.bin_edges[0] == pytest.approx(float(np.min(x)))
    assert h.bin_edges[-1] == pytest.approx(float(np.max(x)))


def test_histogram_bins_are_contiguous(df):
    h = _col(analyze_distribution(df), "a_skew").histogram
    for left, right in zip(h.bins, h.bins[1:]):
        assert left.right_edge == pytest.approx(right.left_edge)


@pytest.mark.parametrize("n", [2, 8, 64, 1024, 200_000])
def test_histogram_bin_count_follows_rule_and_never_exceeds_cap(n):
    from data_engine.eda.distribution import _histogram

    values = np.linspace(0.0, 1.0, n)
    h = _histogram(values)
    expected = min(int(np.ceil(np.log2(n))) + 1, MAX_HISTOGRAM_BINS)
    assert h.n_bins == expected
    assert h.n_bins <= MAX_HISTOGRAM_BINS


# --- degenerate cases --------------------------------------------


def test_constant_column_keeps_location_stats_drops_shape(df):
    c = _col(analyze_distribution(df), "const")
    assert c.status is DistributionStatus.COMPLETED
    assert c.minimum == c.maximum == c.mean == c.median == 3.0
    assert c.variance == 0.0
    assert c.std == 0.0
    assert c.skewness is None
    assert c.kurtosis is None
    assert c.histogram.status is DistributionStatus.UNAVAILABLE
    assert c.histogram.bin_edges == []
    assert any("constant" in n for n in c.notes)


def test_all_missing_column_is_unavailable():
    result = analyze_distribution(pd.DataFrame({"x": [np.nan, np.nan, np.nan]}))
    c = result.columns[0]
    assert c.status is DistributionStatus.UNAVAILABLE
    assert c.reason is not None
    assert c.count == 0
    assert c.missing_count == 3
    assert c.mean is None
    assert all(q.value is None for q in c.quantiles)
    assert c.histogram.status is DistributionStatus.UNAVAILABLE


def test_two_observations_have_no_skew_or_kurtosis():
    c = analyze_distribution(pd.DataFrame({"x": [1.0, 5.0]})).columns[0]
    assert c.status is DistributionStatus.COMPLETED
    assert c.std is not None and c.variance is not None
    assert c.skewness is None
    assert c.kurtosis is None
    assert any("skewness unavailable" in n for n in c.notes)


def test_three_observations_have_skew_but_no_kurtosis():
    c = analyze_distribution(pd.DataFrame({"x": [1.0, 2.0, 9.0]})).columns[0]
    assert c.skewness is not None
    assert c.kurtosis is None


def test_non_finite_values_excluded_not_faked():
    c = analyze_distribution(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.inf, -np.inf]})).columns[0]
    assert c.count == 6
    assert c.minimum == 1.0
    assert c.maximum == 4.0
    assert np.isfinite(c.mean)
    assert any("non-finite" in n for n in c.notes)


def test_missing_counted_and_stats_use_valid_only(df):
    c = _col(analyze_distribution(df), "with_missing")
    valid = df["with_missing"].dropna().to_numpy(dtype=float)
    assert c.count == valid.size
    assert c.missing_count == 60 - valid.size
    assert c.mean == pytest.approx(float(np.mean(valid)))


def test_empty_dataframe_yields_empty_analysis():
    result = analyze_distribution(pd.DataFrame())
    assert result.columns == []
    assert result.notes == []


# --- integration into EDAReport ----------------------------------


def test_analyze_dataframe_includes_distribution(df):
    report = analyze_dataframe(df)
    assert isinstance(report.distribution, DistributionAnalysis)
    assert [c.column for c in report.distribution.columns] == [
        "a_skew",
        "b_norm",
        "const",
        "with_missing",
    ]


def test_old_eda_json_without_distribution_still_validates(df):
    payload = analyze_dataframe(df).model_dump(mode="json")
    payload.pop("distribution")
    payload.pop("quality_cross_reference")
    restored = EDAReport.model_validate(payload)
    assert restored.distribution.columns == []
    assert restored.quality_cross_reference.entries == []
