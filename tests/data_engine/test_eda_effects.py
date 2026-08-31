"""Phase 4 — deterministic effect-size / association measures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data_engine.eda import (
    EDAReport,
    EffectSizeAnalysis,
    EffectSizeResult,
    EffectStatus,
    analyze_dataframe,
    analyze_effect_sizes,
    correlation_ratio,
    cramers_v,
    mutual_information,
)

# --- correctness ------------------------------------------------------


def test_cramers_v_matches_manual_calculation():
    # perfectly dependent 2x2 -> V == 1.0
    perfect = pd.DataFrame({"x": ["p"] * 10 + ["q"] * 10, "y": ["u"] * 10 + ["v"] * 10})
    res = cramers_v(perfect, "x", "y")
    assert res.status is EffectStatus.COMPLETED
    assert res.effect_size == pytest.approx(1.0)
    assert res.n_observations == 20

    # partial dependence -> match sqrt(chi2 / (n*min(r-1,c-1)))
    frame = pd.DataFrame({"x": ["p", "q"] * 10, "y": ["u", "u", "u", "v"] * 5})
    table = pd.crosstab(frame["x"], frame["y"]).sort_index(axis=0).sort_index(axis=1)
    chi2, _, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    r, c = table.shape
    expected = math.sqrt(chi2 / (20 * min(r - 1, c - 1)))
    assert cramers_v(frame, "x", "y").effect_size == pytest.approx(round(expected, 10))


def test_cramers_v_independent_is_zero():
    indep = pd.DataFrame({"x": ["p", "q"] * 10, "y": (["u"] * 2 + ["v"] * 2) * 5})
    res = cramers_v(indep, "x", "y")
    assert res.status is EffectStatus.COMPLETED
    assert res.effect_size == pytest.approx(0.0)


def test_correlation_ratio_matches_manual_calculation():
    # groups: a=[1,1] mean 1 ; b=[3,5] mean 4 ; grand mean 2.5
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "n": [1.0, 1.0, 3.0, 5.0]})
    res = correlation_ratio(frame, "g", "n")
    values = np.array([1.0, 1.0, 3.0, 5.0])
    grand = values.mean()
    ss_total = np.sum((values - grand) ** 2)
    ss_between = 2 * (1.0 - grand) ** 2 + 2 * (4.0 - grand) ** 2
    assert res.status is EffectStatus.COMPLETED
    assert res.effect_size == pytest.approx(round(math.sqrt(ss_between / ss_total), 10))
    assert res.n_groups == 2
    assert res.n_observations == 4


def test_correlation_ratio_full_separation_is_one():
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "n": [1.0, 1.0, 3.0, 3.0]})
    assert correlation_ratio(frame, "g", "n").effect_size == pytest.approx(1.0)


def test_mutual_information_controlled_dataset():
    # perfectly dependent balanced 2-category -> MI = ln(2) nats
    dep = pd.DataFrame({"x": ["p"] * 10 + ["q"] * 10, "y": ["u"] * 10 + ["v"] * 10})
    res = mutual_information(dep, "x", "y")
    assert res.status is EffectStatus.COMPLETED
    assert res.effect_size == pytest.approx(round(math.log(2), 10))
    assert res.n_observations == 20

    # independent -> exactly 0 (a real computed 0, not a fake)
    indep = pd.DataFrame({"x": ["p", "q"] * 10, "y": (["u"] * 2 + ["v"] * 2) * 5})
    zero = mutual_information(indep, "x", "y")
    assert zero.status is EffectStatus.COMPLETED
    assert zero.effect_size == pytest.approx(0.0)


def test_repeated_calculation_is_identical():
    frame = pd.DataFrame(
        {"a": [1.0, 2, 3, 4, 5, 6], "c1": ["x", "y"] * 3, "c2": ["p", "p", "q", "q", "p", "q"]}
    )
    assert (
        analyze_effect_sizes(frame).model_dump() == analyze_effect_sizes(frame.copy()).model_dump()
    )


# --- missing data ------------------------------------------------


def test_missing_rows_excluded_and_counted():
    frame = pd.DataFrame(
        {
            "g": ["a", "a", None, "b", "b", "b"],
            "n": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )
    before = frame.copy(deep=True)
    res = correlation_ratio(frame, "g", "n")
    # valid rows: (a,1) (a,2) (b,5) (b,6)
    assert res.n_observations == 4
    assert res.n_groups == 2
    pd.testing.assert_frame_equal(frame, before)  # unchanged, nothing imputed


def test_cramers_v_missing_rows_excluded():
    frame = pd.DataFrame({"x": ["p", "q", None, "p"], "y": ["u", "v", "u", None]})
    res = cramers_v(frame, "x", "y")
    assert res.n_observations == 2  # only the (p,u) and (q,v) rows


def test_no_values_invented():
    frame = pd.DataFrame({"c": ["x", "x", "y"], "n": [1.0, np.nan, 3.0]})
    frame["allnan"] = pd.Series([np.nan] * 3, dtype="float64")
    before = frame.copy(deep=True)
    analyze_effect_sizes(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert frame["allnan"].isna().all()


# --- degenerate cases ------------------------------------------


def _assert_unavailable(res: EffectSizeResult) -> None:
    assert res.status is EffectStatus.UNAVAILABLE
    assert res.reason
    assert res.effect_size is None
    assert res.n_observations is None


def test_single_category_cramers_v_unavailable():
    frame = pd.DataFrame({"x": ["p", "p", "p"], "y": ["u", "v", "u"]})
    res = cramers_v(frame, "x", "y")
    assert res.status is EffectStatus.UNAVAILABLE
    assert "degenerate" in res.reason


def test_insufficient_groups_correlation_ratio_unavailable():
    frame = pd.DataFrame({"g": ["a", "a", "a"], "n": [1.0, 2.0, 3.0]})
    res = correlation_ratio(frame, "g", "n")
    assert res.status is EffectStatus.UNAVAILABLE
    assert "groups" in res.reason


def test_zero_numeric_variance_correlation_ratio_unavailable():
    frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "n": [5.0, 5.0, 5.0, 5.0]})
    res = correlation_ratio(frame, "g", "n")
    assert res.status is EffectStatus.UNAVAILABLE
    assert "variance" in res.reason


def test_no_valid_observations_unavailable():
    frame = pd.DataFrame({"x": [None, None], "y": [None, None], "n": [np.nan, np.nan]})
    _assert_unavailable(cramers_v(frame, "x", "y"))
    _assert_unavailable(correlation_ratio(frame, "x", "n"))
    _assert_unavailable(mutual_information(frame, "x", "y"))


def test_unsupported_mutual_information_input_unavailable():
    frame = pd.DataFrame({"d": pd.to_datetime(["2021-01-01", "2021-02-01"]), "y": ["a", "b"]})
    res = mutual_information(frame, "d", "y")
    assert res.status is EffectStatus.UNAVAILABLE
    assert "datetime" in res.reason


# --- determinism ---------------------------------------------


@pytest.fixture
def battery_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num1": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "num2": [8.0, 7, 6, 5, 4, 3, 2, 1],
            "catA": ["x", "x", "y", "y", "x", "y", "x", "y"],
            "catB": ["p", "q", "p", "q", "p", "q", "p", "q"],
        }
    )


def test_deterministic_pair_ordering(battery_df):
    result = analyze_effect_sizes(battery_df)
    assert [r.columns for r in result.cramers_v] == [["catA", "catB"]]
    assert [r.columns for r in result.correlation_ratio] == [
        ["catA", "num1"],
        ["catA", "num2"],
        ["catB", "num1"],
        ["catB", "num2"],
    ]
    assert [r.columns for r in result.mutual_information] == [
        ["catA", "catB"],
        ["catA", "num1"],
        ["catA", "num2"],
        ["catB", "num1"],
        ["catB", "num2"],
        ["num1", "num2"],
    ]


def test_deterministic_category_ordering():
    a = pd.DataFrame({"g": ["b", "a", "b", "a"], "n": [1.0, 2.0, 3.0, 4.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert correlation_ratio(a, "g", "n").effect_size == correlation_ratio(b, "g", "n").effect_size
    x = pd.DataFrame({"p": ["y", "x", "y", "x"], "q": ["v", "u", "u", "v"]})
    y = x.iloc[::-1].reset_index(drop=True)
    assert cramers_v(x, "p", "q").effect_size == cramers_v(y, "p", "q").effect_size


def test_deterministic_truncation(battery_df, monkeypatch):
    monkeypatch.setattr("data_engine.eda.effects.MAX_MUTUAL_INFORMATION_PAIRS", 1)
    result = analyze_effect_sizes(battery_df)
    assert len(result.mutual_information) == 1
    assert result.mutual_information[0].columns == ["catA", "catB"]
    assert any("exceed the cap of 1" in note for note in result.notes)
    monkeypatch.setattr("data_engine.eda.effects.MAX_MUTUAL_INFORMATION_PAIRS", 1)
    assert analyze_effect_sizes(battery_df).model_dump() == result.model_dump()


# --- serialization -----------------------------------------


def test_effect_size_result_json_round_trip(battery_df):
    one = analyze_effect_sizes(battery_df).cramers_v[0]
    assert EffectSizeResult.model_validate_json(one.model_dump_json()) == one


def test_effect_size_analysis_json_round_trip(battery_df):
    analysis = analyze_effect_sizes(battery_df)
    assert EffectSizeAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


def test_old_eda_report_without_effect_sizes_still_validates(battery_df):
    payload = analyze_dataframe(battery_df, dataset_id="ds-x").model_dump(mode="json")
    del payload["effect_sizes"]
    restored = EDAReport.model_validate(payload)
    assert restored.effect_sizes.cramers_v == []
    assert restored.effect_sizes.correlation_ratio == []
    assert restored.effect_sizes.mutual_information == []


# --- integration -------------------------------------------


def test_analyze_dataframe_populates_effect_sizes(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert isinstance(report.effect_sizes, EffectSizeAnalysis)
    assert report.effect_sizes.cramers_v
    assert report.effect_sizes.mutual_information


def test_statistical_tests_still_work_alongside_effect_sizes(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert report.statistical_tests.t_tests
    assert report.statistical_tests.alpha == 0.05


def test_existing_eda_fields_unchanged(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert report.column_names == list(battery_df.columns)
    assert [c.column for c in report.univariate.numeric] == ["num1", "num2"]
    assert report.bivariate.numeric_correlations


def test_dataframe_unchanged_by_effect_battery(battery_df):
    before = battery_df.copy(deep=True)
    analyze_effect_sizes(battery_df)
    cramers_v(battery_df, "catA", "catB")
    correlation_ratio(battery_df, "catA", "num1")
    mutual_information(battery_df, "num1", "num2")
    pd.testing.assert_frame_equal(battery_df, before)
