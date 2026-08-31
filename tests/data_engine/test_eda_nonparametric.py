"""Phase 4 — deterministic non-parametric statistical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data_engine.eda import (
    EDAReport,
    NonParametricAnalysis,
    NonParametricTestResult,
    NonParametricTestStatus,
    analyze_dataframe,
    analyze_nonparametric,
    kendall_rank_correlation,
    kruskal_wallis,
    mann_whitney_u,
    spearman_rank_correlation,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "b": [2.0, 1, 4, 3, 6, 5, 8, 7],
            "c": [8.0, 7, 6, 5, 4, 3, 2, 1],
            "grp2": ["x", "x", "y", "y", "x", "y", "x", "y"],
            "grp3": ["p", "q", "r", "p", "q", "r", "p", "q"],
        }
    )


# --- correctness ------------------------------------------------------


def test_spearman_matches_scipy(df):
    res = spearman_rank_correlation(df, "a", "b")
    ref = stats.spearmanr(df["a"], df["b"])
    assert res.status is NonParametricTestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.n_observations == 8
    assert res.significant is (float(ref.pvalue) < 0.05)


def test_kendall_matches_scipy(df):
    res = kendall_rank_correlation(df, "a", "c")
    ref = stats.kendalltau(df["a"], df["c"])
    assert res.status is NonParametricTestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.n_observations == 8


def test_mann_whitney_matches_scipy(df):
    res = mann_whitney_u(df, "grp2", "a")
    ga = df.loc[df["grp2"] == "x", "a"]
    gb = df.loc[df["grp2"] == "y", "a"]
    ref = stats.mannwhitneyu(ga, gb, alternative="two-sided")
    assert res.status is NonParametricTestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.n_groups == 2
    assert res.n_observations == 8
    assert "alternative='two-sided'" in res.notes


def test_kruskal_wallis_matches_scipy(df):
    res = kruskal_wallis(df, "grp3", "a")
    groups = [df.loc[df["grp3"] == label, "a"] for label in sorted(df["grp3"].unique())]
    ref = stats.kruskal(*groups)
    assert res.status is NonParametricTestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.degrees_of_freedom == 2.0
    assert res.n_groups == 3
    assert res.n_observations == 8


# --- missing values --------------------------------------------


def test_missing_rows_excluded_and_counted():
    frame = pd.DataFrame({"a": [1.0, 2, 3, np.nan, 5], "b": [2.0, np.nan, 6, 8, 10]})
    before = frame.copy(deep=True)
    res = spearman_rank_correlation(frame, "a", "b")
    ref = stats.spearmanr([1.0, 3, 5], [2.0, 6, 10])
    assert res.n_observations == 3
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    pd.testing.assert_frame_equal(frame, before)  # nothing invented / filled


def test_kruskal_wallis_excludes_missing_and_documents_dropped_group():
    frame = pd.DataFrame(
        {
            "g": ["x", "x", "x", None, "y", "y", "y", "z"],
            "n": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    res = kruskal_wallis(frame, "g", "n")
    # x -> [1,2] (2), y -> [5,6,7] (3), z -> [8] (1 -> dropped)
    assert res.n_groups == 2
    assert res.n_observations == 5
    assert any("'z' dropped" in note for note in res.notes)


def test_no_values_invented(df):
    frame = df.copy()
    frame["allnan"] = pd.Series([np.nan] * len(frame), dtype="float64")
    before = frame.copy(deep=True)
    analyze_nonparametric(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert frame["allnan"].isna().all()


# --- degenerate cases ------------------------------------------


def _assert_unavailable(res: NonParametricTestResult) -> None:
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert res.reason
    assert res.statistic is None
    assert res.p_value is None
    assert res.significant is None
    assert res.n_observations is None


def test_constant_spearman_column_unavailable():
    frame = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]})
    res = spearman_rank_correlation(frame, "a", "b")
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert "constant" in res.reason


def test_constant_kendall_column_unavailable():
    frame = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
    res = kendall_rank_correlation(frame, "a", "b")
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert "constant" in res.reason


def test_no_valid_observations_unavailable():
    frame = pd.DataFrame(
        {"a": [np.nan, np.nan], "b": [np.nan, np.nan], "g": [None, None], "n": [np.nan, np.nan]}
    )
    _assert_unavailable(spearman_rank_correlation(frame, "a", "b"))
    _assert_unavailable(kendall_rank_correlation(frame, "a", "b"))
    _assert_unavailable(mann_whitney_u(frame, "g", "n"))
    _assert_unavailable(kruskal_wallis(frame, "g", "n"))


def test_mann_whitney_one_group_unavailable():
    frame = pd.DataFrame({"g": ["x", "x", "x"], "n": [1.0, 2.0, 3.0]})
    res = mann_whitney_u(frame, "g", "n")
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert "fewer than two groups" in res.reason


def test_mann_whitney_more_than_two_groups_unavailable(df):
    res = mann_whitney_u(df, "grp3", "a")
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert "more than two groups" in res.reason


def test_kruskal_wallis_one_usable_group_unavailable():
    frame = pd.DataFrame({"g": ["x", "x", "y"], "n": [1.0, 2.0, 3.0]})
    res = kruskal_wallis(frame, "g", "n")
    assert res.status is NonParametricTestStatus.UNAVAILABLE
    assert "groups" in res.reason


def test_non_finite_result_unavailable():
    # every value identical -> Kruskal-Wallis H is undefined
    frame = pd.DataFrame({"g": ["x", "x", "y", "y"], "n": [5.0, 5.0, 5.0, 5.0]})
    res = kruskal_wallis(frame, "g", "n")
    assert res.status is NonParametricTestStatus.UNAVAILABLE


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


def test_deterministic_candidate_ordering(battery_df):
    result = analyze_nonparametric(battery_df)
    assert [r.columns for r in result.spearman] == [["num1", "num2"]]
    assert [r.columns for r in result.kendall] == [["num1", "num2"]]
    assert [r.columns for r in result.mann_whitney_u] == [
        ["catA", "num1"],
        ["catA", "num2"],
        ["catB", "num1"],
        ["catB", "num2"],
    ]
    assert [r.columns for r in result.kruskal_wallis] == [
        ["catA", "num1"],
        ["catA", "num2"],
        ["catB", "num1"],
        ["catB", "num2"],
    ]


def test_deterministic_category_ordering():
    a = pd.DataFrame({"g": ["y", "x", "y", "x"], "n": [1.0, 2.0, 3.0, 4.0]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert mann_whitney_u(a, "g", "n").statistic == mann_whitney_u(b, "g", "n").statistic
    assert kruskal_wallis(a, "g", "n").statistic == kruskal_wallis(b, "g", "n").statistic


def test_deterministic_truncation(battery_df, monkeypatch):
    monkeypatch.setattr("data_engine.eda.nonparametric.MAX_SPEARMAN_PAIRS", 0)
    result = analyze_nonparametric(battery_df)
    assert result.spearman == []
    assert any("exceed the cap of 0" in note and "spearman" in note for note in result.notes)
    monkeypatch.setattr("data_engine.eda.nonparametric.MAX_SPEARMAN_PAIRS", 0)
    assert analyze_nonparametric(battery_df).model_dump() == result.model_dump()


def test_repeated_runs_identical(battery_df):
    assert (
        analyze_nonparametric(battery_df).model_dump()
        == analyze_nonparametric(battery_df.copy()).model_dump()
    )


def test_dataframe_unchanged_by_battery(battery_df):
    before = battery_df.copy(deep=True)
    analyze_nonparametric(battery_df)
    spearman_rank_correlation(battery_df, "num1", "num2")
    kendall_rank_correlation(battery_df, "num1", "num2")
    mann_whitney_u(battery_df, "catA", "num1")
    kruskal_wallis(battery_df, "catA", "num1")
    pd.testing.assert_frame_equal(battery_df, before)


# --- serialization -----------------------------------------


def test_nonparametric_test_result_json_round_trip(battery_df):
    one = analyze_nonparametric(battery_df).kruskal_wallis[0]
    assert NonParametricTestResult.model_validate_json(one.model_dump_json()) == one


def test_nonparametric_analysis_json_round_trip(battery_df):
    analysis = analyze_nonparametric(battery_df)
    assert NonParametricAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


def test_old_eda_report_without_nonparametric_still_validates(battery_df):
    payload = analyze_dataframe(battery_df, dataset_id="ds-x").model_dump(mode="json")
    del payload["nonparametric_tests"]
    restored = EDAReport.model_validate(payload)
    assert restored.nonparametric_tests.spearman == []
    assert restored.nonparametric_tests.mann_whitney_u == []


def test_unavailable_result_uses_none_not_fake_values():
    frame = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 3.0]})
    res = spearman_rank_correlation(frame, "a", "b")
    dumped = res.model_dump(mode="json")
    assert dumped["statistic"] is None
    assert dumped["p_value"] is None
    assert dumped["significant"] is None
    assert dumped["status"] == "unavailable"
    assert dumped["reason"]


# --- integration -------------------------------------------


def test_analyze_dataframe_populates_nonparametric(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert isinstance(report.nonparametric_tests, NonParametricAnalysis)
    assert report.nonparametric_tests.spearman
    assert report.nonparametric_tests.kruskal_wallis
    assert report.nonparametric_tests.alpha == 0.05


def test_existing_statistical_and_effect_sections_still_populated(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert report.statistical_tests.t_tests
    assert report.effect_sizes.cramers_v


def test_existing_eda_fields_unchanged(battery_df):
    report = analyze_dataframe(battery_df, dataset_id="ds-x")
    assert report.column_names == list(battery_df.columns)
    assert [c.column for c in report.univariate.numeric] == ["num1", "num2"]
    assert report.bivariate.numeric_correlations
