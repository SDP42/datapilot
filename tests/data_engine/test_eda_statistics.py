"""Phase 4 — deterministic statistical-hypothesis-testing foundation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data_engine.eda import (
    EDAReport,
    StatisticalAnalysis,
    StatisticalTestResult,
    TestStatus,
    analyze_dataframe,
    analyze_statistics,
    chi_square_independence,
    one_way_anova,
    welch_t_test,
)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            "c": [10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "grade": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C"],
            "flag": ["y", "n", "y", "n", "y", "n", "y", "n", "y", "n"],
        }
    )


# --- statistical correctness ------------------------------------------


def test_welch_t_test_matches_scipy(df):
    res = welch_t_test(df, "a", "b")
    ref = stats.ttest_ind(df["a"], df["b"], equal_var=False)
    assert res.status is TestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.degrees_of_freedom == pytest.approx(round(float(ref.df), 10))
    assert res.n_observations == 10
    assert res.significant is (float(ref.pvalue) < 0.05)


def test_one_way_anova_matches_scipy(df):
    res = one_way_anova(df, "grade", "a")
    a = df.loc[df["grade"] == "A", "a"]
    b = df.loc[df["grade"] == "B", "a"]
    c = df.loc[df["grade"] == "C", "a"]
    ref = stats.f_oneway(a, b, c)
    assert res.status is TestStatus.COMPLETED
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    assert res.p_value == pytest.approx(float(ref.pvalue))
    assert res.n_groups == 3
    assert res.n_observations == 10


def test_chi_square_matches_scipy_and_hardcoded_independence():
    frame = pd.DataFrame({"x": ["p", "p", "q", "q"] * 5, "y": ["u", "v"] * 10})
    res = chi_square_independence(frame, "x", "y")
    table = pd.crosstab(frame["x"], frame["y"]).sort_index(axis=0).sort_index(axis=1)
    chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    assert res.statistic == pytest.approx(round(float(chi2), 10))
    assert res.p_value == pytest.approx(float(p))
    assert res.degrees_of_freedom == pytest.approx(float(dof))
    assert res.n_observations == 20

    perfectly_independent = pd.DataFrame({"x": ["p", "q"] * 20, "y": (["u"] * 2 + ["v"] * 2) * 10})
    ind = chi_square_independence(perfectly_independent, "x", "y")
    assert ind.statistic == pytest.approx(0.0)
    assert ind.p_value == pytest.approx(1.0)
    assert ind.degrees_of_freedom == 1.0


# --- missing-data handling ------------------------------------------


def test_t_test_excludes_missing_and_reports_count():
    frame = pd.DataFrame({"a": [1.0, 2, 3, np.nan, 5], "b": [2.0, np.nan, 6, 8, 10]})
    before = frame.copy(deep=True)
    res = welch_t_test(frame, "a", "b")
    ref = stats.ttest_ind([1.0, 3, 5], [2.0, 6, 10], equal_var=False)
    assert res.n_observations == 3
    assert res.statistic == pytest.approx(round(float(ref.statistic), 10))
    pd.testing.assert_frame_equal(frame, before)


def test_anova_excludes_missing_rows():
    frame = pd.DataFrame(
        {
            "g": ["x", "x", "x", None, "y", "y", "y", "z"],
            "n": [1.0, 2, np.nan, 4, 5, 6, 7, 8],
        }
    )
    res = one_way_anova(frame, "g", "n")
    # x has [1,2] (2 valid), y has [5,6,7] (3), z has [8] (1 -> dropped)
    assert res.n_groups == 2
    assert res.n_observations == 5
    assert any("'z' dropped" in note for note in res.notes)


def test_no_missing_values_are_invented(df):
    frame = df.copy()
    frame["all_nan"] = pd.Series([np.nan] * len(frame), dtype="float64")
    before = frame.copy(deep=True)
    analyze_statistics(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert frame["all_nan"].isna().all()


# --- degenerate / insufficient ------------------------------------


def _assert_unavailable(res: StatisticalTestResult) -> None:
    assert res.status is TestStatus.UNAVAILABLE
    assert res.reason
    assert res.statistic is None
    assert res.p_value is None
    assert res.significant is None
    assert res.n_observations is None


def test_insufficient_t_test_observations_unavailable():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    _assert_unavailable(welch_t_test(frame, "a", "b"))


def test_constant_column_t_test_unavailable():
    frame = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]})
    res = welch_t_test(frame, "a", "b")
    assert res.status is TestStatus.UNAVAILABLE
    assert "variance" in res.reason


def test_insufficient_anova_groups_unavailable():
    frame = pd.DataFrame({"g": ["x", "x", "x"], "n": [1.0, 2.0, 3.0]})
    res = one_way_anova(frame, "g", "n")
    assert res.status is TestStatus.UNAVAILABLE
    assert "groups" in res.reason


def test_degenerate_chi_square_table_unavailable():
    frame = pd.DataFrame({"x": ["p", "p", "p", "p"], "y": ["u", "v", "u", "v"]})
    res = chi_square_independence(frame, "x", "y")
    assert res.status is TestStatus.UNAVAILABLE
    assert "degenerate" in res.reason


def test_chi_square_no_observations_unavailable():
    frame = pd.DataFrame({"x": ["p", None, "q"], "y": [None, "u", None]})
    res = chi_square_independence(frame, "x", "y")
    assert res.status is TestStatus.UNAVAILABLE
    assert "no valid paired observations" in res.reason


def test_constant_inputs_do_not_crash():
    frame = pd.DataFrame(
        {"num": [5.0] * 6, "num2": [5.0] * 6, "cat": ["only"] * 6, "cat2": ["only"] * 6}
    )
    result = analyze_statistics(frame)  # must not raise
    for test in [*result.t_tests, *result.anova, *result.chi_square]:
        assert test.status is TestStatus.UNAVAILABLE


# --- determinism -------------------------------------------------


def test_same_dataframe_same_statistics(df):
    assert analyze_statistics(df).model_dump() == analyze_statistics(df.copy()).model_dump()


def test_deterministic_pair_ordering(df):
    result = analyze_statistics(df)
    assert [t.columns for t in result.t_tests] == [["a", "b"], ["a", "c"], ["b", "c"]]
    assert [t.columns for t in result.chi_square] == [["flag", "grade"]]
    assert [t.columns for t in result.anova] == [
        ["flag", "a"],
        ["flag", "b"],
        ["flag", "c"],
        ["grade", "a"],
        ["grade", "b"],
        ["grade", "c"],
    ]


def test_deterministic_truncation(df, monkeypatch):
    monkeypatch.setattr("data_engine.eda.statistics.MAX_TTEST_PAIRS", 1)
    result = analyze_statistics(df)
    assert len(result.t_tests) == 1
    assert result.t_tests[0].columns == ["a", "b"]  # first sorted pair
    assert any("exceed the cap of 1" in note for note in result.notes)
    # still deterministic
    monkeypatch.setattr("data_engine.eda.statistics.MAX_TTEST_PAIRS", 1)
    assert analyze_statistics(df).model_dump() == result.model_dump()


# --- read-only -------------------------------------------------


def test_dataframe_unchanged_by_statistics(df):
    before = df.copy(deep=True)
    analyze_statistics(df)
    welch_t_test(df, "a", "b")
    one_way_anova(df, "grade", "a")
    chi_square_independence(df, "grade", "flag")
    pd.testing.assert_frame_equal(df, before)


# --- serialization --------------------------------------------


def test_statistical_models_json_round_trip(df):
    analysis = analyze_statistics(df)
    restored = StatisticalAnalysis.model_validate_json(analysis.model_dump_json())
    assert restored == analysis

    one = analysis.t_tests[0]
    assert StatisticalTestResult.model_validate_json(one.model_dump_json()) == one


def test_unavailable_result_uses_none_not_fake_values():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    res = welch_t_test(frame, "a", "b")
    dumped = res.model_dump(mode="json")
    assert dumped["statistic"] is None
    assert dumped["p_value"] is None
    assert dumped["significant"] is None
    assert dumped["n_observations"] is None
    assert dumped["status"] == "unavailable"
    assert dumped["reason"]


# --- integration / backward compatibility --------------------


def test_eda_report_includes_statistical_tests(df):
    report = analyze_dataframe(df, dataset_id="ds-x")
    assert isinstance(report.statistical_tests, StatisticalAnalysis)
    assert report.statistical_tests.t_tests
    assert report.statistical_tests.alpha == 0.05


def test_old_eda_report_json_without_statistical_tests_still_validates(df):
    payload = analyze_dataframe(df, dataset_id="ds-x").model_dump(mode="json")
    del payload["statistical_tests"]
    restored = EDAReport.model_validate(payload)
    assert restored.statistical_tests.t_tests == []
    assert restored.statistical_tests.anova == []
    assert restored.statistical_tests.chi_square == []


def test_analyze_dataframe_still_deterministic(df):
    r1 = analyze_dataframe(df.copy(), dataset_id="ds-x").model_dump(mode="json")
    r2 = analyze_dataframe(df.copy(), dataset_id="ds-x").model_dump(mode="json")
    r1.pop("generated_at")
    r2.pop("generated_at")
    assert r1 == r2
