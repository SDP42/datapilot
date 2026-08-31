"""Phase 4 — deterministic, analysis-only EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.eda import (
    EDAReport,
    UnivariateAnalysis,
    analyze_bivariate,
    analyze_dataframe,
    analyze_dataset_version,
    analyze_univariate,
)
from data_engine.validation import VersionIntegrityError


def _num(df: UnivariateAnalysis, name: str):
    return next(c for c in df.numeric if c.column == name)


def _cat(uni: UnivariateAnalysis, name: str):
    return next(c for c in uni.categorical if c.column == name)


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40, 50, 30, 25, 60],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "empty_num": pd.Series([np.nan] * 8, dtype="float64"),
            "city": ["London", "Paris", "London", "Paris", "London", "Berlin", "London", None],
            "grade": ["A", "B", "A", "B", "C", "A", "B", "C"],
            "joined": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    None,
                    "2021-03-01",
                    "2021-01-15",
                    "2021-02-20",
                    "2021-01-01",
                    "2021-04-01",
                ]
            ),
        }
    )


# --- 1-4: numeric univariate --------------------------------------------


def test_numeric_summary(messy_df):
    uni = analyze_univariate(messy_df)
    age = _num(uni, "age")
    assert age.count == 7
    assert age.missing_count == 1
    assert age.minimum == 25.0
    assert age.maximum == 60.0
    assert age.mean == pytest.approx(np.nanmean(messy_df["age"]))
    assert age.median == pytest.approx(np.nanmedian(messy_df["age"]))
    assert age.std == pytest.approx(messy_df["age"].std())


def test_numeric_quantiles_use_fixed_set(messy_df):
    score = _num(analyze_univariate(messy_df), "score")
    assert [q.quantile for q in score.quantiles] == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert score.quantiles[2].value == pytest.approx(messy_df["score"].quantile(0.5))


def test_numeric_missing_values(messy_df):
    age = _num(analyze_univariate(messy_df), "age")
    assert age.missing_count == 1
    assert age.missing_percentage == pytest.approx(12.5)


def test_entirely_missing_numeric_column_is_kept_and_null(messy_df):
    empty = _num(analyze_univariate(messy_df), "empty_num")
    assert empty.count == 0
    assert empty.missing_count == 8
    assert empty.mean is None and empty.median is None and empty.std is None
    assert empty.minimum is None and empty.maximum is None
    assert all(q.value is None for q in empty.quantiles)


# --- 5-7: categorical univariate ------------------------------------


def test_categorical_frequency_analysis(messy_df):
    city = _cat(analyze_univariate(messy_df), "city")
    assert city.count == 7
    assert city.missing_count == 1
    assert city.unique_count == 3
    assert city.top_values[0].value == "London"
    assert city.top_values[0].count == 4
    assert city.top_values[0].frequency == pytest.approx(4 / 7)


def test_deterministic_categorical_tie_breaking():
    # 'x' and 'y' tie at 2 each; 'a' has 1 -> order must be x, y, a
    df = pd.DataFrame({"c": ["y", "x", "y", "x", "a"]})
    top = analyze_univariate(df).categorical[0].top_values
    assert [t.value for t in top] == ["x", "y", "a"]


def test_categorical_cardinality(messy_df):
    grade = _cat(analyze_univariate(messy_df), "grade")
    assert grade.unique_count == 3
    assert grade.cardinality_ratio == pytest.approx(3 / 8)


def test_categorical_column_with_no_non_null_values():
    df = pd.DataFrame({"blank": pd.Series([None] * 4, dtype="object")})
    cat = analyze_univariate(df).categorical[0]
    assert cat.count == 0
    assert cat.unique_count == 0
    assert cat.top_values == []
    assert cat.cardinality_ratio == 0.0


# --- 8: datetime univariate -----------------------------------------


def test_datetime_summary(messy_df):
    uni = analyze_univariate(messy_df)
    joined = next(c for c in uni.datetime if c.column == "joined")
    assert joined.count == 7
    assert joined.missing_count == 1
    assert joined.minimum == "2021-01-01T00:00:00"
    assert joined.maximum == "2021-04-01T00:00:00"
    assert joined.unique_count == 6


def test_string_dates_are_not_reinterpreted_as_datetime():
    df = pd.DataFrame({"looks_like_date": ["2021-01-01", "2021-02-01", "2021-03-01"]})
    uni = analyze_univariate(df)
    assert uni.datetime == []
    assert [c.column for c in uni.categorical] == ["looks_like_date"]


def test_datetime_column_with_no_valid_values():
    df = pd.DataFrame({"d": pd.Series([pd.NaT] * 3, dtype="datetime64[ns]")})
    dt_analysis = analyze_univariate(df).datetime[0]
    assert dt_analysis.count == 0
    assert dt_analysis.minimum is None and dt_analysis.maximum is None


# --- 9: missingness -------------------------------------------------


def test_missingness_summary(messy_df):
    m = analyze_univariate(messy_df).missingness
    assert m.total_cells == 8 * 6
    # empty_num(8) + age(1) + city(1) + joined(1)
    assert m.total_missing_cells == 11
    assert m.missing_percentage == pytest.approx(100.0 * 11 / 48)
    by_col = {c.column: c.missing_count for c in m.columns}
    assert by_col["empty_num"] == 8
    assert [c.column for c in m.columns] == list(messy_df.columns)


# --- 10-11: numeric-numeric correlation ----------------------------


def test_numeric_numeric_correlation(messy_df):
    biv = analyze_bivariate(messy_df)
    pair = next(c for c in biv.numeric_correlations if {c.column_a, c.column_b} == {"age", "score"})
    paired = messy_df[["age", "score"]].dropna()
    assert pair.n_observations == len(paired)
    assert pair.correlation == pytest.approx(round(paired["age"].corr(paired["score"]), 10))


def test_insufficient_correlation_observations_is_unavailable():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, 2.0, np.nan]})
    pair = analyze_bivariate(df).numeric_correlations[0]
    assert pair.n_observations == 0
    assert pair.correlation is None


def test_zero_variance_correlation_is_unavailable():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0, 4.0]})
    pair = analyze_bivariate(df).numeric_correlations[0]
    assert pair.n_observations == 4
    assert pair.correlation is None


# --- 12-13: categorical bivariate --------------------------------


def test_categorical_numeric_grouped_summary(messy_df):
    biv = analyze_bivariate(messy_df)
    summary = next(
        s
        for s in biv.categorical_numeric
        if s.categorical_column == "grade" and s.numeric_column == "score"
    )
    assert [g.category for g in summary.groups] == ["A", "B", "C"]
    a_group = next(g for g in summary.groups if g.category == "A")
    a_scores = messy_df.loc[messy_df["grade"] == "A", "score"]
    assert a_group.count == len(a_scores)
    assert a_group.mean == pytest.approx(a_scores.mean())
    assert a_group.median == pytest.approx(a_scores.median())


def test_categorical_categorical_contingency_summary():
    df = pd.DataFrame({"a": ["x", "x", "y", "y", "x"], "b": ["p", "q", "p", "p", "p"]})
    contingency = analyze_bivariate(df).categorical_categorical[0]
    rows = [(r.category_a, r.category_b, r.count) for r in contingency.rows]
    assert rows == [("x", "p", 2), ("x", "q", 1), ("y", "p", 2)]


# --- 14: determinism ----------------------------------------------


def test_deterministic_ordering_and_output(messy_df):
    r1 = analyze_dataframe(messy_df.copy(), dataset_id="ds-x")
    r2 = analyze_dataframe(messy_df.copy(), dataset_id="ds-x")
    d1 = r1.model_dump(mode="json")
    d2 = r2.model_dump(mode="json")
    d1.pop("generated_at")
    d2.pop("generated_at")
    assert d1 == d2
    assert r1.column_names == list(messy_df.columns)


# --- 15: read-only ----------------------------------------------


def test_dataframe_remains_unchanged(messy_df):
    before = messy_df.copy(deep=True)
    analyze_dataframe(messy_df, dataset_id="ds-x")
    analyze_univariate(messy_df)
    analyze_bivariate(messy_df)
    pd.testing.assert_frame_equal(messy_df, before)


# --- 16: JSON round-trip --------------------------------------


def test_eda_models_json_round_trip(messy_df):
    report = analyze_dataframe(messy_df, dataset_id="ds-x")
    restored = EDAReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert restored.model_dump(mode="json") == report.model_dump(mode="json")


# --- 17-19: version-aware EDA -------------------------------


def test_version_aware_eda_on_registered_raw_version(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    report = analyze_dataset_version(raw_v)
    assert report.dataset_id == raw_v.dataset_id
    assert report.dataset_version_id == raw_v.dataset_version_id
    assert report.n_rows == len(p.df)


def test_version_aware_eda_on_registered_processed_version(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    report = analyze_dataset_version(proc_v)
    assert report.dataset_version_id == proc_v.dataset_version_id
    assert report.n_rows == len(p.cleaned)


def test_missing_dataset_file_is_surfaced_clearly(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    proc_v = p.version_store.register_from_execution(
        p.report, parent_version_id=raw_v.dataset_version_id, cleaned_df=p.cleaned
    )
    path = proc_v.path
    path.chmod(0o644)
    path.unlink()
    with pytest.raises(VersionIntegrityError, match="missing"):
        analyze_dataset_version(proc_v)


def test_version_aware_eda_does_not_register_a_new_version(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    before = {v.dataset_version_id for v in p.version_store.list_versions(p.reference.dataset_id)}
    analyze_dataset_version(raw_v)
    after = {v.dataset_version_id for v in p.version_store.list_versions(p.reference.dataset_id)}
    assert before == after
    assert p.reference.raw_path.read_bytes()  # raw still present/readable


# --- 20: existing behaviour unchanged (smoke) ---------------


def test_eda_does_not_touch_the_registered_version_record(lineage_pipeline):
    p = lineage_pipeline
    raw_v = p.version_store.register_raw(p.reference, p.df)
    record = p.version_store.version_file_path(raw_v.dataset_version_id)
    before = record.read_bytes()
    raw_bytes = p.reference.raw_path.read_bytes()
    analyze_dataset_version(raw_v)
    assert record.read_bytes() == before
    assert p.reference.raw_path.read_bytes() == raw_bytes
