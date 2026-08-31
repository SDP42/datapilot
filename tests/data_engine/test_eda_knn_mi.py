"""Phase 4 — deterministic k-NN / Kraskov mutual-information estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine import eda
from data_engine.eda import (
    KNN_MI_DEFAULT_K,
    KNNMutualInformationResult,
    KNNMutualInformationStatus,
    analyze_dataframe,
    analyze_effect_sizes,
    analyze_visualizations,
    estimate_mutual_information_knn,
    recommend_visualizations,
)

# --- deterministic synthetic data (fixed seed, mathematical sequences) ---

_N = 400
_X = np.linspace(-3.0, 3.0, _N)
_RNG = np.random.default_rng(20240101)
_NOISE = _RNG.normal(0.0, 0.05, _N)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": _X,
            "y_linear": _X + _NOISE,  # strong linear dependency
            "y_square": _X**2 + _NOISE,  # strong nonlinear dependency
            "y_indep": np.tile([0.3, -0.7, 1.1, -0.2], _N // 4) + _NOISE,  # ~ independent
            "const": np.full(_N, 7.0),
            "grade": (["a", "b", "c", "d"] * (_N // 4)),
            "when": pd.to_datetime(["2021-01-01"] * _N),
        }
    )


def _mi(result: KNNMutualInformationResult) -> float:
    assert result.status is KNNMutualInformationStatus.COMPLETED
    assert result.mutual_information is not None
    return result.mutual_information


# --- API / model -------------------------------------------------


def test_public_symbols_importable():
    assert eda.estimate_mutual_information_knn is estimate_mutual_information_knn
    assert issubclass(eda.KNNMutualInformationResult, __import__("pydantic").BaseModel)
    assert set(eda.KNNMutualInformationStatus) == {
        KNNMutualInformationStatus.COMPLETED,
        KNNMutualInformationStatus.UNAVAILABLE,
    }


def test_return_type(df):
    assert isinstance(
        estimate_mutual_information_knn(df, "x", "y_linear"), KNNMutualInformationResult
    )


def test_json_round_trip(df):
    result = estimate_mutual_information_knn(df, "x", "y_square", k=4)
    dumped = result.model_dump_json()
    assert KNNMutualInformationResult.model_validate_json(dumped) == result


def test_model_holds_only_json_primitives(df):
    payload = estimate_mutual_information_knn(df, "x", "y_linear").model_dump(mode="json")
    for value in payload.values():
        assert value is None or isinstance(value, (str, int, float, bool, list))


def test_metadata_identifies_the_kraskov_estimator(df):
    result = estimate_mutual_information_knn(df, "x", "y_linear")
    assert result.estimator == "kraskov_knn"
    assert result.distance_metric == "chebyshev"
    assert result.k == KNN_MI_DEFAULT_K == 3
    assert result.finite_pair_filtering is True
    assert "nextafter" in result.tie_handling


# --- validation / unavailable ----------------------------------


def test_missing_x_column(df):
    r = estimate_mutual_information_knn(df, "nope", "x")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "not in the DataFrame" in r.reason


def test_missing_y_column(df):
    assert estimate_mutual_information_knn(df, "x", "nope").status is (
        KNNMutualInformationStatus.UNAVAILABLE
    )


def test_same_column_rejected(df):
    r = estimate_mutual_information_knn(df, "x", "x")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "same column" in r.reason


def test_datetime_column_rejected(df):
    r = estimate_mutual_information_knn(df, "x", "when")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "datetime" in r.reason


def test_categorical_column_rejected(df):
    r = estimate_mutual_information_knn(df, "x", "grade")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "not numeric" in r.reason


def test_constant_column_rejected(df):
    r = estimate_mutual_information_knn(df, "x", "const")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "constant" in r.reason


def test_all_invalid_input_unavailable():
    frame = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    r = estimate_mutual_information_knn(frame, "a", "b")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE


def test_insufficient_observations_unavailable():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 4.0]})
    r = estimate_mutual_information_knn(frame, "a", "b", k=3)
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "at least" in r.reason


@pytest.mark.parametrize("bad_k", [True, 1.5, 0, -2, "3"])
def test_invalid_k_unavailable(df, bad_k):
    r = estimate_mutual_information_knn(df, "x", "y_linear", k=bad_k)  # type: ignore[arg-type]
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "k must" in r.reason


def test_k_at_or_above_n_unavailable():
    frame = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(8.0) ** 1.3})
    r = estimate_mutual_information_knn(frame, "a", "b", k=8)
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE


# --- missing / non-finite filtering --------------------------


def test_nan_and_inf_are_excluded(df):
    frame = df.copy()
    frame.loc[0, "y_linear"] = np.nan
    frame.loc[1, "x"] = np.inf
    frame.loc[2, "y_linear"] = -np.inf
    result = estimate_mutual_information_knn(frame, "x", "y_linear")
    assert result.status is KNNMutualInformationStatus.COMPLETED
    assert result.n_observations == _N - 3


def test_observation_count_equals_paired_finite_rows(df):
    result = estimate_mutual_information_knn(df, "x", "y_square")
    assert result.n_observations == _N


# --- numerical correctness / sanity ------------------------


def test_valid_mi_is_finite_and_non_negative(df):
    for y in ("y_linear", "y_square", "y_indep"):
        value = _mi(estimate_mutual_information_knn(df, "x", y))
        assert np.isfinite(value)
        assert value >= 0.0


def test_strong_dependency_much_larger_than_independent(df):
    strong = _mi(estimate_mutual_information_knn(df, "x", "y_linear"))
    weak = _mi(estimate_mutual_information_knn(df, "x", "y_indep"))
    assert strong > weak + 1.0


def test_nonlinear_dependency_detected(df):
    nonlinear = _mi(estimate_mutual_information_knn(df, "x", "y_square"))
    weak = _mi(estimate_mutual_information_knn(df, "x", "y_indep"))
    # Pearson r for x vs x**2 over a symmetric range is ~0; MI still sees it.
    assert abs(np.corrcoef(_X, _X**2)[0, 1]) < 0.05
    assert nonlinear > weak + 1.0


def test_k_variation_is_deterministic_not_random(df):
    values = {
        k: estimate_mutual_information_knn(df, "x", "y_linear", k=k).mutual_information
        for k in (1, 2, 3, 5, 10)
    }
    again = {
        k: estimate_mutual_information_knn(df, "x", "y_linear", k=k).mutual_information
        for k in (1, 2, 3, 5, 10)
    }
    assert values == again
    assert all(v is not None and np.isfinite(v) for v in values.values())


def test_tiny_negative_estimate_is_clamped_and_noted():
    # near-independent, tie-heavy data can push KSG-1 slightly negative
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"a": rng.integers(0, 5, 200).astype(float), "b": rng.integers(0, 5, 200).astype(float)}
    )
    result = estimate_mutual_information_knn(frame, "a", "b")
    assert result.status is KNNMutualInformationStatus.COMPLETED
    assert result.mutual_information is not None
    assert result.mutual_information >= 0.0


def test_genuine_zero_is_distinct_from_unavailable(df):
    result = estimate_mutual_information_knn(df, "x", "y_indep")
    # whatever the value, it is a completed result with a real number, not None
    assert result.status is KNNMutualInformationStatus.COMPLETED
    assert result.mutual_information is not None


# --- determinism ------------------------------------------


def test_repeated_calls_identical_dump(df):
    dumps = {
        estimate_mutual_information_knn(df, "x", "y_square").model_dump_json() for _ in range(5)
    }
    assert len(dumps) == 1


def test_row_shuffle_identical_result(df):
    base = estimate_mutual_information_knn(df, "x", "y_linear")
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    assert (
        estimate_mutual_information_knn(shuffled, "x", "y_linear").model_dump() == base.model_dump()
    )


def test_tie_heavy_data_repeatable():
    frame = pd.DataFrame({"a": np.repeat(np.arange(15.0), 20), "b": np.repeat(np.arange(15.0), 20)})
    a = estimate_mutual_information_knn(frame, "a", "b")
    b = estimate_mutual_information_knn(
        frame.sample(frac=1.0, random_state=1).reset_index(drop=True), "a", "b"
    )
    assert a.model_dump() == b.model_dump()


# --- separation from the existing binned MI --------------


def test_distinct_from_binned_mutual_information(df):
    knn = _mi(estimate_mutual_information_knn(df, "x", "y_linear"))
    binned = next(
        e
        for e in analyze_effect_sizes(df[["x", "y_linear"]]).mutual_information
        if set(e.columns) == {"x", "y_linear"}
    )
    assert binned.effect_size is not None
    # different estimators — not expected to be numerically identical
    assert abs(knn - binned.effect_size) > 1e-6


def test_estimator_not_labelled_mutual_information(df):
    result = estimate_mutual_information_knn(df, "x", "y_linear")
    assert result.estimator != "mutual_information"
    assert result.estimator == "kraskov_knn"


# --- safety -------------------------------------------


def test_input_dataframe_unchanged(df):
    before = df.copy(deep=True)
    estimate_mutual_information_knn(df, "x", "y_linear")
    estimate_mutual_information_knn(df, "x", "y_square", k=5)
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    estimate_mutual_information_knn(df, "x", "y_linear")
    assert list(tmp_path.iterdir()) == []


def test_no_existing_analysis_is_modified(df):
    before_effects = analyze_effect_sizes(df[["x", "y_linear", "y_square"]]).model_dump()
    before_viz = analyze_visualizations(df).model_dump()
    estimate_mutual_information_knn(df, "x", "y_linear")
    assert analyze_effect_sizes(df[["x", "y_linear", "y_square"]]).model_dump() == before_effects
    assert analyze_visualizations(df).model_dump() == before_viz


def test_no_figure_is_created(df, monkeypatch):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    estimate_mutual_information_knn(df, "x", "y_linear")
    assert plt.get_fignums() == before


# --- integration / backward compat -------------------


def test_analyze_dataframe_signature_unchanged():
    import inspect

    assert list(inspect.signature(analyze_dataframe).parameters) == [
        "df",
        "dataset_id",
        "dataset_version_id",
    ]


def test_no_new_eda_report_field():
    frame = pd.DataFrame({"x": _X, "y": _X + _NOISE})
    report = analyze_dataframe(frame)
    assert not hasattr(report, "knn_mutual_information")
    assert not hasattr(report, "mutual_information_knn")


def test_existing_eda_sections_still_populated(df):
    report = analyze_dataframe(df[["x", "y_linear", "y_square", "grade"]])
    assert report.univariate.numeric
    assert report.bivariate.numeric_correlations
    assert report.effect_sizes.mutual_information
    assert report.nonparametric_tests.spearman
    assert report.distribution.columns
    assert report.visualizations.scatter_plots


def test_visualization_recommendation_and_strength_unchanged(df):
    frame = df[["x", "y_linear", "grade"]]
    rec_before = recommend_visualizations(frame, "y_linear").model_dump()
    estimate_mutual_information_knn(frame, "x", "y_linear")
    assert recommend_visualizations(frame, "y_linear").model_dump() == rec_before
