"""Phase 4 — datetime mutual information (KSG estimator over epoch seconds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine import eda
from data_engine.eda import (
    KNN_MI_REPRESENTATION_DATETIME,
    KNNMutualInformationResult,
    KNNMutualInformationStatus,
    analyze_effect_sizes,
    estimate_mutual_information_datetime,
    estimate_mutual_information_knn,
)

_N = 360
_START = pd.Timestamp("2021-03-01 00:00:00")
_TIMES = pd.to_datetime([_START + pd.Timedelta(hours=i) for i in range(_N)])
_RNG = np.random.default_rng(11)
_NOISE = _RNG.normal(0.0, 0.5, _N)


@pytest.fixture
def df() -> pd.DataFrame:
    hours = np.arange(_N, dtype=float)
    return pd.DataFrame(
        {
            "t": _TIMES,
            "t_tz": _TIMES.tz_localize("Europe/Berlin"),
            "t_shifted": _TIMES + pd.Timedelta(days=10),  # perfectly dependent on t
            "y_dep": hours + _NOISE,  # monotone with time
            "y_cycle": np.sin(hours / 12.0) + 0.05 * _NOISE,  # nonlinear temporal
            "y_indep": np.tile([1.0, -2.0, 0.5, 3.0, -1.0, 2.0], _N // 6) + _NOISE,
            "const_num": np.full(_N, 4.0),
            "grade": (["a", "b", "c"] * (_N // 3)),
        }
    )


def _mi(result: KNNMutualInformationResult) -> float:
    assert result.status is KNNMutualInformationStatus.COMPLETED
    assert result.mutual_information is not None
    return result.mutual_information


# --- API / model ------------------------------------------------


def test_importable():
    assert eda.estimate_mutual_information_datetime is estimate_mutual_information_datetime


def test_return_type_and_metadata(df):
    r = estimate_mutual_information_datetime(df, "t", "y_dep")
    assert isinstance(r, KNNMutualInformationResult)
    assert (
        r.representation == KNN_MI_REPRESENTATION_DATETIME == "elapsed_seconds_since_unix_epoch_utc"
    )
    assert r.estimator == "kraskov_knn"
    assert r.x_column == "t" and r.y_column == "y_dep"
    assert r.k == 3


def test_json_round_trip(df):
    r = estimate_mutual_information_datetime(df, "t", "t_shifted", k=4)
    assert KNNMutualInformationResult.model_validate_json(r.model_dump_json()) == r


# --- sanity -----------------------------------------------------


def test_datetime_numeric_dependency_is_finite_and_positive(df):
    value = _mi(estimate_mutual_information_datetime(df, "t", "y_dep"))
    assert np.isfinite(value)
    assert value > 0.5


def test_datetime_datetime_dependency_is_finite(df):
    value = _mi(estimate_mutual_information_datetime(df, "t", "t_shifted"))
    assert np.isfinite(value)
    assert value > 1.0  # t_shifted = t + const → strong dependence


def test_independent_much_weaker_than_dependent(df):
    strong = _mi(estimate_mutual_information_datetime(df, "t", "y_dep"))
    weak = _mi(estimate_mutual_information_datetime(df, "t", "y_indep"))
    assert strong > weak + 0.5


def test_nonlinear_temporal_relationship_detected(df):
    cyclic = _mi(estimate_mutual_information_datetime(df, "t", "y_cycle"))
    weak = _mi(estimate_mutual_information_datetime(df, "t", "y_indep"))
    assert cyclic > weak


# --- validation / unavailable --------------------------------


def test_nat_is_filtered(df):
    frame = df.copy()
    frame.loc[0, "t"] = pd.NaT
    frame.loc[1, "t"] = pd.NaT
    r = estimate_mutual_information_datetime(frame, "t", "y_dep")
    assert r.status is KNNMutualInformationStatus.COMPLETED
    assert r.n_observations == _N - 2


def test_numeric_nan_and_inf_filtered(df):
    frame = df.copy()
    frame.loc[0, "y_dep"] = np.nan
    frame.loc[1, "y_dep"] = np.inf
    r = estimate_mutual_information_datetime(frame, "t", "y_dep")
    assert r.n_observations == _N - 2


def test_constant_datetime_unavailable(df):
    frame = df.assign(t_flat=pd.to_datetime(["2021-01-01"] * _N))
    r = estimate_mutual_information_datetime(frame, "t_flat", "y_dep")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "constant" in r.reason


def test_constant_numeric_unavailable(df):
    r = estimate_mutual_information_datetime(df, "t", "const_num")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "constant" in r.reason


def test_all_missing_datetime_unavailable():
    frame = pd.DataFrame({"t": pd.to_datetime([pd.NaT] * 10), "y": np.arange(10.0)})
    r = estimate_mutual_information_datetime(frame, "t", "y")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE


def test_insufficient_observations_unavailable():
    frame = pd.DataFrame(
        {"t": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]), "y": [1.0, 2.0, 3.0]}
    )
    r = estimate_mutual_information_datetime(frame, "t", "y", k=3)
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "at least" in r.reason


@pytest.mark.parametrize("bad_k", [True, 2.5, 0, -1])
def test_invalid_k_unavailable(df, bad_k):
    r = estimate_mutual_information_datetime(df, "t", "y_dep", k=bad_k)  # type: ignore[arg-type]
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "k must" in r.reason


def test_missing_columns_unavailable(df):
    assert estimate_mutual_information_datetime(df, "nope", "y_dep").status is (
        KNNMutualInformationStatus.UNAVAILABLE
    )
    assert estimate_mutual_information_datetime(df, "t", "nope").status is (
        KNNMutualInformationStatus.UNAVAILABLE
    )


def test_same_column_rejected(df):
    r = estimate_mutual_information_datetime(df, "t", "t")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "same column" in r.reason


def test_non_datetime_first_column_rejected(df):
    r = estimate_mutual_information_datetime(df, "y_dep", "y_indep")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "not a datetime column" in r.reason


def test_categorical_partner_rejected_with_reason(df):
    r = estimate_mutual_information_datetime(df, "t", "grade")
    assert r.status is KNNMutualInformationStatus.UNAVAILABLE
    assert "categorical" in r.reason
    assert "binned mutual_information" in r.reason


# --- timezone / determinism -------------------------------


def test_timezone_aware_input_is_handled_deterministically(df):
    naive = estimate_mutual_information_datetime(df, "t", "y_dep")
    aware = estimate_mutual_information_datetime(df, "t_tz", "y_dep")
    # 't' (naive, read as UTC) and 't_tz' (Berlin) differ by a fixed offset →
    # the same affine shift → after standardization the MI is identical.
    assert naive.mutual_information == aware.mutual_information


def test_repeated_calls_identical(df):
    dumps = {
        estimate_mutual_information_datetime(df, "t", "y_dep").model_dump_json() for _ in range(5)
    }
    assert len(dumps) == 1


def test_row_shuffle_invariance(df):
    base = estimate_mutual_information_datetime(df, "t", "y_dep")
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert (
        estimate_mutual_information_datetime(shuffled, "t", "y_dep").model_dump()
        == base.model_dump()
    )


# --- safety --------------------------------------------


def test_input_dataframe_unchanged(df):
    before = df.copy(deep=True)
    estimate_mutual_information_datetime(df, "t", "y_dep")
    estimate_mutual_information_datetime(df, "t", "t_shifted", k=5)
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    estimate_mutual_information_datetime(df, "t", "y_dep")
    assert list(tmp_path.iterdir()) == []


def test_no_figure_created(df):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    estimate_mutual_information_datetime(df, "t", "y_dep")
    assert plt.get_fignums() == before


def test_existing_numeric_knn_mi_unchanged(df):
    frame = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) ** 1.4})
    r = estimate_mutual_information_knn(frame, "a", "b")
    assert r.status is KNNMutualInformationStatus.COMPLETED
    assert r.representation == "raw_numeric_values"
    assert r.mutual_information is not None


def test_existing_binned_mi_unchanged(df):
    frame = df[["y_dep", "y_indep"]]
    before = analyze_effect_sizes(frame).model_dump()
    estimate_mutual_information_datetime(df, "t", "y_dep")
    assert analyze_effect_sizes(frame).model_dump() == before
