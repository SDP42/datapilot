"""Forecasting foundation — `recommend_temporal_features` / `TemporalFeatureRecommendations`.

Deterministic, recommendation-only lag / rolling feature recommendations for a
time-series-forecasting problem. Never computes a lag, never touches the DataFrame,
``unavailable`` for every non-forecasting task.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_engine import feature_engineering as fe
from data_engine.feature_engineering import (
    FORECASTING_LAG_ORDERS,
    FORECASTING_MIN_ROWS_FOR_TEMPORAL,
    FORECASTING_ROLLING_WINDOWS,
    FeatureEngineeringStatus,
    FeatureOperationType,
    TemporalFeatureRecommendation,
    TemporalFeatureRecommendations,
    inventory_features,
    recommend_temporal_features,
)
from data_engine.problem_understanding import (
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED
_N = 200


def _frame(n: int = _N, *, extra_datetime: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "ds": pd.date_range("2022-01-01", periods=n, freq="D"),
        "exog_a": rng.normal(0.0, 1.0, n),
        "exog_b": rng.normal(5.0, 2.0, n),
        "region": (["n", "s"] * (n // 2 + 1))[:n],
        "demand": np.linspace(0.0, 50.0, n) + rng.normal(0.0, 3.0, n),
    }
    if extra_datetime:
        data["ship_ds"] = pd.date_range("2022-02-01", periods=n, freq="D")
    return pd.DataFrame(data)


def _forecasting_task(target="demand", time_col="ds") -> TaskTypeInference:
    return TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED,
        task_type=TaskType.TIME_SERIES_FORECASTING,
        target_column=target,
        time_column=time_col,
    )


def _inv(df: pd.DataFrame, target="demand"):
    return inventory_features(df, target=target)


# --- API / guards --------------------------------------------------


def test_exported():
    assert fe.recommend_temporal_features is recommend_temporal_features
    for name in (
        "recommend_temporal_features",
        "TemporalFeatureRecommendation",
        "TemporalFeatureRecommendations",
        "FORECASTING_LAG_ORDERS",
        "FORECASTING_ROLLING_WINDOWS",
        "FORECASTING_MIN_ROWS_FOR_TEMPORAL",
        "FORECASTING_TEMPORAL_ROW_MARGIN",
    ):
        assert name in fe.__all__


def test_type_guards():
    df = _frame()
    with pytest.raises(TypeError):
        recommend_temporal_features("nope", _inv(df), _forecasting_task())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recommend_temporal_features(df, "nope", _forecasting_task())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        recommend_temporal_features(df, _inv(df), "nope")  # type: ignore[arg-type]


# --- unavailable precedence -------------------------------------


def test_unavailable_task_not_completed():
    df = _frame()
    tt = TaskTypeInference(status=ProblemUnderstandingStatus.NOT_YET_INFERRED)
    r = recommend_temporal_features(df, _inv(df), tt)
    assert r.status is UNAVAILABLE and "task-type inference is not completed" in r.reason


def test_unavailable_non_forecasting_task():
    df = _frame()
    tt = TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED, task_type=TaskType.REGRESSION
    )
    r = recommend_temporal_features(df, _inv(df), tt)
    assert r.status is UNAVAILABLE
    assert "apply only to time_series_forecasting (task type = regression)" in r.reason


def test_unavailable_no_time_column():
    df = _frame()
    tt = TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED,
        task_type=TaskType.TIME_SERIES_FORECASTING,
        target_column="demand",
        time_column=None,
    )
    r = recommend_temporal_features(df, _inv(df), tt)
    assert r.status is UNAVAILABLE and "no time column was resolved" in r.reason


def test_unavailable_inventory_not_completed():
    df = _frame()
    bad_inv = _inv(df).model_copy(update={"status": UNAVAILABLE})
    r = recommend_temporal_features(df, bad_inv, _forecasting_task())
    assert r.status is UNAVAILABLE and "feature inventory is not completed" in r.reason


# --- completed: target autoregression --------------------------


def test_target_lag_and_rolling_recommendations():
    df = _frame()
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    assert r.status is COMPLETED
    assert r.time_column == "ds" and r.target_column == "demand"

    target_lags = {
        rec.description
        for rec in r.recommendations
        if rec.column == "demand" and rec.operation is FeatureOperationType.LAG_FEATURE
    }
    assert target_lags == {f"lag {o}" for o in FORECASTING_LAG_ORDERS}

    rolling = {
        rec.description
        for rec in r.recommendations
        if rec.column == "demand" and rec.operation is FeatureOperationType.ROLLING_FEATURE
    }
    assert rolling == {
        f"rolling {stat} window {w}"
        for w in FORECASTING_ROLLING_WINDOWS
        for stat in ("mean", "std")
    }


def test_lag_order_fits_the_row_count():
    df = _frame(n=60)  # >= FORECASTING_MIN_ROWS_FOR_TEMPORAL, but not > 14 + 10 margin? 60 > 24 yes
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    # window 30 needs n_rows > 40 -> present at n=60; window 7 present.
    assert any("rolling mean window 30" == rec.description for rec in r.recommendations)
    small = _frame(n=52)  # 52 > 30 + 10 = 40 -> window 30 present; 52 > 14 + 10 -> lag 14 present
    r2 = recommend_temporal_features(small, _inv(small), _forecasting_task())
    assert any(rec.description == "lag 14" for rec in r2.recommendations)


def test_row_count_gate():
    df = _frame(n=FORECASTING_MIN_ROWS_FOR_TEMPORAL - 1)
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    assert r.status is COMPLETED
    assert r.recommendations == []
    assert f"at least {FORECASTING_MIN_ROWS_FOR_TEMPORAL}" in r.reason


# --- exogenous features --------------------------------------


def test_exogenous_numeric_lag_1_by_default():
    df = _frame()
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    exo = {
        (rec.column, rec.description)
        for rec in r.recommendations
        if rec.column in ("exog_a", "exog_b")
    }
    assert exo == {("exog_a", "lag 1"), ("exog_b", "lag 1")}
    # categorical column and the time column are never given lag features
    assert not any(rec.column in ("region", "ds") for rec in r.recommendations)


def test_objective_widens_exogenous_lags():
    df = _frame()
    r = recommend_temporal_features(
        df, _inv(df), _forecasting_task(), objective="use autoregressive lag features"
    )
    exo_a = {rec.description for rec in r.recommendations if rec.column == "exog_a"}
    assert exo_a == {f"lag {o}" for o in FORECASTING_LAG_ORDERS}
    assert r.objective_used is True
    assert any("widened to the full lag set" in n for n in r.notes)


# --- datetime-target forecasting -----------------------------


def test_datetime_target_has_no_target_autoregression():
    df = _frame()
    tt = _forecasting_task(target="ds", time_col="ds")
    r = recommend_temporal_features(df, _inv(df, target="ds"), tt)
    assert r.status is COMPLETED
    assert not any(rec.column == "ds" for rec in r.recommendations)
    assert any("no target autoregression" in n for n in r.notes)
    # exogenous lags still produced
    assert any(rec.column in ("exog_a", "exog_b") for rec in r.recommendations)


# --- ordering / alignment / determinism --------------------


def test_deterministic_ordering():
    df = _frame()
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    cols = [rec.column for rec in r.recommendations]
    assert cols == sorted(cols)
    lags = [
        int(rec.description.split()[-1])
        for rec in r.recommendations
        if rec.column == "demand" and rec.operation is FeatureOperationType.LAG_FEATURE
    ]
    assert lags == sorted(lags)  # 'lag 2' before 'lag 14'


def test_recommended_operations_aligned():
    df = _frame()
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    assert r.recommended_operations == [
        f"{rec.column}: {rec.description}" for rec in r.recommendations
    ]


def test_byte_identical_repeated_and_no_mutation():
    df = _frame()
    before = df.copy(deep=True)
    a = recommend_temporal_features(df, _inv(df), _forecasting_task()).model_dump_json()
    b = recommend_temporal_features(df, _inv(df), _forecasting_task()).model_dump_json()
    assert a == b
    pd.testing.assert_frame_equal(df, before)


def test_notes_always_carry_recommendation_only_and_target_exemption():
    df = _frame()
    r = recommend_temporal_features(df, _inv(df), _forecasting_task())
    assert any("recommendation only" in n for n in r.notes)
    assert any("legitimate autoregressive predictors" in n for n in r.notes)


# --- bare contract / backward compatibility ----------------


def test_bare_model_is_all_null():
    m = TemporalFeatureRecommendations()
    assert m.status is NOT_YET
    assert m.reason is None and m.time_column is None and m.target_column is None
    assert m.recommendations == [] and m.recommended_operations == []
    assert m.objective_used is False and m.notes == []


def test_legacy_json_validates():
    legacy = json.dumps({"status": "not_yet_inferred", "reason": None, "notes": []})
    m = TemporalFeatureRecommendations.model_validate_json(legacy)
    assert m.status is NOT_YET and m.time_column is None


def test_recommendation_model_shape():
    rec = TemporalFeatureRecommendation(
        column="demand",
        operation=FeatureOperationType.LAG_FEATURE,
        description="lag 7",
        reason="x",
    )
    assert rec.evidence == []
