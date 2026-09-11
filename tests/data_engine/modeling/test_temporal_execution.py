"""Forecasting Execution — `build_temporal_features`.

Deterministic, backward-looking lag / rolling feature construction. lag(k) = shift(k),
rolling(w) = shift(1).rolling(w) — every built feature at row i uses only values
strictly before i, so it is leakage-safe for one-step-ahead evaluation regardless of
where the train/test split falls. Never mutates the frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine import modeling
from data_engine.feature_engineering import (
    FeatureOperationType,
    TemporalFeatureRecommendation,
    TransformationRecommendation,
)
from data_engine.modeling import (
    build_calendar_features,
    build_temporal_features,
    temporal_feature_spec,
)


def _rec(column: str, description: str) -> TemporalFeatureRecommendation:
    op = (
        FeatureOperationType.LAG_FEATURE
        if description.startswith("lag")
        else FeatureOperationType.ROLLING_FEATURE
    )
    return TemporalFeatureRecommendation(
        column=column, operation=op, description=description, reason="x"
    )


def _frame(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ds": pd.date_range("2022-01-01", periods=n, freq="D"),
            "demand": np.arange(n, dtype=float) * 2.0,
            "exog": np.arange(n, dtype=float),
        }
    )


# --- API ---------------------------------------------------------


def test_exported():
    assert modeling.build_temporal_features is build_temporal_features
    assert "build_temporal_features" in modeling.__all__
    assert "MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS" in modeling.__all__


# --- lag semantics ---------------------------------------------


def test_lag_is_shift_k():
    df = _frame(20)
    out, built, _ = build_temporal_features(df, [_rec("demand", "lag 3")])
    assert built == ["demand__lag_3"]
    expected = df["demand"].shift(3)
    pd.testing.assert_series_equal(out["demand__lag_3"], expected, check_names=False)
    assert out["demand__lag_3"].isna().sum() == 3  # leading warm-up


# --- rolling semantics (shift(1) mandatory) -------------------


def test_rolling_mean_uses_only_the_past():
    df = _frame(20)
    out, built, _ = build_temporal_features(df, [_rec("demand", "rolling mean window 4")])
    assert built == ["demand__rollmean_4"]
    expected = df["demand"].shift(1).rolling(window=4, min_periods=4).mean()
    pd.testing.assert_series_equal(out["demand__rollmean_4"], expected, check_names=False)
    assert out["demand__rollmean_4"].isna().sum() == 4  # shift(1) + window 4 warm-up


def test_rolling_std_column_name_and_warmup():
    out, built, _ = build_temporal_features(_frame(30), [_rec("demand", "rolling std window 7")])
    assert built == ["demand__rollstd_7"]
    assert out["demand__rollstd_7"].isna().sum() == 7


# --- leakage: a late spike never changes an earlier feature ----


def test_future_spike_does_not_leak_backward():
    df = _frame(40)
    spike = df.copy()
    spike.loc[30, "demand"] = 10_000.0
    recs = [
        _rec("demand", "lag 1"),
        _rec("demand", "lag 7"),
        _rec("demand", "rolling mean window 7"),
        _rec("demand", "rolling std window 7"),
    ]
    base_out, base_built, _ = build_temporal_features(df, recs)
    spike_out, _, _ = build_temporal_features(spike, recs)
    # every built feature at rows strictly before 30 must be identical
    for col in base_built:
        pd.testing.assert_series_equal(
            base_out[col].iloc[:30], spike_out[col].iloc[:30], check_names=False
        )


# --- exogenous, ordering, multiple recs ----------------------


def test_multiple_recs_and_deterministic_order():
    df = _frame(60)
    recs = [
        _rec("demand", "lag 1"),
        _rec("demand", "lag 14"),
        _rec("demand", "rolling mean window 7"),
        _rec("exog", "lag 1"),
    ]
    out, built, _ = build_temporal_features(df, recs)
    assert built == [
        "demand__lag_1",
        "demand__lag_14",
        "demand__rollmean_7",
        "exog__lag_1",
    ]
    assert list(out.columns[: len(df.columns)]) == list(df.columns)


# --- safety / errors -----------------------------------------


def test_df_not_mutated_and_deterministic():
    df = _frame(50)
    before = df.copy(deep=True)
    recs = [_rec("demand", "lag 7"), _rec("demand", "rolling std window 30")]
    a, _, _ = build_temporal_features(df, recs)
    b, _, _ = build_temporal_features(df, recs)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(a, b)


def test_missing_source_column_is_skipped_with_a_note():
    _, built, notes = build_temporal_features(_frame(20), [_rec("ghost", "lag 1")])
    assert built == []
    assert any("'ghost' skipped: not a column of the frame" in n for n in notes)


def test_collision_with_existing_column_raises():
    df = _frame(20)
    df["demand__lag_1"] = 0.0
    with pytest.raises(ValueError, match="collides with an existing column"):
        build_temporal_features(df, [_rec("demand", "lag 1")])


def test_unrecognised_description_raises():
    with pytest.raises(ValueError, match="unrecognised temporal feature description"):
        build_temporal_features(_frame(20), [_rec("demand", "ewm span 5")])


def test_notes_state_the_leakage_property():
    _, _, notes = build_temporal_features(_frame(30), [_rec("demand", "lag 1")])
    assert any("leakage-safe for one-step-ahead evaluation" in n for n in notes)
    assert any("shift(1).rolling(w)" in n for n in notes)


def test_empty_recommendations_returns_copy():
    df = _frame(10)
    out, built, _ = build_temporal_features(df, [])
    assert built == []
    assert out is not df
    pd.testing.assert_frame_equal(out, df)


# --- temporal_feature_spec ------------------------------------


def test_temporal_feature_spec_maps_names_to_provenance():
    recs = [
        _rec("demand", "lag 7"),
        _rec("demand", "rolling mean window 30"),
        _rec("demand", "rolling std window 30"),
        _rec("exog", "lag 1"),
    ]
    spec = temporal_feature_spec(recs)
    assert spec["demand__lag_7"] == ("demand", "lag", 7)
    assert spec["demand__rollmean_30"] == ("demand", "rollmean", 30)
    assert spec["demand__rollstd_30"] == ("demand", "rollstd", 30)
    assert spec["exog__lag_1"] == ("exog", "lag", 1)


def test_temporal_feature_spec_keys_match_build_temporal_features_names():
    df = _frame(60)
    recs = [_rec("demand", "lag 7"), _rec("demand", "rolling mean window 7")]
    _, built, _ = build_temporal_features(df, recs)
    assert set(temporal_feature_spec(recs)) == set(built)


# --- build_calendar_features -----------------------------------


def _dt_rec(column: str, description: str) -> TransformationRecommendation:
    return TransformationRecommendation(
        column=column,
        operation=FeatureOperationType.DATETIME_DERIVATION,
        description=description,
        reason="x",
    )


def test_calendar_derive_parts():
    df = _frame(60)
    recs = [_dt_rec("ds", "derive month"), _dt_rec("ds", "derive day_of_week")]
    out, built, _ = build_calendar_features(df, recs)
    assert built == ["ds__month", "ds__day_of_week"]
    pd.testing.assert_series_equal(
        out["ds__month"], df["ds"].dt.month.astype("float64"), check_names=False
    )
    pd.testing.assert_series_equal(
        out["ds__day_of_week"], df["ds"].dt.dayofweek.astype("float64"), check_names=False
    )


def test_calendar_cyclical_bounded_and_periodic():
    df = _frame(400)  # > 1 year so month repeats
    out, built, _ = build_calendar_features(df, [_dt_rec("ds", "cyclical (sin/cos) month")])
    assert built == ["ds__month_sin", "ds__month_cos"]
    assert out["ds__month_sin"].between(-1.0, 1.0).all()
    assert out["ds__month_cos"].between(-1.0, 1.0).all()
    # same calendar month -> identical cyclical encoding (period = 12)
    jan_rows = df.index[df["ds"].dt.month == 1]
    assert out.loc[jan_rows, "ds__month_sin"].nunique() == 1
    assert out.loc[jan_rows, "ds__month_cos"].nunique() == 1


def test_calendar_features_are_stateless_no_warmup():
    df = _frame(30)
    out, built, _ = build_calendar_features(df, [_dt_rec("ds", "derive quarter")])
    assert out[built[0]].isna().sum() == 0  # no lookback -> no warm-up NaNs


def test_calendar_ignores_non_datetime_derivation_recs():
    df = _frame(20)
    recs = [
        TransformationRecommendation(
            column="exog",
            operation=FeatureOperationType.TRANSFORMATION,
            description="log transform",
            reason="x",
        )
    ]
    out, built, _ = build_calendar_features(df, recs)
    assert built == []
    pd.testing.assert_frame_equal(out, df)


def test_calendar_missing_column_skipped_with_note():
    _, built, notes = build_calendar_features(_frame(10), [_dt_rec("ghost", "derive month")])
    assert built == []
    assert any("'ghost' skipped: not a column of the frame" in n for n in notes)


def test_calendar_unrecognised_part_skipped_with_note():
    _, built, notes = build_calendar_features(_frame(10), [_dt_rec("ds", "derive fortnight")])
    assert built == []
    assert any("unrecognised calendar part" in n for n in notes)


def test_calendar_collision_raises():
    df = _frame(10)
    df["ds__month"] = 0.0
    with pytest.raises(ValueError, match="collides with an existing column"):
        build_calendar_features(df, [_dt_rec("ds", "derive month")])


def test_calendar_df_not_mutated_and_deterministic():
    df = _frame(40)
    before = df.copy(deep=True)
    recs = [_dt_rec("ds", "derive month"), _dt_rec("ds", "cyclical (sin/cos) day_of_week")]
    a, _, _ = build_calendar_features(df, recs)
    b, _, _ = build_calendar_features(df, recs)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(a, b)


def test_calendar_notes_state_the_stateless_property():
    _, _, notes = build_calendar_features(_frame(10), [_dt_rec("ds", "derive month")])
    assert any("stateless row-wise functions of the timestamp" in n for n in notes)
