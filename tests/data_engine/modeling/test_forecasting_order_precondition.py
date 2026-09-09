"""Post-Phase-7 stabilization (audit H1) — forecasting chronological-order guard.

For ``time_ordered_holdout`` the row order *is* the time axis. Phase 7.4
never infers a time column or sorts; it verifies that the frame is
non-decreasing on one of its own datetime columns and returns an explicit
``unavailable`` otherwise. It must not silently pretend arbitrary row
order is chronological.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.feature_engineering import (
    FeatureEngineeringRequest,
    assess_feature_engineering,
    inventory_features,
    recommend_feature_selection,
    recommend_preprocessing,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.modeling import (
    DataSplitStrategy,
    ModelingRequest,
    ModelingStatus,
    assess_model_readiness,
    generate_model_candidates,
    recommend_data_split,
    run_modeling_pipeline,
    train_and_evaluate_models,
    understand_modeling,
)
from data_engine.problem_understanding import (
    ProblemUnderstandingRequest,
    TaskType,
    assess_feasibility,
    identify_target,
    infer_task_type,
    recommend_metrics,
    understand_problem,
)

COMPLETED = ModelingStatus.COMPLETED
UNAVAILABLE = ModelingStatus.UNAVAILABLE
_N = 260
_OBJECTIVE = "forecast future demand over time"


def _forecasting_frame(order: str = "sorted") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=_N, freq="D")
    demand = np.linspace(0.0, 40.0, _N) + rng.normal(0.0, 2.0, _N)
    df = pd.DataFrame({"day": dates, "exog": rng.normal(0.0, 1.0, _N), "demand": demand})
    if order == "shuffled":
        df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    elif order == "reversed":
        df = df.iloc[::-1].reset_index(drop=True)
    elif order == "no_datetime":
        df = df.drop(columns=["day"])
    return df


def _build(df: pd.DataFrame):
    t = identify_target(df, objective=_OBJECTIVE)
    task = infer_task_type(df, t, objective=_OBJECTIVE)
    if task.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting for this frame")
    m = recommend_metrics(df, task, objective=_OBJECTIVE)
    feas = assess_feasibility(df, t, task, m)
    problem = understand_problem(
        ProblemUnderstandingRequest(dataset_id="d", objective=_OBJECTIVE)
    ).model_copy(update={"target": t, "task_type": task, "metrics": m, "feasibility": feas})

    inv = inventory_features(df, target=t.target_column)
    tr = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, task)
    pp = recommend_preprocessing(df, inv, tr, sel)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp)
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d")).model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "assessment": asmt,
        }
    )
    readiness = assess_model_readiness(df, problem, fe)
    split = recommend_data_split(df, problem, fe)
    candidates = generate_model_candidates(df, problem, fe, readiness, split)
    return df, problem, fe, readiness, split, candidates


def _train(df):
    df, problem, fe, readiness, split, candidates = _build(df)
    assert split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    return train_and_evaluate_models(df, problem, fe, readiness, split, candidates)


# --- 1. correctly ordered input succeeds -------------------------


def test_sorted_forecasting_frame_trains():
    out = _train(_forecasting_frame("sorted"))
    assert out.status is COMPLETED
    assert out.successful_runs
    assert any("chronological-order precondition satisfied" in n for n in out.notes)
    assert any("row order is the time axis" in n for n in out.notes)


# --- 2. invalid / unverifiable ordering -> explicit unavailable --


def test_shuffled_forecasting_frame_is_unavailable():
    out = _train(_forecasting_frame("shuffled"))
    assert out.status is UNAVAILABLE
    assert out.runs == []
    assert "chronologically" in (out.reason or "")
    assert "does not reorder rows" in (out.reason or "")
    assert "non-decreasing" in (out.reason or "")


def test_reversed_forecasting_frame_is_unavailable():
    out = _train(_forecasting_frame("reversed"))
    assert out.status is UNAVAILABLE
    assert "non-decreasing" in (out.reason or "")


def test_forecasting_frame_without_datetime_column_is_unavailable():
    # Build a valid forecasting pipeline (needs a datetime column for the
    # Phase-5 task inference), then drop the datetime column before Phase
    # 7.4 so the chronological-order guard cannot verify anything.
    df, problem, fe, readiness, split, candidates = _build(_forecasting_frame("sorted"))
    assert split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    df_no_dt = df.drop(columns=["day"])
    out = train_and_evaluate_models(df_no_dt, problem, fe, readiness, split, candidates)
    assert out.status is UNAVAILABLE
    assert "no datetime column" in (out.reason or "")


# --- 3. deterministic repeated behavior --------------------------


def test_deterministic_sorted():
    df = _forecasting_frame("sorted")
    built = _build(df)
    a = train_and_evaluate_models(*built)
    b = train_and_evaluate_models(*built)
    assert a.model_dump_json() == b.model_dump_json()


def test_deterministic_shuffled_unavailable():
    df = _forecasting_frame("shuffled")
    built = _build(df)
    a = train_and_evaluate_models(*built)
    b = train_and_evaluate_models(*built)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.status is UNAVAILABLE


# --- 4. the non-temporal contract is unchanged -----------------


def test_random_split_still_row_order_invariant():
    rng = np.random.default_rng(9)
    s = rng.uniform(0.0, 10.0, _N)
    df = pd.DataFrame(
        {"s": s, "t": rng.normal(0.0, 1.0, _N), "price": 3.0 * s + rng.normal(0.0, 1.0, _N)}
    )
    t = identify_target(df, objective="predict the price")
    task = infer_task_type(df, t, objective="predict the price")
    m = recommend_metrics(df, task)
    feas = assess_feasibility(df, t, task, m)
    problem = understand_problem(ProblemUnderstandingRequest(dataset_id="d")).model_copy(
        update={"target": t, "task_type": task, "metrics": m, "feasibility": feas}
    )
    inv = inventory_features(df, target=t.target_column)
    tr = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, task)
    pp = recommend_preprocessing(df, inv, tr, sel)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp)
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d")).model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "assessment": asmt,
        }
    )
    readiness = assess_model_readiness(df, problem, fe)
    split = recommend_data_split(df, problem, fe)
    assert split.strategy is DataSplitStrategy.RANDOM_HOLDOUT
    candidates = generate_model_candidates(df, problem, fe, readiness, split)
    base = train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    shuffled = df.sample(frac=1.0, random_state=21).reset_index(drop=True)
    other = train_and_evaluate_models(shuffled, problem, fe, readiness, split, candidates)
    assert base == other


# --- pipeline-level view --------------------------------------


def test_pipeline_forecasting_sorted_completes():
    spec = run_modeling_pipeline(
        _forecasting_frame("sorted"), ModelingRequest(dataset_id="f", objective=_OBJECTIVE)
    )
    assert spec.status is ModelingStatus.COMPLETED
    assert spec.split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    assert spec.selection.selected_family is not None


def test_pipeline_forecasting_shuffled_unavailable():
    spec = run_modeling_pipeline(
        _forecasting_frame("shuffled"), ModelingRequest(dataset_id="f", objective=_OBJECTIVE)
    )
    assert spec.status is ModelingStatus.UNAVAILABLE
    assert spec.training.status is ModelingStatus.UNAVAILABLE
    assert spec.selection.selected_family is None
    assert "model training" in (spec.reason or "")


def test_understand_modeling_unaffected():
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective=_OBJECTIVE))
    assert spec.status is ModelingStatus.NOT_YET_INFERRED
