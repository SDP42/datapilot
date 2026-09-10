"""Forecasting chronological-order precondition (audit H1 + forecasting foundation).

For ``time_ordered_holdout`` the row order *is* the time axis. Since the
forecasting foundation the primary catch is **Phase 5 feasibility**
(``assess_feasibility`` checks the resolved ``time_column`` is non-decreasing),
which cascades to ``readiness.ready is False`` and an overall ``unavailable``.
Phase 7.4's ``_verify_chronological_order`` remains as **defense-in-depth** for a
caller who bypasses feasibility. Neither ever sorts or reorders the rows.
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
    recommend_temporal_features,
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
from data_engine.modeling.training import _verify_chronological_order
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
    return df


def _build(df: pd.DataFrame, *, time_column=None, run_feasibility: bool = True):
    t = identify_target(df, objective=_OBJECTIVE)
    task = infer_task_type(df, t, objective=_OBJECTIVE, time_column=time_column)
    if task.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting for this frame")
    m = recommend_metrics(df, task, objective=_OBJECTIVE)
    updates = {"target": t, "task_type": task, "metrics": m}
    if run_feasibility:
        updates["feasibility"] = assess_feasibility(df, t, task, m)
    problem = understand_problem(
        ProblemUnderstandingRequest(dataset_id="d", objective=_OBJECTIVE)
    ).model_copy(update=updates)

    inv = inventory_features(df, target=t.target_column)
    tr = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, task)
    pp = recommend_preprocessing(df, inv, tr, sel)
    tmp = recommend_temporal_features(df, inv, task)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp, temporal=tmp)
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d")).model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "temporal": tmp,
            "assessment": asmt,
        }
    )
    readiness = assess_model_readiness(df, problem, fe)
    split = recommend_data_split(df, problem, fe)
    candidates = generate_model_candidates(df, problem, fe, readiness, split)
    return df, problem, fe, readiness, split, candidates


# --- _verify_chronological_order (unit) --------------------------


def test_verify_declared_column_sorted_ok():
    df = _forecasting_frame("sorted")
    ok, detail = _verify_chronological_order(df, "day")
    assert ok is True
    assert "declared time column 'day' is non-decreasing" in detail


def test_verify_declared_column_missing():
    df = _forecasting_frame("sorted")
    ok, detail = _verify_chronological_order(df, "no_such")
    assert ok is False
    assert "not in the DataFrame" in detail


def test_verify_declared_column_shuffled():
    df = _forecasting_frame("shuffled")
    ok, detail = _verify_chronological_order(df, "day")
    assert ok is False
    assert "not in chronological order on the declared time column 'day'" in detail
    assert "does not reorder rows" in detail


def test_verify_heuristic_fallback_when_no_declared_column():
    assert _verify_chronological_order(_forecasting_frame("sorted"), None)[0] is True
    assert _verify_chronological_order(_forecasting_frame("shuffled"), None)[0] is False
    ok, detail = _verify_chronological_order(
        _forecasting_frame("sorted").drop(columns=["day"]), None
    )
    assert ok is False and "no datetime column" in detail


# --- primary catch: Phase 5 feasibility -> readiness -> unavailable ----


def test_sorted_forecasting_pipeline_trains():
    built = _build(_forecasting_frame("sorted"))
    assert built[4].strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    out = train_and_evaluate_models(*built)
    assert out.status is COMPLETED
    assert out.successful_runs
    assert any("chronological-order precondition satisfied" in n for n in out.notes)


@pytest.mark.parametrize("order", ["shuffled", "reversed"])
def test_unordered_forecasting_blocked_at_feasibility(order):
    df, problem, fe, readiness, split, candidates = _build(_forecasting_frame(order))
    assert problem.feasibility.feasible is False
    assert any("not in chronological order" in b for b in problem.feasibility.blocking_issues)
    assert readiness.ready is False
    out = train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    assert out.status is UNAVAILABLE
    assert out.runs == []


# --- defense-in-depth: Phase 7.4 when feasibility is skipped ----------


def test_unordered_forecasting_caught_by_phase_7_4_when_feasibility_skipped():
    built = _build(_forecasting_frame("shuffled"), run_feasibility=False)
    _, problem, _, readiness, *_ = built
    assert problem.feasibility.status.value == "not_yet_inferred"
    assert readiness.ready is True  # feasibility is advisory when not run
    out = train_and_evaluate_models(*built)
    assert out.status is UNAVAILABLE
    assert "not in chronological order on the declared time column 'day'" in (out.reason or "")
    assert "does not reorder rows" in (out.reason or "")


def test_declared_time_column_dropped_before_7_4_is_unavailable():
    df, problem, fe, readiness, split, candidates = _build(
        _forecasting_frame("sorted"), run_feasibility=False
    )
    out = train_and_evaluate_models(
        df.drop(columns=["day"]), problem, fe, readiness, split, candidates
    )
    assert out.status is UNAVAILABLE
    assert "declared time column 'day' is not in the DataFrame" in (out.reason or "")


# --- determinism -----------------------------------------------


def test_deterministic_sorted():
    built = _build(_forecasting_frame("sorted"))
    assert (
        train_and_evaluate_models(*built).model_dump_json()
        == train_and_evaluate_models(*built).model_dump_json()
    )


def test_deterministic_unordered_unavailable():
    built = _build(_forecasting_frame("shuffled"), run_feasibility=False)
    a = train_and_evaluate_models(*built)
    b = train_and_evaluate_models(*built)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.status is UNAVAILABLE


# --- forecasting execution: temporal features + contiguity -----


def test_sorted_forecasting_run_builds_temporal_features():
    out = train_and_evaluate_models(*_build(_forecasting_frame("sorted")))
    assert out.status is COMPLETED
    for run in out.runs:
        assert run.temporal_features_built > 0
        assert run.rows_consumed_as_history >= 30
    assert any("leakage-safe for one-step-ahead evaluation" in n for n in out.notes)
    assert any("consumed as lag / rolling history" in n for n in out.notes)


def test_leading_and_trailing_missing_target_are_trimmed_contiguously():
    df = _forecasting_frame("sorted")
    df.loc[:3, "demand"] = np.nan
    df.loc[_N - 3 :, "demand"] = np.nan
    out = train_and_evaluate_models(*_build(df))
    assert out.status is COMPLETED
    assert any("trimmed (contiguity preserved" in n for n in out.notes)


def test_internal_target_gap_is_blocked_at_feasibility():
    df = _forecasting_frame("sorted")
    df.loc[100:105, "demand"] = np.nan
    df2, problem, fe, readiness, split, candidates = _build(df)
    assert problem.feasibility.feasible is False
    assert readiness.ready is False
    out = train_and_evaluate_models(df2, problem, fe, readiness, split, candidates)
    assert out.status is UNAVAILABLE


def test_too_few_modelable_rows_after_history_is_unavailable():
    # a hand-built temporal section with a large warm-up on a modest frame:
    # 60 rows - lag 50 warm-up = 10 modelable < MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS.
    from data_engine.feature_engineering import TemporalFeatureRecommendation
    from data_engine.feature_engineering.models import FeatureOperationType

    df, problem, fe, readiness, split, candidates = _build(_forecasting_frame("sorted").iloc[:60])
    big_lag = TemporalFeatureRecommendation(
        column="demand",
        operation=FeatureOperationType.LAG_FEATURE,
        description="lag 50",
        reason="x",
    )
    fe_big = fe.model_copy(
        update={
            "temporal": fe.temporal.model_copy(
                update={
                    "recommendations": [big_lag],
                    "recommended_operations": ["demand: lag 50"],
                }
            )
        }
    )
    out = train_and_evaluate_models(df, problem, fe_big, readiness, split, candidates)
    assert out.status is UNAVAILABLE
    assert "modelable row(s) remain after consuming" in (out.reason or "")


# --- the non-temporal contract is unchanged --------------------


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
    tmp = recommend_temporal_features(df, inv, task)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp, temporal=tmp)
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d")).model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "temporal": tmp,
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
    assert spec.readiness.ready is False
    assert spec.training.status is ModelingStatus.UNAVAILABLE
    assert spec.selection.selected_family is None
    assert "chronological order" in (spec.reason or "")


def test_understand_modeling_unaffected():
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective=_OBJECTIVE))
    assert spec.status is ModelingStatus.NOT_YET_INFERRED
