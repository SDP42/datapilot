"""Post-Phase-7 stabilization — `summarize_evaluation` / `EvaluationResults`.

`ModelingSpec.training` (`TrainingOutcome`) is the single source of truth
for evaluation. `EvaluationResults` is a deterministic status mirror; it
must never recompute or re-store a metric value.
"""

from __future__ import annotations

import json

import pytest

from data_engine import modeling
from data_engine.modeling import (
    EvaluationResults,
    ModelFamily,
    ModelingStatus,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
    summarize_evaluation,
)

COMPLETED = ModelingStatus.COMPLETED
UNAVAILABLE = ModelingStatus.UNAVAILABLE
NOT_YET = ModelingStatus.NOT_YET_INFERRED


def _run(family: str, estimator: str, status: TrainingRunStatus, metrics=None, reason=None):
    return TrainingRun(
        family=ModelFamily(family),
        estimator_name=estimator,
        status=status,
        metrics=metrics or {},
        reason=reason,
    )


def _training(
    *runs: TrainingRun, status: ModelingStatus = COMPLETED, reason=None
) -> TrainingOutcome:
    completed = [r.family.value for r in runs if r.status is TrainingRunStatus.COMPLETED]
    failed = [r.family.value for r in runs if r.status is not TrainingRunStatus.COMPLETED]
    return TrainingOutcome(
        status=status,
        reason=reason,
        runs=list(runs),
        successful_runs=completed,
        failed_runs=failed,
    )


# --- API -----------------------------------------------------------


def test_exported():
    assert modeling.summarize_evaluation is summarize_evaluation
    assert "summarize_evaluation" in modeling.__all__


def test_non_training_outcome_raises():
    with pytest.raises(TypeError):
        summarize_evaluation({"status": "completed"})  # type: ignore[arg-type]


# --- bare contract / backward compatibility ----------------------


def test_bare_evaluation_results_is_all_null():
    e = EvaluationResults()
    assert e.status is NOT_YET
    assert e.reason is None
    assert e.source is None
    assert e.evaluated_run_count == 0
    assert e.successful_run_count == 0
    assert e.metric_names == []
    assert e.notes == []


def test_legacy_json_without_additive_fields_validates():
    legacy = {"status": "not_yet_inferred", "reason": None, "notes": []}
    e = EvaluationResults.model_validate(legacy)
    assert e.status is NOT_YET
    assert e.source is None


def test_bare_json_values_are_null_or_empty():
    data = json.loads(EvaluationResults().model_dump_json())
    assert data["status"] == "not_yet_inferred"
    for value in data.values():
        assert value in (None, [], {}, 0, False, "not_yet_inferred")


# --- mirror semantics -------------------------------------------


def test_unavailable_when_training_not_completed():
    e = summarize_evaluation(_training(status=UNAVAILABLE, reason="scikit-learn missing"))
    assert e.status is UNAVAILABLE
    assert e.source == "training_outcome"
    assert "not completed" in (e.reason or "")


def test_mirrors_run_counts_and_metric_names():
    training = _training(
        _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0, "mae": 0.5}),
        _run(
            "tree_based",
            "DecisionTreeRegressor",
            TrainingRunStatus.COMPLETED,
            {"rmse": 2.0, "r2": 0.3},
        ),
        _run("ensemble", "RandomForestRegressor", TrainingRunStatus.FAILED, reason="boom"),
    )
    e = summarize_evaluation(training)
    assert e.status is COMPLETED
    assert e.source == "training_outcome"
    assert e.evaluated_run_count == 3
    assert e.successful_run_count == 2
    assert e.metric_names == ["mae", "r2", "rmse"]  # sorted union, names only


def test_does_not_store_metric_values():
    training = _training(
        _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 3.14159}),
    )
    payload = json.dumps(json.loads(summarize_evaluation(training).model_dump_json()))
    assert "3.14159" not in payload  # the value lives only in TrainingOutcome


def test_deterministic_repeated():
    training = _training(
        _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 0.9}),
    )
    assert (
        summarize_evaluation(training).model_dump_json()
        == summarize_evaluation(training).model_dump_json()
    )


def test_completed_training_with_zero_successes():
    e = summarize_evaluation(
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.FAILED, reason="x"),
            status=COMPLETED,
            reason="all 1 candidate model family(ies) failed to train",
        )
    )
    assert e.status is COMPLETED
    assert e.successful_run_count == 0
    assert e.metric_names == []
