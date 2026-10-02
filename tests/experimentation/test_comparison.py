"""Phase 9.3 — deterministic comparison of recorded experiments
(`experimentation.comparison.compare_experiments`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.modeling import (
    ModelingRequest,
    ModelingSpec,
    TrainingRunStatus,
    run_modeling_pipeline,
)
from data_engine.problem_understanding import TaskType
from dl_engine import DLEvaluationResult, DLModelingResult, DLSelectionResult
from experimentation.comparison import compare_experiments
from experimentation.contracts import ExperimentSource, record_experiment


def _regression_df(seed, noise=0.1, n=60):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(size=n) * noise
    return df


def _classical_record(seed, noise):
    spec = run_modeling_pipeline(
        _regression_df(seed, noise), ModelingRequest(dataset_id=f"ds{seed}", objective="predict y")
    )
    return record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)


def _dl_modeling_record(rmse=1.0):
    result = DLModelingResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=TaskType.REGRESSION,
        evaluation=DLEvaluationResult(
            status=TrainingRunStatus.COMPLETED,
            task_type=TaskType.REGRESSION,
            sample_count=10,
            metrics={"rmse": rmse, "mae": rmse / 2},
            primary_metric="rmse",
        ),
    )
    return record_experiment(source=ExperimentSource.DL_MODELING, dl_modeling_result=result)


def _dl_selection_record(metric="rmse", direction="minimize", score=0.5):
    result = DLSelectionResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=TaskType.REGRESSION,
        selection_metric=metric,
        selection_direction=direction,
        selected_candidate_id="c1",
        selected_score=score,
        ranking=[],
    )
    return record_experiment(source=ExperimentSource.DL_SELECTION, dl_selection_result=result)


# --- empty / no-eligible paths ---------------------------------------------


def test_no_records_returns_failed():
    result = compare_experiments([])
    assert result.status is TrainingRunStatus.FAILED
    assert "no experiment records" in (result.reason or "")
    assert result.entries == []


def test_record_with_no_selection_is_ineligible():
    empty_spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    record = record_experiment(
        source=ExperimentSource.CLASSICAL_MODELING, classical_result=empty_spec
    )
    result = compare_experiments([record])
    assert result.status is TrainingRunStatus.FAILED
    assert len(result.entries) == 1
    assert result.entries[0].rank is None


def test_dl_modeling_record_always_ineligible():
    record = _dl_modeling_record()
    result = compare_experiments([record])
    assert result.status is TrainingRunStatus.FAILED
    assert result.entries[0].rank is None
    assert "no established selection metric" in result.entries[0].reason


# --- eligible comparisons ---------------------------------------------------


def test_two_classical_records_ranked_correctly():
    good = _classical_record(seed=1, noise=0.01)
    bad = _classical_record(seed=2, noise=5.0)
    result = compare_experiments([good, bad])
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.metric == "rmse"
    assert result.direction == "minimize"
    assert result.selected_experiment_id == good.experiment_id
    ranked = sorted(result.entries, key=lambda e: e.rank or 999)
    assert ranked[0].experiment_id == good.experiment_id
    assert ranked[0].rank == 1


def test_dl_selection_records_ranked_correctly():
    better = _dl_selection_record(score=0.1)
    worse = _dl_selection_record(score=2.0)
    result = compare_experiments([better, worse])
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selected_experiment_id == better.experiment_id


def test_mismatched_metric_direction_fails():
    rmse_record = _dl_selection_record(metric="rmse", direction="minimize", score=0.5)
    f1_record = _dl_selection_record(metric="f1", direction="maximize", score=0.8)
    result = compare_experiments([rmse_record, f1_record])
    assert result.status is TrainingRunStatus.FAILED
    assert "disagree" in (result.reason or "")


def test_mixed_eligible_and_ineligible_records():
    eligible = _dl_selection_record(score=0.3)
    ineligible = _dl_modeling_record()
    result = compare_experiments([eligible, ineligible])
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.selected_experiment_id == eligible.experiment_id
    reasons = {e.experiment_id: e.rank for e in result.entries}
    assert reasons[eligible.experiment_id] == 1
    assert reasons[ineligible.experiment_id] is None


def test_comparison_never_mutates_input_records():
    records = [_dl_selection_record(score=0.2), _dl_selection_record(score=0.4)]
    before = [r.model_dump_json() for r in records]
    compare_experiments(records)
    after = [r.model_dump_json() for r in records]
    assert before == after


def test_deterministic_repeated_comparison():
    records = [_dl_selection_record(score=0.2), _dl_selection_record(score=0.2)]  # tie
    result_a = compare_experiments(records)
    result_b = compare_experiments(records)
    assert result_a.model_dump_json() == result_b.model_dump_json()
