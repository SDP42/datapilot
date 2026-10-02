"""Phase 12.1 — the deterministic tool executor (`ai_engine.execution`)."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from data_engine.modeling import TrainingRunStatus, ModelingRequest, run_modeling_pipeline
from data_engine.ingestion import RawDataStore, ingest_dataset
from ai_engine.execution import ExecutionContext, execute_tool
from experimentation import ExperimentSource, ExperimentStore, record_experiment


@pytest.fixture
def dataset(tmp_path):
    csv_path = tmp_path / "d.csv"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(size=60) * 0.1
    df.to_csv(csv_path, index=False)
    ref = ingest_dataset(csv_path, raw_store=RawDataStore(tmp_path / "raw"))
    return df, ref


def test_unknown_tool_returns_unavailable():
    result = execute_tool("not_a_real_tool", {}, ExecutionContext())
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert "not a known tool" in (result.reason or "")


def test_analyze_quality_missing_context_returns_unavailable():
    result = execute_tool("analyze_quality", {}, ExecutionContext())
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert "reference" in (result.reason or "")


def test_analyze_quality_runs_with_real_context(dataset):
    df, ref = dataset
    result = execute_tool("analyze_quality", {}, ExecutionContext(df=df, reference=ref))
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.tool == "analyze_quality"
    assert result.output is not None


def test_analyze_dataframe_runs_with_real_context(dataset):
    df, ref = dataset
    result = execute_tool("analyze_dataframe", {}, ExecutionContext(df=df, reference=ref))
    assert result.status is TrainingRunStatus.COMPLETED


def test_understand_problem_requires_objective(dataset):
    df, ref = dataset
    result = execute_tool("understand_problem", {}, ExecutionContext(df=df, reference=ref))
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert "objective" in (result.reason or "")


def test_understand_problem_completes_with_objective(dataset):
    df, ref = dataset
    result = execute_tool(
        "understand_problem", {"objective": "predict y"}, ExecutionContext(df=df, reference=ref)
    )
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.output["status"] == "completed"


def test_run_modeling_pipeline_completes(dataset):
    df, ref = dataset
    result = execute_tool(
        "run_modeling_pipeline",
        {"objective": "predict y"},
        ExecutionContext(df=df, reference=ref),
    )
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.output["status"] == "completed"


def test_select_dl_models_without_candidates_is_unavailable():
    result = execute_tool("select_dl_models", {}, ExecutionContext())
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert "dl_candidates" in (result.reason or "")
    assert "no default architecture" in (result.reason or "")


def test_compute_permutation_importance_without_model_is_unavailable():
    result = execute_tool("compute_permutation_importance", {}, ExecutionContext())
    assert result.status is TrainingRunStatus.UNAVAILABLE


def test_compute_permutation_importance_runs_with_real_context():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] * 3 + X[:, 1]
    model = LinearRegression().fit(X, y)
    context = ExecutionContext(fitted_model=model, X_eval=X, y_eval=y, feature_names=["a", "b"])
    result = execute_tool("compute_permutation_importance", {}, context)
    assert result.status is TrainingRunStatus.COMPLETED


def test_compare_experiments_without_store_is_unavailable():
    result = execute_tool("compare_experiments", {"experiment_ids": ["a", "b"]}, ExecutionContext())
    assert result.status is TrainingRunStatus.UNAVAILABLE


def test_compare_experiments_requires_at_least_two_ids():
    store = ExperimentStore(tempfile.mkdtemp())
    result = execute_tool(
        "compare_experiments",
        {"experiment_ids": ["only-one"]},
        ExecutionContext(experiment_store=store),
    )
    assert result.status is TrainingRunStatus.UNAVAILABLE


def test_compare_experiments_unknown_id_returns_failed():
    store = ExperimentStore(tempfile.mkdtemp())
    result = execute_tool(
        "compare_experiments",
        {"experiment_ids": ["missing-1", "missing-2"]},
        ExecutionContext(experiment_store=store),
    )
    assert result.status is TrainingRunStatus.FAILED


def test_compare_experiments_runs_with_real_context(dataset, tmp_path):
    df, ref = dataset
    spec = run_modeling_pipeline(
        df, ModelingRequest(dataset_id=ref.dataset_id, objective="predict y")
    )
    r1 = record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)
    r2 = record_experiment(source=ExperimentSource.CLASSICAL_MODELING, classical_result=spec)
    store = ExperimentStore(tmp_path / "experiments")
    store.register(r1)
    store.register(r2)
    result = execute_tool(
        "compare_experiments",
        {"experiment_ids": [r1.experiment_id, r2.experiment_id]},
        ExecutionContext(experiment_store=store),
    )
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.output["status"] == "completed"


def test_execute_tool_never_mutates_context(dataset):
    df, ref = dataset
    df_before = df.copy()
    execute_tool("analyze_quality", {}, ExecutionContext(df=df, reference=ref))
    pd.testing.assert_frame_equal(df, df_before)


def test_result_is_json_roundtrippable(dataset):
    from ai_engine.execution import ExecutionResult

    df, ref = dataset
    result = execute_tool("analyze_quality", {}, ExecutionContext(df=df, reference=ref))
    restored = ExecutionResult.model_validate_json(result.model_dump_json())
    assert restored == result
