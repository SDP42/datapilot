"""Phase 9.4 — optional MLflow logging
(`experimentation.mlflow_integration.log_experiment_to_mlflow`).

Environment-independent failure-path tests (MLflow missing) run in every
environment via the injectable `_import` seam. A test that actually logs
to MLflow starts with `pytest.importorskip("mlflow")` and skips cleanly
when MLflow is not installed.
"""

from __future__ import annotations

import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine import DLEvaluationResult, DLModelingResult
from experimentation.contracts import ExperimentSource, record_experiment
from experimentation.mlflow_integration import log_experiment_to_mlflow


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _dl_modeling_record():
    result = DLModelingResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=TaskType.REGRESSION,
        evaluation=DLEvaluationResult(
            status=TrainingRunStatus.COMPLETED,
            task_type=TaskType.REGRESSION,
            sample_count=10,
            metrics={"rmse": 1.23, "mae": 0.9},
            primary_metric="rmse",
        ),
    )
    return record_experiment(source=ExperimentSource.DL_MODELING, dl_modeling_result=result, seed=7)


def test_unavailable_when_mlflow_missing():
    record = _dl_modeling_record()
    result = log_experiment_to_mlflow(record, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.experiment_id == record.experiment_id
    assert result.mlflow_run_id is None
    assert "mlflow" in (result.reason or "").lower()


def test_result_is_json_serialisable_on_unavailable_path():
    from experimentation.contracts import MLflowLogResult

    record = _dl_modeling_record()
    result = log_experiment_to_mlflow(record, _import=_raise_import_error)
    restored = MLflowLogResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_real_mlflow_logging(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    import mlflow

    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")

    record = _dl_modeling_record()
    result = log_experiment_to_mlflow(record, experiment_name="datapilot-test")

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.mlflow_run_id is not None
    assert result.logged_metrics == {"rmse": 1.23, "mae": 0.9}

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(result.mlflow_run_id)
    assert run.data.params["experiment_id"] == record.experiment_id
    assert run.data.params["seed"] == "7"
    assert run.data.metrics["rmse"] == 1.23
