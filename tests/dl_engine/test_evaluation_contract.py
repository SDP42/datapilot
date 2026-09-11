"""Phase 8.4 — the DL evaluation contract (`dl_engine.contracts.DLEvaluationResult`).

Pure Pydantic — runs in every environment, no PyTorch required.
"""

from __future__ import annotations

import json

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.contracts import DLEvaluationResult


def _result(**overrides: object) -> DLEvaluationResult:
    defaults: dict[str, object] = {
        "status": TrainingRunStatus.COMPLETED,
        "task_type": TaskType.REGRESSION,
        "architecture_name": "mlp",
        "sample_count": 10,
        "metrics": {"rmse": 1.23, "mae": 0.98},
        "primary_metric": "rmse",
    }
    defaults.update(overrides)
    return DLEvaluationResult.model_validate(defaults)


def test_construction_with_required_fields_only():
    result = DLEvaluationResult(status=TrainingRunStatus.UNAVAILABLE, task_type=TaskType.REGRESSION)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.sample_count == 0
    assert result.metrics == {}
    assert result.primary_metric is None
    assert result.architecture_name is None
    assert result.reason is None
    assert result.notes == []


def test_completed_result_carries_metrics():
    result = _result()
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.metrics["rmse"] == 1.23
    assert result.primary_metric == "rmse"


def test_no_runtime_only_fields():
    field_names = set(DLEvaluationResult.model_fields)
    assert field_names.isdisjoint(
        {"model", "tensor", "optimizer_instance", "gradient", "experiment_id", "mlflow_run_id"}
    )


def test_no_model_selection_or_experiment_tracking_fields():
    field_names = {name.lower() for name in DLEvaluationResult.model_fields}
    for forbidden in ("rank", "selected", "experiment", "mlflow", "hyperparameter", "artifact"):
        assert not any(forbidden in name for name in field_names)


# --- serialization ---------------------------------------------------------


def test_json_serialisable():
    result = _result()
    payload = result.model_dump_json()
    json.loads(payload)


def test_json_round_trip():
    result = _result(metrics={"accuracy": 0.9, "f1": 0.88}, primary_metric="f1")
    restored = DLEvaluationResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_repeated_serialisation_is_deterministic():
    result = _result()
    assert result.model_dump_json() == result.model_dump_json()


def test_failed_result_carries_reason_and_no_metrics():
    result = DLEvaluationResult(
        status=TrainingRunStatus.FAILED,
        task_type=TaskType.BINARY_CLASSIFICATION,
        reason="model output shape mismatch",
    )
    assert result.metrics == {}
    assert result.reason == "model output shape mismatch"


def test_notes_can_explain_an_undefined_metric():
    result = _result(
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics={"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        primary_metric="f1",
        notes=[
            "roc_auc omitted: mathematically undefined — the evaluation data contains only one class"
        ],
    )
    assert "roc_auc" not in result.metrics
    assert any("roc_auc" in n for n in result.notes)
