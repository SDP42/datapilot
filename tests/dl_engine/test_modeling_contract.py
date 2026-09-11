"""Phase 8.5 — the DL modeling aggregate contract
(`dl_engine.contracts.DLModelingResult`).

Pure Pydantic — runs in every environment, no PyTorch required.
"""

from __future__ import annotations

import json

from data_engine.modeling import ModelFamily, TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.contracts import DLEvaluationResult, DLModelingResult, DLTrainingResult


def _training_result(**overrides: object) -> DLTrainingResult:
    defaults: dict[str, object] = {
        "status": TrainingRunStatus.COMPLETED,
        "epochs_requested": 3,
        "epochs_completed": 3,
        "batch_size": 8,
        "learning_rate": 0.01,
        "optimizer": "adam",
        "loss": "mse",
        "seed": 42,
        "deterministic_mode": True,
        "loss_history": [1.0, 0.5, 0.2],
        "final_loss": 0.2,
    }
    defaults.update(overrides)
    return DLTrainingResult.model_validate(defaults)


def _evaluation_result(**overrides: object) -> DLEvaluationResult:
    defaults: dict[str, object] = {
        "status": TrainingRunStatus.COMPLETED,
        "task_type": TaskType.REGRESSION,
        "sample_count": 10,
        "metrics": {"rmse": 0.3, "mae": 0.2},
        "primary_metric": "rmse",
    }
    defaults.update(overrides)
    return DLEvaluationResult.model_validate(defaults)


def _modeling_result(**overrides: object) -> DLModelingResult:
    defaults: dict[str, object] = {
        "status": TrainingRunStatus.COMPLETED,
        "task_type": TaskType.REGRESSION,
        "architecture_name": "mlp",
        "training": _training_result(),
        "evaluation": _evaluation_result(),
    }
    defaults.update(overrides)
    return DLModelingResult.model_validate(defaults)


# --- construction / defaults ------------------------------------------


def test_construction_with_required_fields_only():
    result = DLModelingResult(status=TrainingRunStatus.UNAVAILABLE, task_type=TaskType.REGRESSION)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.family is ModelFamily.NEURAL
    assert result.architecture_name is None
    assert result.training is None
    assert result.evaluation is None
    assert result.reason is None
    assert result.notes == []


def test_completed_result_nests_training_and_evaluation():
    result = _modeling_result()
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.training is not None
    assert result.training.final_loss == 0.2
    assert result.evaluation is not None
    assert result.evaluation.metrics["rmse"] == 0.3


def test_training_or_evaluation_none_when_stage_did_not_run():
    result = _modeling_result(
        status=TrainingRunStatus.UNAVAILABLE, training=None, evaluation=None, reason="unavailable"
    )
    assert result.training is None
    assert result.evaluation is None


# --- no runtime-only / experiment-tracking fields -----------------------


def test_no_runtime_only_fields():
    field_names = set(DLModelingResult.model_fields)
    assert field_names.isdisjoint({"model", "tensor", "optimizer_instance", "gradient"})


def test_no_experiment_tracking_or_selection_fields():
    field_names = {name.lower() for name in DLModelingResult.model_fields}
    for forbidden in (
        "experiment",
        "mlflow",
        "rank",
        "selected",
        "hyperparameter",
        "artifact",
        "timestamp",
        "uuid",
    ):
        assert not any(forbidden in name for name in field_names)


# --- serialization -------------------------------------------------------


def test_json_serialisable():
    result = _modeling_result()
    json.loads(result.model_dump_json())


def test_json_round_trip():
    result = _modeling_result()
    restored = DLModelingResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_repeated_serialisation_is_deterministic():
    result = _modeling_result()
    assert result.model_dump_json() == result.model_dump_json()


def test_failed_result_names_the_stage_in_reason():
    result = _modeling_result(
        status=TrainingRunStatus.FAILED,
        evaluation=None,
        reason="training stage did not complete: device unavailable",
    )
    assert "training stage" in (result.reason or "")
    assert result.evaluation is None
