"""Phase 8.1 — the deterministic DL training configuration contract
(`dl_engine.contracts.DLTrainingConfig`).
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_engine.modeling import ModelFamily
from data_engine.problem_understanding import TaskType
from dl_engine.contracts import (
    DLDevice,
    DLLoss,
    DLOptimizer,
    DLTrainingConfig,
    DLTrainingStatus,
)

_RUNTIME_ONLY_ATTR_NAMES = {
    "model",
    "module",
    "tensor",
    "array",
    "dataloader",
    "optimizer_instance",
    "device_handle",
}


def _config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {"architecture_name": "mlp", "task_type": TaskType.REGRESSION}
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


# --- construction / defaults -----------------------------------------


def test_construction_with_required_fields_only():
    config = _config()
    assert config.architecture_name == "mlp"
    assert config.task_type is TaskType.REGRESSION


def test_defaults_do_not_imply_training_occurred():
    config = _config()
    assert config.status is DLTrainingStatus.NOT_YET_STARTED
    assert config.reason is None


def test_defaults_match_documented_conventions():
    config = _config()
    assert config.family is ModelFamily.NEURAL
    assert config.seed == 42  # mirrors MODEL_TRAINING_RANDOM_SEED
    assert config.epochs == 1
    assert config.batch_size == 32
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.optimizer is DLOptimizer.ADAM
    assert config.loss is DLLoss.MSE
    assert config.device is DLDevice.CPU
    assert config.deterministic_mode is True


def test_missing_required_fields_raises():
    with pytest.raises(ValidationError):
        DLTrainingConfig()  # type: ignore[call-arg]


# --- validation --------------------------------------------------------


@pytest.mark.parametrize("epochs", [0, -1])
def test_epochs_must_be_at_least_one(epochs):
    with pytest.raises(ValidationError):
        _config(epochs=epochs)


@pytest.mark.parametrize("batch_size", [0, -5])
def test_batch_size_must_be_at_least_one(batch_size):
    with pytest.raises(ValidationError):
        _config(batch_size=batch_size)


@pytest.mark.parametrize("lr", [0.0, -0.1])
def test_learning_rate_must_be_positive(lr):
    with pytest.raises(ValidationError):
        _config(learning_rate=lr)


def test_unrecognised_optimizer_raises():
    with pytest.raises(ValidationError):
        _config(optimizer="rmsprop")


def test_unrecognised_loss_raises():
    with pytest.raises(ValidationError):
        _config(loss="huber")


def test_unrecognised_device_raises():
    with pytest.raises(ValidationError):
        _config(device="tpu")


def test_arbitrary_task_type_and_family_accepted():
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION, family=ModelFamily.NEURAL)
    assert config.task_type is TaskType.MULTICLASS_CLASSIFICATION


# --- JSON serialisation / round-trip / determinism ----------------------


def test_json_serialisable():
    config = _config()
    payload = config.model_dump_json()
    json.loads(payload)  # must be valid JSON


def test_json_round_trip():
    config = _config(epochs=5, batch_size=64, optimizer=DLOptimizer.SGD)
    payload = config.model_dump_json()
    restored = DLTrainingConfig.model_validate_json(payload)
    assert restored == config


def test_repeated_serialisation_is_deterministic():
    config = _config()
    assert config.model_dump_json() == config.model_dump_json()


def test_unavailable_state_is_representable_and_serialisable():
    config = _config(status=DLTrainingStatus.UNAVAILABLE, reason="PyTorch is not installed")
    payload = config.model_dump_json()
    restored = DLTrainingConfig.model_validate_json(payload)
    assert restored.status is DLTrainingStatus.UNAVAILABLE
    assert restored.reason == "PyTorch is not installed"


# --- no runtime-only objects --------------------------------------------


def test_no_runtime_only_fields_declared():
    field_names = set(DLTrainingConfig.model_fields)
    assert field_names.isdisjoint(_RUNTIME_ONLY_ATTR_NAMES)


def test_every_field_value_is_a_json_primitive_or_enum():
    config = _config()
    dumped = config.model_dump(mode="json")
    for value in dumped.values():
        assert isinstance(value, (str, int, float, bool, type(None)))
