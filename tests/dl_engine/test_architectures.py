"""Phase 8.3 — the MLP architecture configuration contract
(`dl_engine.architectures.MLPArchitectureConfig`).

Pure Pydantic — runs in every environment, no PyTorch required.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPActivation, MLPArchitectureConfig


def _config(**overrides: object) -> MLPArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_features": 4,
        "output_dim": 1,
        "hidden_layer_sizes": [8],
    }
    defaults.update(overrides)
    return MLPArchitectureConfig.model_validate(defaults)


# --- valid configurations ------------------------------------------------


def test_valid_regression_config():
    config = _config()
    assert config.task_type is TaskType.REGRESSION
    assert config.output_dim == 1
    assert config.architecture_name == "mlp"


def test_valid_binary_classification_config():
    config = _config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2)
    assert config.output_dim == 2


def test_valid_multiclass_classification_config():
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=5)
    assert config.output_dim == 5


def test_defaults():
    config = _config()
    assert config.activation is MLPActivation.RELU
    assert config.dropout == 0.0


def test_multiple_hidden_layers_and_custom_activation_accepted():
    config = _config(hidden_layer_sizes=[32, 16, 8], activation=MLPActivation.GELU, dropout=0.2)
    assert config.hidden_layer_sizes == [32, 16, 8]
    assert config.activation is MLPActivation.GELU
    assert config.dropout == pytest.approx(0.2)


# --- invalid configurations -----------------------------------------------


def test_zero_input_features_rejected():
    with pytest.raises(ValidationError):
        _config(input_features=0)


def test_negative_input_features_rejected():
    with pytest.raises(ValidationError):
        _config(input_features=-1)


def test_zero_output_dim_rejected():
    with pytest.raises(ValidationError):
        _config(output_dim=0)


def test_negative_output_dim_rejected():
    with pytest.raises(ValidationError):
        _config(output_dim=-1)


def test_empty_hidden_layers_rejected():
    with pytest.raises(ValidationError):
        _config(hidden_layer_sizes=[])


def test_zero_hidden_layer_size_rejected():
    with pytest.raises(ValidationError, match="positive integer"):
        _config(hidden_layer_sizes=[8, 0])


def test_negative_hidden_layer_size_rejected():
    with pytest.raises(ValidationError, match="positive integer"):
        _config(hidden_layer_sizes=[-4])


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_rejected(dropout):
    with pytest.raises(ValidationError):
        _config(dropout=dropout)


def test_unsupported_task_type_rejected():
    with pytest.raises(ValidationError, match="unsupported task_type"):
        _config(task_type=TaskType.CLUSTERING, output_dim=1)


def test_other_task_forecasting_rejected():
    with pytest.raises(ValidationError, match="unsupported task_type"):
        _config(task_type=TaskType.TIME_SERIES_FORECASTING, output_dim=1)


# --- invalid task/output_dim combinations ----------------------------------


def test_regression_with_output_dim_other_than_one_rejected():
    with pytest.raises(ValidationError, match="output_dim == 1"):
        _config(task_type=TaskType.REGRESSION, output_dim=2)


def test_binary_classification_with_output_dim_other_than_two_rejected():
    with pytest.raises(ValidationError, match="output_dim == 2"):
        _config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=1)


def test_binary_classification_with_output_dim_three_rejected():
    with pytest.raises(ValidationError, match="output_dim == 2"):
        _config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=3)


def test_multiclass_classification_with_output_dim_one_rejected():
    with pytest.raises(ValidationError, match="output_dim >= 2"):
        _config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=1)


def test_unrecognised_activation_rejected():
    with pytest.raises(ValidationError):
        _config(activation="sigmoid")


def test_architecture_name_is_fixed_literal():
    with pytest.raises(ValidationError):
        _config(architecture_name="cnn")


# --- serialization ---------------------------------------------------------


def test_json_serialisable():
    config = _config()
    payload = config.model_dump_json()
    json.loads(payload)


def test_json_round_trip():
    config = _config(hidden_layer_sizes=[16, 8], dropout=0.1)
    restored = MLPArchitectureConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_repeated_serialisation_is_deterministic():
    config = _config()
    assert config.model_dump_json() == config.model_dump_json()


def test_no_runtime_only_fields():
    field_names = set(MLPArchitectureConfig.model_fields)
    assert field_names.isdisjoint({"model", "module", "tensor", "network"})
