"""Phase 8.7 — the advanced architecture contracts
(`dl_engine.architectures.CNNArchitectureConfig` / `LSTMArchitectureConfig`
/ `TransformerArchitectureConfig`).

Pure Pydantic — runs in every environment, no PyTorch required.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from data_engine.problem_understanding import TaskType
from dl_engine.architectures import (
    CNNArchitectureConfig,
    LSTMArchitectureConfig,
    MLPActivation,
    MLPArchitectureConfig,
    TransformerArchitectureConfig,
)


def _cnn(**overrides: object) -> CNNArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_channels": 1,
        "sequence_length": 10,
        "conv_channels": [8, 16],
        "kernel_size": 3,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return CNNArchitectureConfig.model_validate(defaults)


def _lstm(**overrides: object) -> LSTMArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return LSTMArchitectureConfig.model_validate(defaults)


def _transformer(**overrides: object) -> TransformerArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_size": 4,
        "d_model": 8,
        "num_heads": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 16,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return TransformerArchitectureConfig.model_validate(defaults)


# --- MLPArchitectureConfig is untouched -----------------------------------


def test_mlp_architecture_config_unchanged():
    # sanity: the existing Phase-8.3 contract still works exactly as before
    config = MLPArchitectureConfig(
        task_type=TaskType.REGRESSION, input_features=4, output_dim=1, hidden_layer_sizes=[8]
    )
    assert config.architecture_name == "mlp"


# --- CNN: valid configuration ----------------------------------------------


def test_cnn_valid_configuration():
    config = _cnn()
    assert config.architecture_name == "cnn"
    assert config.activation is MLPActivation.RELU
    assert config.pooling is False
    assert config.dropout == 0.0


def test_cnn_valid_with_pooling_and_dropout():
    config = _cnn(pooling=True, dropout=0.2, activation=MLPActivation.GELU)
    assert config.pooling is True
    assert config.dropout == pytest.approx(0.2)
    assert config.activation is MLPActivation.GELU


def test_cnn_binary_and_multiclass_configurations():
    binary = _cnn(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2)
    assert binary.output_dim == 2
    multiclass = _cnn(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=5)
    assert multiclass.output_dim == 5


# --- CNN: invalid configurations --------------------------------------------


def test_cnn_even_kernel_size_rejected():
    with pytest.raises(ValidationError, match="odd"):
        _cnn(kernel_size=4)


def test_cnn_kernel_larger_than_sequence_length_rejected():
    with pytest.raises(ValidationError, match="sequence_length"):
        _cnn(sequence_length=3, kernel_size=5)


def test_cnn_empty_conv_channels_rejected():
    with pytest.raises(ValidationError):
        _cnn(conv_channels=[])


def test_cnn_non_positive_conv_channel_rejected():
    with pytest.raises(ValidationError, match="positive integer"):
        _cnn(conv_channels=[8, 0])


def test_cnn_non_positive_input_channels_rejected():
    with pytest.raises(ValidationError):
        _cnn(input_channels=0)


def test_cnn_non_positive_sequence_length_rejected():
    with pytest.raises(ValidationError):
        _cnn(sequence_length=0)


def test_cnn_invalid_dropout_rejected():
    with pytest.raises(ValidationError):
        _cnn(dropout=1.5)


def test_cnn_unsupported_task_type_rejected():
    with pytest.raises(ValidationError, match="unsupported task_type"):
        _cnn(task_type=TaskType.CLUSTERING)


def test_cnn_regression_wrong_output_dim_rejected():
    with pytest.raises(ValidationError, match="output_dim == 1"):
        _cnn(task_type=TaskType.REGRESSION, output_dim=2)


def test_cnn_binary_wrong_output_dim_rejected():
    with pytest.raises(ValidationError, match="output_dim == 2"):
        _cnn(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=1)


def test_cnn_multiclass_output_dim_too_low_rejected():
    with pytest.raises(ValidationError, match="output_dim >= 2"):
        _cnn(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=1)


# --- LSTM: valid configuration -----------------------------------------


def test_lstm_valid_configuration():
    config = _lstm()
    assert config.architecture_name == "lstm"
    assert config.bidirectional is False
    assert config.dropout == 0.0


def test_lstm_valid_bidirectional_multilayer():
    config = _lstm(num_layers=2, bidirectional=True, dropout=0.3)
    assert config.num_layers == 2
    assert config.bidirectional is True
    assert config.dropout == pytest.approx(0.3)


# --- LSTM: invalid configurations -------------------------------------


def test_lstm_dropout_with_single_layer_rejected():
    with pytest.raises(ValidationError, match="num_layers == 1"):
        _lstm(num_layers=1, dropout=0.5)


def test_lstm_dropout_allowed_with_multiple_layers():
    config = _lstm(num_layers=2, dropout=0.5)
    assert config.dropout == pytest.approx(0.5)


def test_lstm_non_positive_hidden_size_rejected():
    with pytest.raises(ValidationError):
        _lstm(hidden_size=0)


def test_lstm_non_positive_num_layers_rejected():
    with pytest.raises(ValidationError):
        _lstm(num_layers=0)


def test_lstm_non_positive_input_size_rejected():
    with pytest.raises(ValidationError):
        _lstm(input_size=0)


def test_lstm_unsupported_task_type_rejected():
    with pytest.raises(ValidationError, match="unsupported task_type"):
        _lstm(task_type=TaskType.TIME_SERIES_FORECASTING)


def test_lstm_multiclass_output_dim_too_low_rejected():
    with pytest.raises(ValidationError, match="output_dim >= 2"):
        _lstm(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=1)


# --- Transformer: valid configuration -----------------------------------


def test_transformer_valid_configuration():
    config = _transformer()
    assert config.architecture_name == "transformer"
    assert config.d_model % config.num_heads == 0


def test_transformer_valid_multihead_multilayer():
    config = _transformer(d_model=16, num_heads=4, num_encoder_layers=3, dim_feedforward=32)
    assert config.num_heads == 4
    assert config.num_encoder_layers == 3


# --- Transformer: invalid configurations --------------------------------


def test_transformer_heads_not_dividing_d_model_rejected():
    with pytest.raises(ValidationError, match="evenly divide"):
        _transformer(d_model=10, num_heads=3)


def test_transformer_non_positive_d_model_rejected():
    with pytest.raises(ValidationError):
        _transformer(d_model=0)


def test_transformer_non_positive_num_heads_rejected():
    with pytest.raises(ValidationError):
        _transformer(num_heads=0)


def test_transformer_non_positive_dim_feedforward_rejected():
    with pytest.raises(ValidationError):
        _transformer(dim_feedforward=0)


def test_transformer_invalid_dropout_rejected():
    with pytest.raises(ValidationError):
        _transformer(dropout=1.0)


def test_transformer_unsupported_task_type_rejected():
    with pytest.raises(ValidationError, match="unsupported task_type"):
        _transformer(task_type=TaskType.OTHER)


def test_transformer_binary_wrong_output_dim_rejected():
    with pytest.raises(ValidationError, match="output_dim == 2"):
        _transformer(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=3)


# --- deterministic JSON serialization ------------------------------------


def test_cnn_json_serialisable_and_round_trip():
    config = _cnn(pooling=True, dropout=0.1)
    payload = config.model_dump_json()
    json.loads(payload)
    restored = CNNArchitectureConfig.model_validate_json(payload)
    assert restored == config


def test_lstm_json_serialisable_and_round_trip():
    config = _lstm(num_layers=2, bidirectional=True, dropout=0.2)
    payload = config.model_dump_json()
    json.loads(payload)
    restored = LSTMArchitectureConfig.model_validate_json(payload)
    assert restored == config


def test_transformer_json_serialisable_and_round_trip():
    config = _transformer(d_model=16, num_heads=4)
    payload = config.model_dump_json()
    json.loads(payload)
    restored = TransformerArchitectureConfig.model_validate_json(payload)
    assert restored == config


def test_repeated_serialisation_is_deterministic():
    cnn = _cnn()
    lstm = _lstm()
    transformer = _transformer()
    assert cnn.model_dump_json() == cnn.model_dump_json()
    assert lstm.model_dump_json() == lstm.model_dump_json()
    assert transformer.model_dump_json() == transformer.model_dump_json()


def test_no_runtime_only_fields_on_any_new_contract():
    forbidden = {"model", "tensor", "optimizer_instance", "gradient", "experiment_id", "uuid"}
    for cls in (CNNArchitectureConfig, LSTMArchitectureConfig, TransformerArchitectureConfig):
        assert forbidden.isdisjoint(set(cls.model_fields))


def test_no_hyperparameter_search_fields():
    field_names = set()
    for cls in (CNNArchitectureConfig, LSTMArchitectureConfig, TransformerArchitectureConfig):
        field_names |= {name.lower() for name in cls.model_fields}
    for forbidden in ("search", "trial", "study", "sweep", "registry", "checkpoint"):
        assert not any(forbidden in name for name in field_names)
