"""Phase 8.7 — the advanced architecture builders
(`dl_engine.cnn.build_cnn` / `dl_engine.lstm.build_lstm` /
`dl_engine.transformer.build_transformer`).

Every test that constructs a real model starts with
`pytest.importorskip("torch")` and skips cleanly when PyTorch is not
installed. A few environment-independent tests (PyTorch missing) run in
every environment via the injectable `_import` seam.
"""

from __future__ import annotations

import pytest

from data_engine.problem_understanding import TaskType
from dl_engine.architectures import (
    CNNArchitectureConfig,
    LSTMArchitectureConfig,
    TransformerArchitectureConfig,
)
from dl_engine.cnn import build_cnn
from dl_engine.lstm import build_lstm
from dl_engine.runtime import seed_everything
from dl_engine.transformer import build_transformer


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _cnn_config(**overrides: object) -> CNNArchitectureConfig:
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


def _lstm_config(**overrides: object) -> LSTMArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_size": 4,
        "hidden_size": 8,
        "num_layers": 1,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return LSTMArchitectureConfig.model_validate(defaults)


def _transformer_config(**overrides: object) -> TransformerArchitectureConfig:
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


# --- PyTorch-unavailable behavior (no real torch needed) -----------------


def test_build_cnn_raises_runtime_error_when_torch_missing():
    with pytest.raises(RuntimeError, match="PyTorch"):
        build_cnn(_cnn_config(), _import=_raise_import_error)


def test_build_lstm_raises_runtime_error_when_torch_missing():
    with pytest.raises(RuntimeError, match="PyTorch"):
        build_lstm(_lstm_config(), _import=_raise_import_error)


def test_build_transformer_raises_runtime_error_when_torch_missing():
    with pytest.raises(RuntimeError, match="PyTorch"):
        build_transformer(_transformer_config(), _import=_raise_import_error)


def test_importing_dl_engine_still_succeeds_without_torch():
    import dl_engine

    assert dl_engine.build_cnn is not None
    assert dl_engine.build_lstm is not None
    assert dl_engine.build_transformer is not None


# --- CNN: returns an nn.Module, correct output shapes ---------------------


def test_build_cnn_returns_an_nn_module():
    torch = pytest.importorskip("torch")
    model = build_cnn(_cnn_config())
    assert isinstance(model, torch.nn.Module)


def test_cnn_regression_output_shape():
    torch = pytest.importorskip("torch")
    config = _cnn_config(task_type=TaskType.REGRESSION, output_dim=1)
    model = build_cnn(config)
    x = torch.randn(5, config.input_channels, config.sequence_length)
    out = model(x)
    assert out.shape == (5, 1)


def test_cnn_binary_classification_output_shape_two_logits():
    torch = pytest.importorskip("torch")
    config = _cnn_config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2)
    model = build_cnn(config)
    x = torch.randn(7, config.input_channels, config.sequence_length)
    out = model(x)
    assert out.shape == (7, 2)


def test_cnn_multiclass_classification_output_shape():
    torch = pytest.importorskip("torch")
    config = _cnn_config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=5)
    model = build_cnn(config)
    x = torch.randn(4, config.input_channels, config.sequence_length)
    out = model(x)
    assert out.shape == (4, 5)


def test_cnn_with_pooling_output_shape():
    torch = pytest.importorskip("torch")
    config = _cnn_config(pooling=True)
    model = build_cnn(config)
    x = torch.randn(3, config.input_channels, config.sequence_length)
    out = model(x)
    assert out.shape == (3, 1)


def test_cnn_rejects_invalid_input_shape():
    pytest.importorskip("torch")
    import torch

    config = _cnn_config()
    model = build_cnn(config)
    wrong_channels = torch.randn(5, config.input_channels + 1, config.sequence_length)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_channels)
    wrong_length = torch.randn(5, config.input_channels, config.sequence_length + 1)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_length)
    wrong_ndim = torch.randn(5, config.sequence_length)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_ndim)


def test_cnn_classification_output_is_raw_logits_not_probabilities():
    torch = pytest.importorskip("torch")
    config = _cnn_config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=3)
    model = build_cnn(config)
    x = torch.randn(10, config.input_channels, config.sequence_length) * 10
    out = model(x)
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_cnn_deterministic_construction_when_seeded():
    torch = pytest.importorskip("torch")
    config = _cnn_config()

    seed_everything(42)
    model_a = build_cnn(config)
    seed_everything(42)
    model_b = build_cnn(config)

    params_a = list(model_a.parameters())
    params_b = list(model_b.parameters())
    assert len(params_a) == len(params_b)
    for pa, pb in zip(params_a, params_b, strict=True):
        assert torch.equal(pa, pb)


def test_build_cnn_does_not_mutate_config():
    pytest.importorskip("torch")
    config = _cnn_config()
    before = config.model_dump_json()
    build_cnn(config)
    assert config.model_dump_json() == before


# --- LSTM: returns an nn.Module, correct output shapes --------------------


def test_build_lstm_returns_an_nn_module():
    torch = pytest.importorskip("torch")
    model = build_lstm(_lstm_config())
    assert isinstance(model, torch.nn.Module)


def test_lstm_regression_output_shape():
    torch = pytest.importorskip("torch")
    config = _lstm_config(task_type=TaskType.REGRESSION, output_dim=1)
    model = build_lstm(config)
    x = torch.randn(6, 12, config.input_size)  # arbitrary seq_len
    out = model(x)
    assert out.shape == (6, 1)


def test_lstm_binary_classification_output_shape_two_logits():
    torch = pytest.importorskip("torch")
    config = _lstm_config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2)
    model = build_lstm(config)
    x = torch.randn(5, 8, config.input_size)
    out = model(x)
    assert out.shape == (5, 2)


def test_lstm_multiclass_classification_output_shape():
    torch = pytest.importorskip("torch")
    config = _lstm_config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=4)
    model = build_lstm(config)
    x = torch.randn(3, 6, config.input_size)
    out = model(x)
    assert out.shape == (3, 4)


def test_lstm_bidirectional_output_shape():
    torch = pytest.importorskip("torch")
    config = _lstm_config(num_layers=2, bidirectional=True, dropout=0.1)
    model = build_lstm(config)
    x = torch.randn(4, 7, config.input_size)
    out = model(x)
    assert out.shape == (4, 1)


def test_lstm_varying_sequence_length_still_works():
    torch = pytest.importorskip("torch")
    config = _lstm_config()
    model = build_lstm(config)
    for seq_len in (1, 5, 20):
        x = torch.randn(2, seq_len, config.input_size)
        out = model(x)
        assert out.shape == (2, 1)


def test_lstm_rejects_invalid_input_shape():
    pytest.importorskip("torch")
    import torch

    config = _lstm_config()
    model = build_lstm(config)
    wrong_feature_count = torch.randn(5, 10, config.input_size + 1)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_feature_count)
    wrong_ndim = torch.randn(5, config.input_size)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_ndim)


def test_lstm_classification_output_is_raw_logits():
    torch = pytest.importorskip("torch")
    config = _lstm_config(task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=3)
    model = build_lstm(config)
    x = torch.randn(10, 5, config.input_size) * 10
    out = model(x)
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_lstm_deterministic_construction_when_seeded():
    torch = pytest.importorskip("torch")
    config = _lstm_config(num_layers=2)

    seed_everything(7)
    model_a = build_lstm(config)
    seed_everything(7)
    model_b = build_lstm(config)

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_build_lstm_does_not_mutate_config():
    pytest.importorskip("torch")
    config = _lstm_config()
    before = config.model_dump_json()
    build_lstm(config)
    assert config.model_dump_json() == before


# --- Transformer: returns an nn.Module, correct output shapes -------------


def test_build_transformer_returns_an_nn_module():
    torch = pytest.importorskip("torch")
    model = build_transformer(_transformer_config())
    assert isinstance(model, torch.nn.Module)


def test_transformer_regression_output_shape():
    torch = pytest.importorskip("torch")
    config = _transformer_config(task_type=TaskType.REGRESSION, output_dim=1)
    model = build_transformer(config)
    x = torch.randn(5, 9, config.input_size)
    out = model(x)
    assert out.shape == (5, 1)


def test_transformer_binary_classification_output_shape_two_logits():
    torch = pytest.importorskip("torch")
    config = _transformer_config(task_type=TaskType.BINARY_CLASSIFICATION, output_dim=2)
    model = build_transformer(config)
    x = torch.randn(4, 6, config.input_size)
    out = model(x)
    assert out.shape == (4, 2)


def test_transformer_multiclass_classification_output_shape():
    torch = pytest.importorskip("torch")
    config = _transformer_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=6, d_model=16, num_heads=4
    )
    model = build_transformer(config)
    x = torch.randn(3, 5, config.input_size)
    out = model(x)
    assert out.shape == (3, 6)


def test_transformer_varying_sequence_length_still_works():
    torch = pytest.importorskip("torch")
    config = _transformer_config()
    model = build_transformer(config)
    for seq_len in (1, 4, 15):
        x = torch.randn(2, seq_len, config.input_size)
        out = model(x)
        assert out.shape == (2, 1)


def test_transformer_rejects_invalid_input_shape():
    pytest.importorskip("torch")
    import torch

    config = _transformer_config()
    model = build_transformer(config)
    wrong_feature_count = torch.randn(5, 6, config.input_size + 1)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_feature_count)
    wrong_ndim = torch.randn(5, config.input_size)
    with pytest.raises(ValueError, match="expected input shape"):
        model(wrong_ndim)


def test_transformer_classification_output_is_raw_logits():
    torch = pytest.importorskip("torch")
    config = _transformer_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION, output_dim=3, d_model=16, num_heads=4
    )
    model = build_transformer(config)
    x = torch.randn(10, 5, config.input_size) * 10
    out = model(x)
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_transformer_deterministic_construction_when_seeded():
    torch = pytest.importorskip("torch")
    config = _transformer_config(d_model=16, num_heads=4, num_encoder_layers=2)

    seed_everything(11)
    model_a = build_transformer(config)
    seed_everything(11)
    model_b = build_transformer(config)

    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_build_transformer_does_not_mutate_config():
    pytest.importorskip("torch")
    config = _transformer_config()
    before = config.model_dump_json()
    build_transformer(config)
    assert config.model_dump_json() == before


# --- no unexpected parameter mutation during construction -----------------


def test_repeated_construction_produces_equivalent_structure():
    pytest.importorskip("torch")
    cnn_config = _cnn_config()
    a = build_cnn(cnn_config)
    b = build_cnn(cnn_config)
    a_shapes = [tuple(p.shape) for p in a.parameters()]
    b_shapes = [tuple(p.shape) for p in b.parameters()]
    assert a_shapes == b_shapes

    lstm_config = _lstm_config(num_layers=2)
    c = build_lstm(lstm_config)
    d = build_lstm(lstm_config)
    assert [tuple(p.shape) for p in c.parameters()] == [tuple(p.shape) for p in d.parameters()]

    tf_config = _transformer_config()
    e = build_transformer(tf_config)
    f = build_transformer(tf_config)
    assert [tuple(p.shape) for p in e.parameters()] == [tuple(p.shape) for p in f.parameters()]
