"""Phase 8.8 — wiring the Phase-8.7 CNN / LSTM / Transformer architectures
into real training/evaluation (`dl_engine.execution.run_cnn_modeling` /
`run_lstm_modeling` / `run_transformer_modeling`).

Environment-independent failure-path tests (PyTorch missing, mismatched
task types) run in every environment via the injectable `_import` seam.
Every test that actually trains/evaluates starts with
`pytest.importorskip("torch")` and skips cleanly when PyTorch is not
installed. Mirrors `tests/dl_engine/test_execution.py`'s structure for
`run_mlp_modeling`, scoped to the three new architectures only — the MLP
path itself is untouched and already covered there.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import (
    CNNArchitectureConfig,
    LSTMArchitectureConfig,
    TransformerArchitectureConfig,
)
from dl_engine.contracts import DLLoss, DLOptimizer, DLTrainingConfig
from dl_engine.execution import run_cnn_modeling, run_lstm_modeling, run_transformer_modeling


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _sequence_regression_split(n_train=32, n_eval=12, seq_len=6, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, channels, seq_len)).astype(np.float64)
    y_train = X_train.sum(axis=(1, 2))
    X_eval = rng.normal(size=(n_eval, channels, seq_len)).astype(np.float64) + 50.0
    y_eval = X_eval.sum(axis=(1, 2))
    return X_train, y_train, X_eval, y_eval


def _cnn_arch(**overrides: object) -> CNNArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_channels": 2,
        "sequence_length": 6,
        "conv_channels": [4],
        "kernel_size": 3,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return CNNArchitectureConfig.model_validate(defaults)


def _lstm_arch(**overrides: object) -> LSTMArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_size": 2,
        "hidden_size": 8,
        "num_layers": 1,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return LSTMArchitectureConfig.model_validate(defaults)


def _transformer_arch(**overrides: object) -> TransformerArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_size": 2,
        "d_model": 8,
        "num_heads": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 16,
        "output_dim": 1,
    }
    defaults.update(overrides)
    return TransformerArchitectureConfig.model_validate(defaults)


def _config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {
        "architecture_name": "arch",
        "task_type": TaskType.REGRESSION,
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.05,
        "optimizer": DLOptimizer.ADAM,
        "loss": DLLoss.MSE,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


# --- environment-independent failure paths (all three architectures) -----


@pytest.mark.parametrize(
    ("run_fn", "arch_fn"),
    [
        (run_cnn_modeling, _cnn_arch),
        (run_lstm_modeling, _lstm_arch),
        (run_transformer_modeling, _transformer_arch),
    ],
)
def test_unavailable_when_torch_missing(run_fn, arch_fn):
    X_train, y_train, X_eval, y_eval = _sequence_regression_split()
    result = run_fn(
        X_train, y_train, X_eval, y_eval, arch_fn(), _config(), _import=_raise_import_error
    )
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.training is None
    assert result.evaluation is None


@pytest.mark.parametrize(
    ("run_fn", "arch_fn"),
    [
        (run_cnn_modeling, _cnn_arch),
        (run_lstm_modeling, _lstm_arch),
        (run_transformer_modeling, _transformer_arch),
    ],
)
def test_architecture_task_type_mismatch_returns_failed_without_torch(run_fn, arch_fn):
    X_train, y_train, X_eval, y_eval = _sequence_regression_split()
    arch = arch_fn(task_type=TaskType.REGRESSION)
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION)
    result = run_fn(X_train, y_train, X_eval, y_eval, arch, config, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.FAILED
    assert "task_type" in (result.reason or "")
    assert result.training is None


def test_2d_input_fails_cleanly_for_sequence_architectures():
    # the shared X/y must already be 3D for CNN/LSTM/Transformer; a plain
    # 2D matrix fails tensor conversion rather than being silently reshaped
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(10, 4))
    y_train = rng.normal(size=10)
    result = run_cnn_modeling(X_train, y_train, X_train, y_train, _cnn_arch(), _config())
    assert result.status is TrainingRunStatus.FAILED
    assert "tensor conversion" in (result.reason or "")


# --- full execution per architecture (real torch) -------------------------


def test_cnn_regression_full_execution():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _sequence_regression_split(n_train=40, n_eval=15)
    result = run_cnn_modeling(X_train, y_train, X_eval, y_eval, _cnn_arch(), _config())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.sample_count == 15
    assert {"rmse", "mae"} <= set(result.evaluation.metrics)


def test_lstm_regression_full_execution():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _sequence_regression_split(n_train=40, n_eval=15)
    # LSTM/Transformer expect (batch, seq_len, input_size); the shared helper
    # builds (batch, channels, seq_len) for CNN, so transpose the trailing
    # two axes for this architecture's own convention.
    X_train_t = np.transpose(X_train, (0, 2, 1))
    X_eval_t = np.transpose(X_eval, (0, 2, 1))
    result = run_lstm_modeling(X_train_t, y_train, X_eval_t, y_eval, _lstm_arch(), _config())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.sample_count == 15


def test_transformer_regression_full_execution():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _sequence_regression_split(n_train=40, n_eval=15)
    X_train_t = np.transpose(X_train, (0, 2, 1))
    X_eval_t = np.transpose(X_eval, (0, 2, 1))
    result = run_transformer_modeling(
        X_train_t, y_train, X_eval_t, y_eval, _transformer_arch(), _config()
    )
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.sample_count == 15


# --- determinism -----------------------------------------------------------


def test_cnn_full_execution_is_deterministic():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _sequence_regression_split(n_train=30, n_eval=10, seed=3)
    arch = _cnn_arch()
    config = _config()

    result_a = run_cnn_modeling(X_train, y_train, X_eval, y_eval, arch, config)
    result_b = run_cnn_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result_a.model_dump_json() == result_b.model_dump_json()
