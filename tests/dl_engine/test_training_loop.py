"""Phase 8.2 — the minimal deterministic training loop (`dl_engine.training_loop`).

Uses a tiny synthetic dataset and a tiny linear model throughout. The
PyTorch-unavailable / task-type-mismatch paths run in every environment
(no real torch needed, via the injectable `_import` seam); every test
that actually trains starts with `pytest.importorskip("torch")` and
skips cleanly when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import ModelFamily, TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.contracts import DLDevice, DLLoss, DLOptimizer, DLTrainingConfig, DLTrainingResult
from dl_engine.tensors import TensorBatch, to_tensors
from dl_engine.training_loop import train_model


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _regression_config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {
        "architecture_name": "linear",
        "task_type": TaskType.REGRESSION,
        "epochs": 5,
        "batch_size": 4,
        "learning_rate": 0.05,
        "optimizer": DLOptimizer.SGD,
        "loss": DLLoss.MSE,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


def _synthetic_regression(n=20, f=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w
    return X, y


def _synthetic_classification(n=20, f=3, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


# --- environment-level failure paths — run in every environment --------


def test_task_type_mismatch_returns_failed_without_touching_torch():
    config = _regression_config(task_type=TaskType.REGRESSION)
    fake_batch = TensorBatch(
        features=object(),
        targets=object(),
        task_type=TaskType.BINARY_CLASSIFICATION,
        n_rows=1,
        n_features=1,
    )
    result = train_model(object(), fake_batch, config, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.FAILED
    assert "task_type" in (result.reason or "")
    assert result.family is ModelFamily.NEURAL
    assert result.epochs_completed == 0


def test_unavailable_when_torch_missing():
    config = _regression_config()
    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    result = train_model(object(), fake_batch, config, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.reason is not None
    assert result.epochs_completed == 0
    assert result.device_used is None


def test_result_is_json_serialisable_on_unavailable_path():
    config = _regression_config()
    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    result = train_model(object(), fake_batch, config, _import=_raise_import_error)
    payload = result.model_dump_json()
    restored = DLTrainingResult.model_validate_json(payload)
    assert restored == result


def test_requested_cuda_device_unavailable_via_injection_returns_unavailable_result():
    # simulates "torch installed, CUDA not present" without needing real torch or hardware
    class _FakeModule:
        pass

    class _FakeNN:
        Module = _FakeModule

    class _FakeCuda:
        def is_available(self):
            return False

    class _FakeTorchModule:
        __version__ = "0.0.0-fake"
        cuda = _FakeCuda()
        nn = _FakeNN()

    def _import(name):
        assert name == "torch"
        return _FakeTorchModule()

    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    config = _regression_config(device=DLDevice.CUDA)
    result = train_model(_FakeModule(), fake_batch, config, _import=_import)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert "CUDA" in (result.reason or "")


# --- real-torch training tests (each skips cleanly if unavailable) -----


def test_epochs_and_batch_size_respected():
    torch = pytest.importorskip("torch")
    X, y = _synthetic_regression(n=17)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    model = torch.nn.Linear(3, 1)
    config = _regression_config(epochs=7, batch_size=5)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.epochs_requested == 7
    assert result.epochs_completed == 7
    assert len(result.loss_history) == 7
    assert result.batch_size == 5
    assert result.device_used is DLDevice.CPU


def test_training_reduces_loss_for_a_simple_linear_problem():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    X, y = _synthetic_regression(n=64)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(epochs=50, batch_size=16, learning_rate=0.05)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.final_loss is not None
    assert result.final_loss < result.loss_history[0]


def test_model_parameters_actually_change_after_training():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    before = [p.detach().clone() for p in model.parameters()]
    X, y = _synthetic_regression(n=32)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(epochs=5)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


@pytest.mark.parametrize("optimizer", [DLOptimizer.ADAM, DLOptimizer.ADAMW, DLOptimizer.SGD])
def test_optimizer_selection_works_for_every_supported_optimizer(optimizer):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    X, y = _synthetic_regression(n=20)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(optimizer=optimizer, epochs=3)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.optimizer is optimizer
    assert len(result.loss_history) == 3
    assert all(np.isfinite(v) for v in result.loss_history)


@pytest.mark.parametrize("loss", [DLLoss.MSE, DLLoss.MAE])
def test_loss_selection_works_for_regression_losses(loss):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    X, y = _synthetic_regression(n=20)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(loss=loss, epochs=3)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.loss is loss


def test_cross_entropy_loss_works_for_multiclass_classification():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    n_classes = 3
    model = torch.nn.Linear(3, n_classes)
    X, y = _synthetic_classification(n=24, n_classes=n_classes)
    batch = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    config = _regression_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        loss=DLLoss.CROSS_ENTROPY,
        epochs=3,
    )
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    assert len(result.loss_history) == 3


def test_learning_rate_affects_training_outcome():
    torch = pytest.importorskip("torch")

    def _run(lr):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 1)
        X, y = _synthetic_regression(n=32)
        batch = to_tensors(X, y, TaskType.REGRESSION)
        config = _regression_config(learning_rate=lr, epochs=3)
        return train_model(model, batch, config)

    low = _run(1e-4)
    high = _run(1.0)
    assert low.final_loss != high.final_loss


def test_model_training_incompatibility_returns_failed_with_context():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    # output has 2 columns but the regression target column is shape (n, 1)
    model = torch.nn.Linear(3, 2)
    X, y = _synthetic_regression(n=10)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(epochs=2)
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.FAILED
    assert result.reason is not None and len(result.reason) > 0


def test_result_json_round_trip_after_real_training():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    X, y = _synthetic_regression(n=20)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _regression_config(epochs=3)
    result = train_model(model, batch, config)
    payload = result.model_dump_json()
    restored = DLTrainingResult.model_validate_json(payload)
    assert restored == result


# --- determinism --------------------------------------------------------


def test_repeated_training_runs_are_deterministic():
    torch = pytest.importorskip("torch")
    from dl_engine.runtime import seed_everything

    X, y = _synthetic_regression(n=40)
    config = _regression_config(epochs=6, batch_size=8, learning_rate=0.05)

    seed_everything(config.seed)
    model_a = torch.nn.Linear(3, 1)
    batch_a = to_tensors(X, y, TaskType.REGRESSION)
    result_a = train_model(model_a, batch_a, config)

    seed_everything(config.seed)
    model_b = torch.nn.Linear(3, 1)
    batch_b = to_tensors(X, y, TaskType.REGRESSION)
    result_b = train_model(model_b, batch_b, config)

    assert result_a.loss_history == result_b.loss_history
    assert result_a.final_loss == result_b.final_loss
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_deterministic_tensor_conversion_feeding_into_training():
    torch = pytest.importorskip("torch")
    X, y = _synthetic_regression(n=15)
    batch_a = to_tensors(X, y, TaskType.REGRESSION)
    batch_b = to_tensors(X, y, TaskType.REGRESSION)
    assert torch.equal(batch_a.features, batch_b.features)
    assert torch.equal(batch_a.targets, batch_b.targets)
