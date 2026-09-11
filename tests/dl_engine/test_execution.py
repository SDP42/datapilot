"""Phase 8.5 — the complete single-model DL execution path
(`dl_engine.execution.run_mlp_modeling`).

Environment-independent failure-path tests (PyTorch missing, mismatched
task types) run in every environment via the injectable `_import` seam.
Every test that actually trains/evaluates starts with
`pytest.importorskip("torch")` and skips cleanly when PyTorch is not
installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPArchitectureConfig
from dl_engine.contracts import DLDevice, DLLoss, DLModelingResult, DLOptimizer, DLTrainingConfig
from dl_engine.execution import run_mlp_modeling


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _regression_split(n_train=32, n_eval=12, f=4, seed=0):
    rng = np.random.default_rng(seed)
    true_w = rng.normal(size=f)
    X_train = rng.normal(size=(n_train, f)).astype(np.float64)
    y_train = X_train @ true_w
    X_eval = rng.normal(size=(n_eval, f)).astype(np.float64) + 100.0  # distinguishable range
    y_eval = X_eval @ true_w
    return X_train, y_train, X_eval, y_eval


def _classification_split(n_train=32, n_eval=12, f=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, f)).astype(np.float64)
    y_train = np.array([i % n_classes for i in range(n_train)], dtype=np.int64)
    rng.shuffle(y_train)
    X_eval = rng.normal(size=(n_eval, f)).astype(np.float64) + 100.0  # distinguishable range
    y_eval = np.array([i % n_classes for i in range(n_eval)], dtype=np.int64)
    rng.shuffle(y_eval)
    return X_train, y_train, X_eval, y_eval


def _arch(**overrides: object) -> MLPArchitectureConfig:
    defaults: dict[str, object] = {
        "task_type": TaskType.REGRESSION,
        "input_features": 4,
        "output_dim": 1,
        "hidden_layer_sizes": [16],
    }
    defaults.update(overrides)
    return MLPArchitectureConfig.model_validate(defaults)


def _config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {
        "architecture_name": "mlp",
        "task_type": TaskType.REGRESSION,
        "epochs": 5,
        "batch_size": 8,
        "learning_rate": 0.05,
        "optimizer": DLOptimizer.ADAM,
        "loss": DLLoss.MSE,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


# --- environment-independent failure paths ------------------------------


def test_unavailable_when_torch_missing():
    X_train, y_train, X_eval, y_eval = _regression_split()
    result = run_mlp_modeling(
        X_train, y_train, X_eval, y_eval, _arch(), _config(), _import=_raise_import_error
    )
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.training is None
    assert result.evaluation is None
    assert result.reason is not None


def test_architecture_task_type_mismatch_returns_failed_without_torch():
    X_train, y_train, X_eval, y_eval = _regression_split()
    arch = _arch(task_type=TaskType.REGRESSION)
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION)
    result = run_mlp_modeling(
        X_train, y_train, X_eval, y_eval, arch, config, _import=_raise_import_error
    )
    assert result.status is TrainingRunStatus.FAILED
    assert "task_type" in (result.reason or "")
    assert result.training is None


def test_result_is_json_serialisable_on_unavailable_path():
    X_train, y_train, X_eval, y_eval = _regression_split()
    result = run_mlp_modeling(
        X_train, y_train, X_eval, y_eval, _arch(), _config(), _import=_raise_import_error
    )
    restored = DLModelingResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_requested_cuda_device_unavailable_with_real_torch():
    # Real PyTorch, requesting CUDA in an environment with no GPU (the sandbox
    # / CI default) — exercises the genuine device-resolution-inside-
    # train_model path without needing to fake torch's tensor machinery
    # (build_mlp / to_tensors need real nn.Linear / as_tensor numerics that a
    # hand-rolled fake torch module cannot meaningfully simulate).
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split()
    arch = _arch()  # device lives on the training config, not the architecture
    config = _config(device=DLDevice.CUDA)
    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)
    if result.status is TrainingRunStatus.COMPLETED:
        pytest.skip("CUDA is actually available in this environment")
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.training is not None
    assert "CUDA" in (result.training.reason or "")


# --- regression ------------------------------------------------------------


def test_regression_full_execution():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=40, n_eval=15, f=4)
    arch = _arch(task_type=TaskType.REGRESSION, input_features=4, output_dim=1)
    config = _config(task_type=TaskType.REGRESSION)

    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.task_type is TaskType.REGRESSION
    assert result.training is not None
    assert result.training.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.status is TrainingRunStatus.COMPLETED
    assert result.evaluation.sample_count == 15
    assert result.evaluation.primary_metric == "rmse"
    assert {"rmse", "mae"} <= set(result.evaluation.metrics)


# --- binary classification --------------------------------------------------


def test_binary_classification_full_execution():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _classification_split(
        n_train=40, n_eval=16, f=4, n_classes=2
    )
    arch = _arch(task_type=TaskType.BINARY_CLASSIFICATION, input_features=4, output_dim=2)
    config = _config(task_type=TaskType.BINARY_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY)

    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.sample_count == 16
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in result.evaluation.metrics
    assert 0.0 <= result.evaluation.metrics["roc_auc"] <= 1.0


# --- multiclass classification ----------------------------------------------


def test_multiclass_classification_full_execution():
    pytest.importorskip("torch")
    n_classes = 4
    X_train, y_train, X_eval, y_eval = _classification_split(
        n_train=48, n_eval=20, f=5, n_classes=n_classes
    )
    arch = _arch(
        task_type=TaskType.MULTICLASS_CLASSIFICATION, input_features=5, output_dim=n_classes
    )
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY)

    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert set(result.evaluation.metrics) == {"accuracy", "precision", "recall", "f1"}
    assert "roc_auc" not in result.evaluation.metrics


# --- data separation -----------------------------------------------------


def test_evaluation_uses_the_supplied_eval_data_not_training_data():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=30, n_eval=10, f=3)
    arch = _arch(task_type=TaskType.REGRESSION, input_features=3, output_dim=1)
    config = _config(task_type=TaskType.REGRESSION)

    captured_eval_inputs = []
    # execution.py did `from .evaluation import evaluate_model`, so the name
    # actually called is dl_engine.execution.evaluate_model — patching
    # dl_engine.evaluation.evaluate_model would not affect that bound
    # reference at all.
    import dl_engine.execution as execution_module

    original_evaluate = execution_module.evaluate_model

    def spy_evaluate(model, batch, **kwargs):
        captured_eval_inputs.append(batch.features.clone())
        return original_evaluate(model, batch, **kwargs)

    execution_module.evaluate_model = spy_evaluate
    try:
        run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)
    finally:
        execution_module.evaluate_model = original_evaluate

    assert len(captured_eval_inputs) == 1
    eval_features_used = captured_eval_inputs[0].numpy()
    np.testing.assert_array_equal(eval_features_used, X_eval.astype(np.float32))
    assert eval_features_used.shape[0] == X_eval.shape[0] != X_train.shape[0]


# --- failure paths (real torch) -------------------------------------------


def test_training_failure_skips_evaluation():
    pytest.importorskip("torch")
    # mismatched input_features vs actual X_train columns -> training fails inside train_model
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=20, n_eval=8, f=7)
    arch = _arch(task_type=TaskType.REGRESSION, input_features=4, output_dim=1)  # wrong
    config = _config(task_type=TaskType.REGRESSION)

    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result.status is TrainingRunStatus.FAILED
    assert result.training is not None
    assert result.training.status is TrainingRunStatus.FAILED
    assert result.evaluation is None  # evaluation never attempted


def test_invalid_tensor_conversion_returns_failed():
    pytest.importorskip("torch")
    X_train = np.array([["a", "b"], ["c", "d"]], dtype=object)
    y_train = np.zeros(2)
    X_eval, y_eval = X_train, y_train
    arch = _arch(task_type=TaskType.REGRESSION, input_features=2, output_dim=1)
    config = _config(task_type=TaskType.REGRESSION)

    result = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result.status is TrainingRunStatus.FAILED
    assert "tensor conversion" in (result.reason or "")
    assert result.training is None


# --- determinism -------------------------------------------------------


def test_full_execution_is_deterministic():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _regression_split(n_train=32, n_eval=12, f=3, seed=5)
    arch = _arch(task_type=TaskType.REGRESSION, input_features=3, output_dim=1)
    config = _config(task_type=TaskType.REGRESSION, epochs=6, batch_size=8)

    result_a = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)
    result_b = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result_a.training is not None and result_b.training is not None
    assert result_a.training.loss_history == result_b.training.loss_history
    assert result_a.training.final_loss == result_b.training.final_loss
    assert result_a.evaluation is not None and result_b.evaluation is not None
    assert result_a.evaluation.metrics == result_b.evaluation.metrics
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_classification_full_execution_is_deterministic():
    pytest.importorskip("torch")
    X_train, y_train, X_eval, y_eval = _classification_split(
        n_train=30, n_eval=12, f=3, n_classes=3
    )
    arch = _arch(task_type=TaskType.MULTICLASS_CLASSIFICATION, input_features=3, output_dim=3)
    config = _config(task_type=TaskType.MULTICLASS_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY)

    result_a = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)
    result_b = run_mlp_modeling(X_train, y_train, X_eval, y_eval, arch, config)

    assert result_a.model_dump_json() == result_b.model_dump_json()
