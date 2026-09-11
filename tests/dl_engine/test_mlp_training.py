"""Phase 8.3 — end-to-end integration: MLP architecture -> to_tensors() ->
train_model() -> DLTrainingResult.

Proves the first Phase-8 neural architecture trains through the existing
Phase-8.2 infrastructure with no infrastructure changes. Every test needs
real PyTorch and starts with `pytest.importorskip("torch")`.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPArchitectureConfig
from dl_engine.contracts import DLLoss, DLOptimizer, DLTrainingConfig, DLTrainingResult
from dl_engine.mlp import build_mlp
from dl_engine.runtime import seed_everything
from dl_engine.tensors import to_tensors
from dl_engine.training_loop import train_model


def _regression_data(n=32, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    true_w = rng.normal(size=f)
    y = X @ true_w
    return X, y


def _classification_data(n=32, f=4, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


def _training_config(**overrides: object) -> DLTrainingConfig:
    defaults: dict[str, object] = {
        "architecture_name": "mlp",
        "task_type": TaskType.REGRESSION,
        "epochs": 4,
        "batch_size": 8,
        "learning_rate": 0.05,
        "optimizer": DLOptimizer.ADAM,
        "loss": DLLoss.MSE,
        "seed": 42,
    }
    defaults.update(overrides)
    return DLTrainingConfig.model_validate(defaults)


# --- regression -------------------------------------------------------


def test_regression_end_to_end():
    torch = pytest.importorskip("torch")
    X, y = _regression_data(n=40, f=4)

    seed_everything(42)
    arch = MLPArchitectureConfig(
        task_type=TaskType.REGRESSION, input_features=4, output_dim=1, hidden_layer_sizes=[16]
    )
    model = build_mlp(arch)

    batch = to_tensors(X, y, TaskType.REGRESSION)
    assert batch.features.shape == (40, 4)
    assert batch.targets.shape == (40, 1)

    config = _training_config(task_type=TaskType.REGRESSION, loss=DLLoss.MSE)
    result = train_model(model, batch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.epochs_completed == config.epochs
    assert len(result.loss_history) == config.epochs
    assert result.final_loss is not None

    # output shape sanity: forward pass on the whole batch
    with torch.no_grad():
        out = model(batch.features)
    assert out.shape == (40, 1)


# --- binary classification ---------------------------------------------


def test_binary_classification_end_to_end():
    torch = pytest.importorskip("torch")
    X, y = _classification_data(n=40, f=4, n_classes=2)

    seed_everything(42)
    arch = MLPArchitectureConfig(
        task_type=TaskType.BINARY_CLASSIFICATION,
        input_features=4,
        output_dim=2,
        hidden_layer_sizes=[16],
    )
    model = build_mlp(arch)

    batch = to_tensors(X, y, TaskType.BINARY_CLASSIFICATION)
    assert batch.targets.dtype == torch.int64

    config = _training_config(task_type=TaskType.BINARY_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY)
    result = train_model(model, batch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert len(result.loss_history) == config.epochs

    with torch.no_grad():
        out = model(batch.features)
    assert out.shape == (40, 2)  # two-logit CrossEntropyLoss convention


# --- multiclass classification -------------------------------------------


def test_multiclass_classification_end_to_end():
    torch = pytest.importorskip("torch")
    n_classes = 4
    X, y = _classification_data(n=48, f=5, n_classes=n_classes)

    seed_everything(42)
    arch = MLPArchitectureConfig(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        input_features=5,
        output_dim=n_classes,
        hidden_layer_sizes=[24, 12],
    )
    model = build_mlp(arch)

    batch = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    config = _training_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        loss=DLLoss.CROSS_ENTROPY,
        epochs=3,
    )
    result = train_model(model, batch, config)

    assert result.status is TrainingRunStatus.COMPLETED
    assert len(result.loss_history) == 3

    with torch.no_grad():
        out = model(batch.features)
    assert out.shape == (48, n_classes)


# --- determinism ---------------------------------------------------------


def test_full_pipeline_is_deterministic_across_independently_seeded_runs():
    torch = pytest.importorskip("torch")
    X, y = _regression_data(n=32, f=3, seed=1)
    arch = MLPArchitectureConfig(
        task_type=TaskType.REGRESSION, input_features=3, output_dim=1, hidden_layer_sizes=[10]
    )
    config = _training_config(task_type=TaskType.REGRESSION, epochs=5, batch_size=8)

    seed_everything(config.seed)
    model_a = build_mlp(arch)
    initial_params_a = [p.detach().clone() for p in model_a.parameters()]
    batch_a = to_tensors(X, y, TaskType.REGRESSION)
    result_a = train_model(model_a, batch_a, config)

    seed_everything(config.seed)
    model_b = build_mlp(arch)
    initial_params_b = [p.detach().clone() for p in model_b.parameters()]
    batch_b = to_tensors(X, y, TaskType.REGRESSION)
    result_b = train_model(model_b, batch_b, config)

    # identical initial parameters
    for pa, pb in zip(initial_params_a, initial_params_b, strict=True):
        assert torch.equal(pa, pb)

    # identical loss histories / final loss / structured result serialization
    assert result_a.loss_history == result_b.loss_history
    assert result_a.final_loss == result_b.final_loss
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_classification_pipeline_deterministic():
    pytest.importorskip("torch")
    X, y = _classification_data(n=30, f=3, n_classes=3, seed=2)
    arch = MLPArchitectureConfig(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        input_features=3,
        output_dim=3,
        hidden_layer_sizes=[12],
    )
    config = _training_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY, epochs=4
    )

    seed_everything(config.seed)
    model_a = build_mlp(arch)
    batch_a = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    result_a = train_model(model_a, batch_a, config)

    seed_everything(config.seed)
    model_b = build_mlp(arch)
    batch_b = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    result_b = train_model(model_b, batch_b, config)

    assert result_a.loss_history == result_b.loss_history
    assert result_a.model_dump_json() == result_b.model_dump_json()


# --- boundary behavior -----------------------------------------------------


def test_incompatible_feature_dimensions_returns_failed():
    pytest.importorskip("torch")
    X, y = _regression_data(n=20, f=6)  # 6 features
    arch = MLPArchitectureConfig(
        task_type=TaskType.REGRESSION,
        input_features=4,  # mismatched on purpose
        output_dim=1,
        hidden_layer_sizes=[8],
    )
    seed_everything(42)
    model = build_mlp(arch)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _training_config(task_type=TaskType.REGRESSION)
    result = train_model(model, batch, config)

    assert result.status is TrainingRunStatus.FAILED
    assert result.reason is not None and len(result.reason) > 0


def test_invalid_class_count_returns_failed():
    pytest.importorskip("torch")
    # 4 actual classes but the model is only built for 2 (CrossEntropyLoss will
    # see a target index >= output_dim and raise).
    X, y = _classification_data(n=40, f=4, n_classes=4)
    arch = MLPArchitectureConfig(
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        input_features=4,
        output_dim=2,
        hidden_layer_sizes=[8],
    )
    seed_everything(42)
    model = build_mlp(arch)
    batch = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    config = _training_config(
        task_type=TaskType.MULTICLASS_CLASSIFICATION, loss=DLLoss.CROSS_ENTROPY
    )
    result = train_model(model, batch, config)

    assert result.status is TrainingRunStatus.FAILED
    assert result.reason is not None and len(result.reason) > 0


def test_invalid_architecture_configuration_raises_before_torch():
    # architecture validation happens at MLPArchitectureConfig construction,
    # entirely independent of build_mlp / train_model
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MLPArchitectureConfig(
            task_type=TaskType.REGRESSION, input_features=4, output_dim=1, hidden_layer_sizes=[]
        )


def test_invalid_task_model_combination_raises_before_torch():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MLPArchitectureConfig(
            task_type=TaskType.BINARY_CLASSIFICATION,
            input_features=4,
            output_dim=5,  # binary requires exactly 2
            hidden_layer_sizes=[8],
        )


def test_config_task_type_mismatch_with_batch_returns_failed_without_torch_needed():
    # This exercises the existing 8.2 train_model guard — proving the MLP path
    # doesn't bypass it. No torch import required for this specific check.
    def _raise_import_error(name: str):
        raise ImportError(f"No module named '{name}'")

    from dl_engine.tensors import TensorBatch

    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    config = _training_config(task_type=TaskType.MULTICLASS_CLASSIFICATION)
    result = train_model(object(), fake_batch, config, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.FAILED


def test_result_is_json_serialisable_and_round_trips():
    pytest.importorskip("torch")
    X, y = _regression_data(n=20, f=3)
    arch = MLPArchitectureConfig(
        task_type=TaskType.REGRESSION, input_features=3, output_dim=1, hidden_layer_sizes=[8]
    )
    seed_everything(42)
    model = build_mlp(arch)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    config = _training_config(task_type=TaskType.REGRESSION, epochs=2)
    result = train_model(model, batch, config)

    payload = result.model_dump_json()
    restored = DLTrainingResult.model_validate_json(payload)
    assert restored == result
