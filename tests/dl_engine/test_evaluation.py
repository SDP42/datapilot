"""Phase 8.4 — deep learning evaluation (`dl_engine.evaluation.evaluate_model`).

Every test that actually runs a model starts with `pytest.importorskip
("torch")` and skips cleanly when PyTorch is not installed. A few
environment-independent failure-path tests (PyTorch missing, empty
batch) run in every environment via the injectable `_import` seam.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine.architectures import MLPArchitectureConfig
from dl_engine.contracts import DLEvaluationResult, DLLoss, DLOptimizer, DLTrainingConfig
from dl_engine.evaluation import evaluate_model
from dl_engine.mlp import build_mlp
from dl_engine.runtime import seed_everything
from dl_engine.tensors import TensorBatch, to_tensors
from dl_engine.training_loop import train_model


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _regression_data(n=40, f=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    true_w = rng.normal(size=f)
    y = X @ true_w
    return X, y


def _classification_data(n=40, f=4, n_classes=2, seed=0, balanced=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float64)
    if balanced:
        y = np.array([i % n_classes for i in range(n)], dtype=np.int64)
        rng.shuffle(y)
    else:
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
    return X, y


def _train(task_type, output_dim, input_features, n_train=40, loss=None, seed=42, n_classes=None):
    if task_type is TaskType.REGRESSION:
        X, y = _regression_data(n=n_train, f=input_features, seed=seed)
        loss = loss or DLLoss.MSE
    else:
        X, y = _classification_data(
            n=n_train, f=input_features, n_classes=n_classes or output_dim, seed=seed
        )
        loss = loss or DLLoss.CROSS_ENTROPY

    seed_everything(seed)
    arch = MLPArchitectureConfig(
        task_type=task_type,
        input_features=input_features,
        output_dim=output_dim,
        hidden_layer_sizes=[16],
    )
    model = build_mlp(arch)
    batch = to_tensors(X, y, task_type)
    config = DLTrainingConfig(
        architecture_name="mlp",
        task_type=task_type,
        epochs=5,
        batch_size=8,
        learning_rate=0.05,
        optimizer=DLOptimizer.ADAM,
        loss=loss,
        seed=seed,
    )
    result = train_model(model, batch, config)
    assert result.status is TrainingRunStatus.COMPLETED
    return model


# --- environment-independent failure paths -------------------------------


def test_unavailable_when_torch_missing():
    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    result = evaluate_model(object(), fake_batch, _import=_raise_import_error)
    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.reason is not None
    assert result.sample_count == 0


def test_empty_batch_returns_failed():
    empty_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=0, n_features=3
    )
    result = evaluate_model(object(), empty_batch)
    assert result.status is TrainingRunStatus.FAILED
    assert "zero rows" in (result.reason or "")


def test_result_is_json_serialisable_on_unavailable_path():
    fake_batch = TensorBatch(
        features=object(), targets=object(), task_type=TaskType.REGRESSION, n_rows=1, n_features=1
    )
    result = evaluate_model(object(), fake_batch, _import=_raise_import_error)
    restored = DLEvaluationResult.model_validate_json(result.model_dump_json())
    assert restored == result


# --- regression ------------------------------------------------------------


def test_regression_evaluation():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=4)
    X_eval, y_eval = _regression_data(n=15, f=4, seed=99)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)

    result = evaluate_model(model, eval_batch, architecture_name="mlp")

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.sample_count == 15
    assert result.primary_metric == "rmse"
    assert set(result.metrics) >= {"rmse", "mae"}
    assert result.metrics["rmse"] >= 0.0
    assert result.metrics["mae"] >= 0.0
    # r2 is present since y_eval has nonzero variance
    assert "r2" in result.metrics


# --- binary classification --------------------------------------------------


def test_binary_classification_evaluation_both_classes_present():
    pytest.importorskip("torch")
    model = _train(TaskType.BINARY_CLASSIFICATION, output_dim=2, input_features=4, n_classes=2)
    X_eval, y_eval = _classification_data(n=20, f=4, n_classes=2, seed=7, balanced=True)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.BINARY_CLASSIFICATION)

    result = evaluate_model(model, eval_batch, architecture_name="mlp")

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.sample_count == 20
    assert result.primary_metric == "f1"
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert key in result.metrics
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0


def test_binary_classification_one_class_roc_auc_omitted():
    pytest.importorskip("torch")
    model = _train(TaskType.BINARY_CLASSIFICATION, output_dim=2, input_features=4, n_classes=2)
    rng = np.random.default_rng(3)
    X_eval = rng.normal(size=(10, 4))
    y_eval = np.zeros(10, dtype=np.int64)  # only class 0 present
    eval_batch = to_tensors(X_eval, y_eval, TaskType.BINARY_CLASSIFICATION)

    result = evaluate_model(model, eval_batch)

    assert result.status is TrainingRunStatus.COMPLETED
    assert "roc_auc" not in result.metrics
    assert any("roc_auc" in n and "one class" in n for n in result.notes)
    # the other metrics are still computed
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in result.metrics


# --- multiclass classification ----------------------------------------------


def test_multiclass_classification_evaluation():
    pytest.importorskip("torch")
    n_classes = 4
    model = _train(
        TaskType.MULTICLASS_CLASSIFICATION,
        output_dim=n_classes,
        input_features=5,
        n_classes=n_classes,
    )
    X_eval, y_eval = _classification_data(n=24, f=5, n_classes=n_classes, seed=11)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.MULTICLASS_CLASSIFICATION)

    result = evaluate_model(model, eval_batch, architecture_name="mlp")

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.sample_count == 24
    assert result.primary_metric == "f1"
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1"}
    assert "roc_auc" not in result.metrics  # never computed for > 2 classes


def test_prediction_derivation_uses_argmax_of_logits():
    torch = pytest.importorskip("torch")
    model = _train(TaskType.MULTICLASS_CLASSIFICATION, output_dim=3, input_features=3, n_classes=3)
    X_eval, y_eval = _classification_data(n=12, f=3, n_classes=3, seed=5)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.MULTICLASS_CLASSIFICATION)

    with torch.no_grad():
        logits = model(eval_batch.features)
    expected_predictions = logits.argmax(dim=1).numpy()
    y_true = eval_batch.targets.numpy()
    from sklearn.metrics import accuracy_score

    expected_accuracy = round(float(accuracy_score(y_true, expected_predictions)), 6)

    result = evaluate_model(model, eval_batch)
    assert result.metrics["accuracy"] == expected_accuracy


# --- gradient / mode / non-mutation ----------------------------------------


def test_evaluation_leaves_pretraining_grad_unchanged():
    # Training leaves whatever .grad its last backward() step produced — that's
    # normal (nothing zeroes .grad after the loop ends). The invariant to prove
    # is that evaluate_model computes no *new* gradients, i.e. .grad is
    # bit-identical before and after the call — not that .grad is zero/None,
    # which would be a false assumption about post-training state.
    torch = pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    grads_before = [None if p.grad is None else p.grad.clone() for p in model.parameters()]

    X_eval, y_eval = _regression_data(n=10, f=3, seed=2)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
    evaluate_model(model, eval_batch)

    grads_after = [p.grad for p in model.parameters()]
    for before, after in zip(grads_before, grads_after, strict=True):
        if before is None:
            assert after is None
        else:
            assert torch.equal(before, after)


def test_evaluation_disables_gradient_computation():
    # Directly proves the model's forward pass runs with torch.is_grad_enabled()
    # False during evaluate_model — i.e. genuinely under torch.no_grad(), not
    # merely happening to leave old .grad values alone.
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    X_eval, y_eval = _regression_data(n=10, f=3, seed=2)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)

    import torch

    grad_enabled_during_forward: list[bool] = []
    original_forward = model.forward

    def spy_forward(*args, **kwargs):
        grad_enabled_during_forward.append(torch.is_grad_enabled())
        return original_forward(*args, **kwargs)

    model.forward = spy_forward
    try:
        result = evaluate_model(model, eval_batch)
    finally:
        model.forward = original_forward

    assert result.status.value == "completed"
    assert grad_enabled_during_forward == [False]


def test_model_parameters_unchanged_after_evaluation():
    torch = pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    before = [p.detach().clone() for p in model.parameters()]

    X_eval, y_eval = _regression_data(n=10, f=3, seed=8)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
    evaluate_model(model, eval_batch)

    after = list(model.parameters())
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a)


def test_model_mode_restored_after_evaluation_when_previously_training():
    model = None
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    model.train()  # explicitly set to training mode before evaluation
    assert model.training is True

    X_eval, y_eval = _regression_data(n=10, f=3, seed=4)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
    evaluate_model(model, eval_batch)

    assert model.training is True  # restored, not left in eval mode


def test_model_mode_restored_after_evaluation_when_previously_eval():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    model.eval()
    assert model.training is False

    X_eval, y_eval = _regression_data(n=10, f=3, seed=6)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
    evaluate_model(model, eval_batch)

    assert model.training is False


def test_mode_restored_even_on_failure():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    model.train()

    # mismatched feature count -> forward pass raises -> failed result
    bad_batch = to_tensors(*_regression_data(n=10, f=99, seed=1), TaskType.REGRESSION)
    result = evaluate_model(model, bad_batch)

    assert result.status is TrainingRunStatus.FAILED
    assert model.training is True  # still restored despite the failure


def test_source_arrays_not_mutated_by_evaluation():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    X_eval, y_eval = _regression_data(n=10, f=3, seed=9)
    X_before, y_before = X_eval.copy(), y_eval.copy()
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
    evaluate_model(model, eval_batch)
    np.testing.assert_array_equal(X_eval, X_before)
    np.testing.assert_array_equal(y_eval, y_before)


# --- determinism -------------------------------------------------------------


def test_repeated_evaluation_is_deterministic():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=4)
    X_eval, y_eval = _regression_data(n=20, f=4, seed=13)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)

    result_a = evaluate_model(model, eval_batch, architecture_name="mlp")
    result_b = evaluate_model(model, eval_batch, architecture_name="mlp")

    assert result_a.metrics == result_b.metrics
    assert result_a.primary_metric == result_b.primary_metric
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_classification_evaluation_deterministic():
    pytest.importorskip("torch")
    model = _train(TaskType.MULTICLASS_CLASSIFICATION, output_dim=3, input_features=3, n_classes=3)
    X_eval, y_eval = _classification_data(n=15, f=3, n_classes=3, seed=21)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.MULTICLASS_CLASSIFICATION)

    result_a = evaluate_model(model, eval_batch)
    result_b = evaluate_model(model, eval_batch)
    assert result_a.model_dump_json() == result_b.model_dump_json()


# --- boundary / failure behavior ---------------------------------------------


def test_invalid_model_type_returns_failed():
    pytest.importorskip("torch")
    eval_batch = to_tensors(*_regression_data(n=5, f=3), TaskType.REGRESSION)
    result = evaluate_model(object(), eval_batch)
    assert result.status is TrainingRunStatus.FAILED
    assert "torch.nn.Module" in (result.reason or "")


def test_incompatible_feature_dimension_returns_failed():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=4)
    X_eval, y_eval = _regression_data(n=10, f=7, seed=1)  # wrong number of features
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)

    result = evaluate_model(model, eval_batch)
    assert result.status is TrainingRunStatus.FAILED
    assert result.reason is not None and len(result.reason) > 0


def test_invalid_class_index_returns_failed():
    pytest.importorskip("torch")
    # model only has 2 output classes but eval targets go up to 3
    model = _train(TaskType.BINARY_CLASSIFICATION, output_dim=2, input_features=4, n_classes=2)
    X_eval, _ = _classification_data(n=10, f=4, n_classes=2, seed=2)
    y_eval = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.MULTICLASS_CLASSIFICATION)

    result = evaluate_model(model, eval_batch)
    assert result.status is TrainingRunStatus.FAILED
    assert "class index" in (result.reason or "") or "out of range" in (result.reason or "")


def test_never_shuffles_evaluation_row_order():
    pytest.importorskip("torch")
    model = _train(TaskType.REGRESSION, output_dim=1, input_features=3)
    X_eval = np.arange(30, dtype=np.float64).reshape(10, 3)
    y_eval = np.arange(10, dtype=np.float64)
    eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)

    # row order in the tensor matches the row order of X_eval exactly
    np.testing.assert_array_equal(eval_batch.features.numpy(), X_eval.astype(np.float32))
    evaluate_model(model, eval_batch)
    # to_tensors output is untouched by evaluation, still in original order
    np.testing.assert_array_equal(eval_batch.features.numpy(), X_eval.astype(np.float32))
