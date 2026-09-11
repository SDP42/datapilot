"""Phase 8.2 — the dataset-to-tensor boundary (`dl_engine.tensors`).

Validation (shape / dtype / row-count / finiteness) is pure NumPy and
runs in every environment. Tests that need a real converted tensor skip
cleanly when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_engine.problem_understanding import TaskType
from dl_engine.tensors import to_tensors


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _Xy(n=10, f=3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, f)).astype(np.float64)
    y = rng.normal(size=n).astype(np.float64)
    return X, y


# --- validation: torch-free, runs everywhere ----------------------------


def test_non_ndarray_features_rejected():
    _, y = _Xy()
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_tensors([[1.0, 2.0]], y, TaskType.REGRESSION)


def test_non_ndarray_targets_rejected():
    X, _ = _Xy()
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_tensors(X, [1.0, 2.0], TaskType.REGRESSION)


def test_1d_features_rejected():
    X = np.arange(10, dtype=np.float64)
    y = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="2D feature matrix"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_2d_targets_rejected():
    X, _ = _Xy()
    y = np.zeros((10, 1))
    with pytest.raises(ValueError, match="1D target vector"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_empty_input_rejected():
    X = np.empty((0, 3))
    y = np.empty((0,))
    with pytest.raises(ValueError, match="no rows"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_zero_feature_columns_rejected():
    X = np.empty((5, 0))
    y = np.zeros(5)
    with pytest.raises(ValueError, match="no rows or no features"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_inconsistent_row_counts_rejected():
    X, y = _Xy(n=10)
    y_short = y[:8]
    with pytest.raises(ValueError, match="row counts must match"):
        to_tensors(X, y_short, TaskType.REGRESSION)


def test_non_numeric_feature_dtype_rejected():
    X = np.array([["a", "b"], ["c", "d"]], dtype=object)
    y = np.zeros(2)
    with pytest.raises(ValueError, match="numeric array"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_non_numeric_target_dtype_rejected():
    X, _ = _Xy(n=2)
    y = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="numeric array"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_nan_in_features_rejected():
    X, y = _Xy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN / Inf"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_inf_in_targets_rejected():
    X, y = _Xy()
    y[0] = np.inf
    with pytest.raises(ValueError, match="NaN / Inf"):
        to_tensors(X, y, TaskType.REGRESSION)


def test_unsupported_task_type_rejected():
    X, y = _Xy()
    with pytest.raises(ValueError, match="unsupported task_type"):
        to_tensors(X, y, TaskType.CLUSTERING)


def test_validation_runs_without_torch_installed():
    # empty input is rejected before torch is ever imported
    X = np.empty((0, 3))
    y = np.empty((0,))
    with pytest.raises(ValueError):
        to_tensors(X, y, TaskType.REGRESSION, _import=_raise_import_error)


def test_torch_missing_raises_runtime_error_after_validation_passes():
    X, y = _Xy()
    with pytest.raises(RuntimeError, match="PyTorch"):
        to_tensors(X, y, TaskType.REGRESSION, _import=_raise_import_error)


# --- source data non-mutation (torch-free: validated before conversion) --


def test_source_arrays_not_mutated_on_validation_path():
    X, y = _Xy()
    X_before = X.copy()
    y_before = y.copy()
    try:
        to_tensors(X, y, TaskType.REGRESSION, _import=_raise_import_error)
    except RuntimeError:
        pass
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


# --- real-torch conversion tests (skip cleanly if unavailable) ----------


def test_regression_tensor_shapes_and_dtypes():
    torch = pytest.importorskip("torch")
    X, y = _Xy(n=10, f=4)
    batch = to_tensors(X, y, TaskType.REGRESSION)
    assert batch.features.shape == (10, 4)
    assert batch.features.dtype == torch.float32
    assert batch.targets.shape == (10, 1)
    assert batch.targets.dtype == torch.float32
    assert batch.n_rows == 10
    assert batch.n_features == 4
    assert batch.task_type is TaskType.REGRESSION


def test_binary_classification_tensor_shapes_and_dtypes():
    torch = pytest.importorskip("torch")
    X, _ = _Xy(n=8, f=3)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
    batch = to_tensors(X, y, TaskType.BINARY_CLASSIFICATION)
    assert batch.features.shape == (8, 3)
    assert batch.targets.shape == (8,)
    assert batch.targets.dtype == torch.int64


def test_multiclass_classification_tensor_shapes_and_dtypes():
    torch = pytest.importorskip("torch")
    X, _ = _Xy(n=9, f=2)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    batch = to_tensors(X, y, TaskType.MULTICLASS_CLASSIFICATION)
    assert batch.features.shape == (9, 2)
    assert batch.targets.shape == (9,)
    assert batch.targets.dtype == torch.int64
    assert int(batch.targets.max()) == 2


def test_row_order_preserved():
    pytest.importorskip("torch")
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    y = np.array([10.0, 20.0, 30.0, 40.0])
    batch = to_tensors(X, y, TaskType.REGRESSION)
    np.testing.assert_array_equal(batch.features.numpy(), X.astype(np.float32))
    np.testing.assert_array_equal(batch.targets.numpy().ravel(), y.astype(np.float32))


def test_conversion_does_not_mutate_source_arrays():
    pytest.importorskip("torch")
    X, y = _Xy()
    X_before = X.copy()
    y_before = y.copy()
    batch = to_tensors(X, y, TaskType.REGRESSION)
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)
    # mutating the returned tensor must not affect the source array either
    batch.features[0, 0] = 999.0
    assert X[0, 0] != 999.0


def test_deterministic_repeated_conversion():
    pytest.importorskip("torch")
    X, y = _Xy()
    a = to_tensors(X, y, TaskType.REGRESSION)
    b = to_tensors(X, y, TaskType.REGRESSION)
    import torch

    assert torch.equal(a.features, b.features)
    assert torch.equal(a.targets, b.targets)
