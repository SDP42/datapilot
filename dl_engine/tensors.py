"""Phase 8.2 — the dataset-to-tensor boundary.

Converts **already-prepared, fully numeric** modeling data (a 2D feature
matrix + a 1D target vector — exactly what a Phase-6.5 / Phase-7.4
preprocessing pipeline already produces) into PyTorch tensors ready for
:func:`dl_engine.training_loop.train_model`.

This is deliberately a **narrow boundary, not a preprocessing engine**:
imputation, scaling, encoding, feature generation/selection, and
lag / rolling / calendar feature construction all remain Phase 6.5 /
Phase 7.4's job, executed **before** this module ever sees the data.
Nothing here mutates the source arrays, reorders rows, or accepts an
arbitrary Python object in place of a numeric array.

Only :class:`~data_engine.problem_understanding.TaskType` values with a
defined tensor convention are supported: ``regression`` (float32 target,
column-vector shape), ``binary_classification`` /
``multiclass_classification`` (``int64`` class-index target, ready for
``nn.CrossEntropyLoss``). Validation (shape, row-count, dtype, finiteness)
never requires PyTorch to be installed — only the final tensor
construction does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from data_engine.problem_understanding import TaskType

from .availability import torch_availability

if TYPE_CHECKING:
    import torch

_SUPPORTED_TASK_TYPES = (
    TaskType.REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
)


@dataclass(frozen=True)
class TensorBatch:
    """Converted, ready-to-train tensors for one dataset.

    **Not** a JSON-serialisable public contract — an internal container
    handed from :func:`to_tensors` to the training loop; a tensor is never
    stored in a Pydantic contract. ``features`` is always ``float32``
    shape ``(n_rows, n_features)``. ``targets`` is ``float32`` shape
    ``(n_rows, 1)`` for regression, or ``int64`` shape ``(n_rows,)`` for
    binary / multiclass classification (class indices).
    """

    features: torch.Tensor
    targets: torch.Tensor
    task_type: TaskType
    n_rows: int
    n_features: int


def _validate_inputs(
    X: np.ndarray, y: np.ndarray, task_type: TaskType, *, expected_ndim: int
) -> None:
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy.ndarray, got {type(X).__name__}")
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be a numpy.ndarray, got {type(y).__name__}")
    if X.ndim != expected_ndim:
        if expected_ndim == 2:
            raise ValueError(
                f"X must be a 2D feature matrix (n_rows, n_features); got shape {X.shape}"
            )
        raise ValueError(
            f"X must be a 3D sequence tensor (n_rows, dim1, dim2); got shape {X.shape}"
        )
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D target vector; got shape {y.shape}")
    if X.shape[0] == 0 or any(dim == 0 for dim in X.shape[1:]):
        if expected_ndim == 2:
            raise ValueError(f"X has no rows or no features (shape {X.shape}); nothing to convert")
        raise ValueError(
            f"X has no rows or an empty dimension (shape {X.shape}); nothing to convert"
        )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} rows; row counts must match"
        )
    if not np.issubdtype(X.dtype, np.number) or X.dtype == np.bool_:
        raise ValueError(
            f"X must be a numeric array; got dtype {X.dtype}. The DL tensor boundary accepts "
            "only already-prepared numeric modeling data — encoding / imputation remain Phase "
            "6.5 / 7.4's job."
        )
    if not np.issubdtype(y.dtype, np.number) or y.dtype == np.bool_:
        raise ValueError(f"y must be a numeric array; got dtype {y.dtype}")
    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN / Inf values; the DL tensor boundary expects fully prepared, "
            "finite numeric data (Phase 6.5 / 7.4 preprocessing must run first)"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains NaN / Inf values")
    if task_type not in _SUPPORTED_TASK_TYPES:
        raise ValueError(
            f"unsupported task_type for DL tensor conversion: '{task_type.value}'; supported: "
            f"{', '.join(t.value for t in _SUPPORTED_TASK_TYPES)}"
        )


def _build_targets(torch_module: ModuleType, y: np.ndarray, task_type: TaskType) -> torch.Tensor:
    if task_type is TaskType.REGRESSION:
        return torch_module.as_tensor(
            np.array(y, dtype=np.float32, copy=True), dtype=torch_module.float32
        ).unsqueeze(1)
    # binary_classification / multiclass_classification
    return torch_module.as_tensor(np.array(y, dtype=np.int64, copy=True), dtype=torch_module.long)


def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> TensorBatch:
    """Convert an already-prepared numeric ``(X, y)`` pair into a :class:`TensorBatch`.

    ``X`` must be a 2D feature matrix ``(n_rows, n_features)`` — see
    :func:`to_sequence_tensors` for 3D sequence input (CNN / LSTM /
    Transformer). Row order is preserved exactly — nothing here shuffles
    or reorders. ``X`` / ``y`` are copied before conversion, so the
    returned tensors never alias (and therefore never mutate) the
    caller's arrays.

    Raises ``TypeError`` / ``ValueError`` for invalid input — empty input,
    a shape / row-count mismatch, a non-numeric dtype, or non-finite
    values — **without requiring PyTorch to be installed** (validation is
    pure NumPy). Raises ``RuntimeError`` only once validation has passed,
    if PyTorch is not installed to actually build the tensors.
    """
    _validate_inputs(X, y, task_type, expected_ndim=2)

    availability = torch_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    torch_module = _import("torch")

    features = torch_module.as_tensor(
        np.array(X, dtype=np.float32, copy=True), dtype=torch_module.float32
    )
    targets = _build_targets(torch_module, y, task_type)

    return TensorBatch(
        features=features,
        targets=targets,
        task_type=task_type,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )


def to_sequence_tensors(
    X: np.ndarray,
    y: np.ndarray,
    task_type: TaskType,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> TensorBatch:
    """Convert an already-prepared numeric, 3D ``(X, y)`` pair into a :class:`TensorBatch`.

    The sequence-shaped counterpart of :func:`to_tensors`, for the
    Phase-8.7 CNN / LSTM / Transformer architectures: ``X`` must already
    be a 3D array — ``(n_rows, input_channels, sequence_length)`` for
    :func:`dl_engine.cnn.build_cnn`, or ``(n_rows, seq_len, input_size)``
    for :func:`dl_engine.lstm.build_lstm` /
    :func:`dl_engine.transformer.build_transformer`. This boundary never
    reshapes ``X`` to fit an architecture — a caller that needs a
    different layout reshapes before calling this function, exactly as
    every Phase-8.7 builder's own ``forward`` already requires and
    rejects a mismatch for.

    Row order is preserved exactly; ``X`` / ``y`` are copied before
    conversion, so the returned tensors never alias the caller's arrays.
    ``TensorBatch.n_features`` holds ``X.shape[-1]`` (the trailing
    dimension) — for CNN that is ``sequence_length``, not
    ``input_channels``; it is informational only and never consumed by
    :func:`dl_engine.training_loop.train_model` or
    :func:`dl_engine.evaluation.evaluate_model`.

    Raises ``TypeError`` / ``ValueError`` for invalid input — non-3D
    shape, empty input, a shape / row-count mismatch, a non-numeric
    dtype, or non-finite values — **without requiring PyTorch to be
    installed**. Raises ``RuntimeError`` only once validation has passed,
    if PyTorch is not installed to actually build the tensors.
    """
    _validate_inputs(X, y, task_type, expected_ndim=3)

    availability = torch_availability(_import=_import)
    if not availability.available:
        raise RuntimeError(availability.reason)

    torch_module = _import("torch")

    features = torch_module.as_tensor(
        np.array(X, dtype=np.float32, copy=True), dtype=torch_module.float32
    )
    targets = _build_targets(torch_module, y, task_type)

    return TensorBatch(
        features=features,
        targets=targets,
        task_type=task_type,
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[-1]),
    )
