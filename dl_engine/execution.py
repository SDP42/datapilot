"""Phase 8.5 / 8.8 — deep learning modeling pipeline integration.

:func:`run_mlp_modeling` (Phase 8.5), :func:`run_cnn_modeling`,
:func:`run_lstm_modeling`, and :func:`run_transformer_modeling` (Phase
8.8) are the modeling-facing entry points that each chain the existing
Phase-8 components into one complete single-model run for their own
architecture::

    prepared modeling data
            |
    architecture configuration (e.g. MLPArchitectureConfig)
            |
    seed (DLTrainingConfig.seed) -> seed_everything()
            |
    build_<architecture>()
            |
    to_tensors() / to_sequence_tensors() (training data)
            |
    train_model()
            |
    to_tensors() / to_sequence_tensors() (evaluation data)
            |
    evaluate_model()
            |
    DLModelingResult

All four are **composition layers only** — each calls the existing
Phase-8.2/8.3/8.4/8.7 functions via one shared private helper
(``_run_dl_modeling``), so none of that logic is duplicated across the
four architectures. Training and evaluation data are **always** two
separate arrays the caller supplies explicitly; none of these functions
split data themselves or evaluate on the rows they trained on.

**No preprocessing happens here.** ``X_train`` / ``X_eval`` must already
be fully numeric — imputation, scaling, encoding, feature engineering,
and lag / rolling feature construction remain the Phase 6.5 / 7.4
boundary, exactly as :func:`dl_engine.tensors.to_tensors` /
:func:`dl_engine.tensors.to_sequence_tensors` already require.

**This is not a replacement for the Phase-7 modeling pipeline.**
``data_engine.modeling`` is not imported here beyond the same stable
Phase-7 contracts (`TaskType`, `ModelFamily`, `TrainingRunStatus`)
``dl_engine`` already reuses throughout; ``run_modeling_pipeline`` and
the scikit-learn ``ModelFamily.NEURAL`` baseline are untouched by this
module and never invoked from it. This is a deliberately separate,
opt-in Phase-8 entry point — nothing here changes what happens when a
caller uses the existing Phase-7 API.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType

from .architectures import (
    CNNArchitectureConfig,
    LSTMArchitectureConfig,
    MLPArchitectureConfig,
    TransformerArchitectureConfig,
)
from .cnn import build_cnn
from .contracts import DLEvaluationResult, DLModelingResult, DLTrainingConfig, DLTrainingResult
from .evaluation import evaluate_model
from .lstm import build_lstm
from .mlp import build_mlp
from .runtime import seed_everything
from .tensors import TensorBatch, to_sequence_tensors, to_tensors
from .training_loop import train_model
from .transformer import build_transformer

if TYPE_CHECKING:
    import torch


def _incomplete(
    status: TrainingRunStatus,
    task_type: TaskType,
    architecture_name: str | None,
    reason: str,
    *,
    training: DLTrainingResult | None = None,
    evaluation: DLEvaluationResult | None = None,
) -> DLModelingResult:
    return DLModelingResult(
        status=status,
        task_type=task_type,
        architecture_name=architecture_name,
        training=training,
        evaluation=evaluation,
        reason=reason,
    )


_ArchitectureConfig = (
    MLPArchitectureConfig
    | CNNArchitectureConfig
    | LSTMArchitectureConfig
    | TransformerArchitectureConfig
)


def _run_dl_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    architecture: _ArchitectureConfig,
    training_config: DLTrainingConfig,
    *,
    build_fn: Callable[..., torch.nn.Module],
    to_tensors_fn: Callable[..., TensorBatch],
    unavailable_reason: str,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLModelingResult:
    """Shared build -> train -> evaluate composition for one architecture family.

    Private to this module — every public ``run_*_modeling`` function below
    is a thin, architecture-specific wrapper around this one implementation
    (which build function, which tensor boundary, which "PyTorch missing"
    message to use); none of them duplicate this logic.
    """
    architecture_name = training_config.architecture_name

    if architecture.task_type != training_config.task_type:
        return _incomplete(
            TrainingRunStatus.FAILED,
            training_config.task_type,
            architecture_name,
            f"architecture.task_type ('{architecture.task_type.value}') does not match "
            f"training_config.task_type ('{training_config.task_type.value}')",
        )

    task_type = training_config.task_type

    seeded = seed_everything(
        training_config.seed, deterministic=training_config.deterministic_mode, _import=_import
    )
    if not seeded:
        return _incomplete(
            TrainingRunStatus.UNAVAILABLE, task_type, architecture_name, unavailable_reason
        )

    try:
        model = build_fn(architecture, _import=_import)
    except RuntimeError as exc:
        return _incomplete(TrainingRunStatus.UNAVAILABLE, task_type, architecture_name, str(exc))

    try:
        train_batch = to_tensors_fn(X_train, y_train, task_type, _import=_import)
    except (TypeError, ValueError) as exc:
        return _incomplete(
            TrainingRunStatus.FAILED,
            task_type,
            architecture_name,
            f"training-data tensor conversion failed: {exc}",
        )
    except RuntimeError as exc:
        return _incomplete(TrainingRunStatus.UNAVAILABLE, task_type, architecture_name, str(exc))

    training_result = train_model(model, train_batch, training_config, _import=_import)
    if training_result.status is not TrainingRunStatus.COMPLETED:
        return _incomplete(
            training_result.status,
            task_type,
            architecture_name,
            f"training stage did not complete: {training_result.reason}",
            training=training_result,
        )

    try:
        eval_batch = to_tensors_fn(X_eval, y_eval, task_type, _import=_import)
    except (TypeError, ValueError) as exc:
        return _incomplete(
            TrainingRunStatus.FAILED,
            task_type,
            architecture_name,
            f"evaluation-data tensor conversion failed: {exc}",
            training=training_result,
        )
    except RuntimeError as exc:
        return _incomplete(
            TrainingRunStatus.UNAVAILABLE,
            task_type,
            architecture_name,
            str(exc),
            training=training_result,
        )

    evaluation_result = evaluate_model(
        model, eval_batch, architecture_name=architecture_name, _import=_import
    )
    if evaluation_result.status is not TrainingRunStatus.COMPLETED:
        return _incomplete(
            evaluation_result.status,
            task_type,
            architecture_name,
            f"evaluation stage did not complete: {evaluation_result.reason}",
            training=training_result,
            evaluation=evaluation_result,
        )

    return DLModelingResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=task_type,
        architecture_name=architecture_name,
        training=training_result,
        evaluation=evaluation_result,
    )


def run_mlp_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    architecture: MLPArchitectureConfig,
    training_config: DLTrainingConfig,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLModelingResult:
    """Build, train, and evaluate one MLP in a single deterministic call.

    ``X_train`` / ``y_train`` and ``X_eval`` / ``y_eval`` must be two
    **already-separate**, 2D ``(n_rows, n_features)`` arrays — this
    function never splits data itself and never evaluates on rows it
    trained on. Both must already satisfy the Phase 6.5 / 7.4
    preprocessing boundary (fully numeric, no missing values) — see
    :func:`dl_engine.tensors.to_tensors`.

    ``architecture.task_type`` and ``training_config.task_type`` must
    agree — a mismatch is a configuration error and is reported as a
    structured ``failed`` result rather than silently preferring one.

    Never raises for an environment-level condition (PyTorch missing,
    the requested device unavailable, invalid/incompatible data, a
    training or evaluation failure) — every such condition is reported
    as a structured ``DLModelingResult`` with ``status`` and ``reason``
    naming the stage that stopped the run. A run is only ever reported
    ``completed`` when **both** training and evaluation completed;
    evaluation is never attempted after a training stage that did not
    complete. Nothing is retried with a different configuration, and no
    device silently falls back to another.
    """
    return _run_dl_modeling(
        X_train,
        y_train,
        X_eval,
        y_eval,
        architecture,
        training_config,
        build_fn=build_mlp,
        to_tensors_fn=to_tensors,
        unavailable_reason="PyTorch is not installed; the Phase-8 MLP execution path cannot run",
        _import=_import,
    )


def run_cnn_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    architecture: CNNArchitectureConfig,
    training_config: DLTrainingConfig,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLModelingResult:
    """Build, train, and evaluate one CNN in a single deterministic call.

    Phase-8.8 counterpart of :func:`run_mlp_modeling` for
    :class:`~dl_engine.architectures.CNNArchitectureConfig` /
    :func:`dl_engine.cnn.build_cnn`. ``X_train`` / ``X_eval`` must already
    be 3D ``(n_rows, input_channels, sequence_length)`` arrays — see
    :func:`dl_engine.tensors.to_sequence_tensors`; this function never
    reshapes a caller's 2D feature matrix into that shape. Every other
    guarantee (separate train/eval data, task-type agreement, structured
    ``unavailable`` / ``failed`` results, no retry, no device fallback) is
    identical to :func:`run_mlp_modeling`.
    """
    return _run_dl_modeling(
        X_train,
        y_train,
        X_eval,
        y_eval,
        architecture,
        training_config,
        build_fn=build_cnn,
        to_tensors_fn=to_sequence_tensors,
        unavailable_reason="PyTorch is not installed; the Phase-8 CNN execution path cannot run",
        _import=_import,
    )


def run_lstm_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    architecture: LSTMArchitectureConfig,
    training_config: DLTrainingConfig,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLModelingResult:
    """Build, train, and evaluate one LSTM in a single deterministic call.

    Phase-8.8 counterpart of :func:`run_mlp_modeling` for
    :class:`~dl_engine.architectures.LSTMArchitectureConfig` /
    :func:`dl_engine.lstm.build_lstm`. ``X_train`` / ``X_eval`` must
    already be 3D ``(n_rows, seq_len, input_size)`` arrays — see
    :func:`dl_engine.tensors.to_sequence_tensors`. Every other guarantee
    is identical to :func:`run_mlp_modeling`.
    """
    return _run_dl_modeling(
        X_train,
        y_train,
        X_eval,
        y_eval,
        architecture,
        training_config,
        build_fn=build_lstm,
        to_tensors_fn=to_sequence_tensors,
        unavailable_reason="PyTorch is not installed; the Phase-8 LSTM execution path cannot run",
        _import=_import,
    )


def run_transformer_modeling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    architecture: TransformerArchitectureConfig,
    training_config: DLTrainingConfig,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLModelingResult:
    """Build, train, and evaluate one Transformer encoder in a single deterministic call.

    Phase-8.8 counterpart of :func:`run_mlp_modeling` for
    :class:`~dl_engine.architectures.TransformerArchitectureConfig` /
    :func:`dl_engine.transformer.build_transformer`. ``X_train`` /
    ``X_eval`` must already be 3D ``(n_rows, seq_len, input_size)``
    arrays — see :func:`dl_engine.tensors.to_sequence_tensors`. Every
    other guarantee is identical to :func:`run_mlp_modeling`.
    """
    return _run_dl_modeling(
        X_train,
        y_train,
        X_eval,
        y_eval,
        architecture,
        training_config,
        build_fn=build_transformer,
        to_tensors_fn=to_sequence_tensors,
        unavailable_reason=(
            "PyTorch is not installed; the Phase-8 Transformer execution path cannot run"
        ),
        _import=_import,
    )


__all__ = ["run_cnn_modeling", "run_lstm_modeling", "run_mlp_modeling", "run_transformer_modeling"]
