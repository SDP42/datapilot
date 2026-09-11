"""Phase 8.5 — deep learning modeling pipeline integration.

:func:`run_mlp_modeling` is the single, coherent modeling-facing entry
point that chains the existing Phase-8 components into one complete
single-model run::

    prepared modeling data
            |
    MLP configuration (MLPArchitectureConfig)
            |
    seed (DLTrainingConfig.seed) -> seed_everything()
            |
    build_mlp()
            |
    to_tensors() (training data)
            |
    train_model()
            |
    to_tensors() (evaluation data)
            |
    evaluate_model()
            |
    DLModelingResult

It is a **composition layer only** — it calls the existing Phase-8.2/8.3/
8.4 functions and duplicates none of their logic. Training and
evaluation data are **always** two separate arrays the caller supplies
explicitly; this function never splits data itself and never evaluates
on the rows it trained on.

**No preprocessing happens here.** ``X_train`` / ``X_eval`` must already
be fully numeric — imputation, scaling, encoding, feature engineering,
and lag / rolling feature construction remain the Phase 6.5 / 7.4
boundary, exactly as :func:`dl_engine.tensors.to_tensors` already
requires.

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

import numpy as np

from data_engine.modeling import TrainingRunStatus
from data_engine.problem_understanding import TaskType

from .architectures import MLPArchitectureConfig
from .contracts import DLEvaluationResult, DLModelingResult, DLTrainingConfig, DLTrainingResult
from .evaluation import evaluate_model
from .mlp import build_mlp
from .runtime import seed_everything
from .tensors import to_tensors
from .training_loop import train_model


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
    **already-separate** arrays — this function never splits data itself
    and never evaluates on rows it trained on. Both must already satisfy
    the Phase 6.5 / 7.4 preprocessing boundary (fully numeric, no
    missing values) — see :func:`dl_engine.tensors.to_tensors`.

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
            TrainingRunStatus.UNAVAILABLE,
            task_type,
            architecture_name,
            "PyTorch is not installed; the Phase-8 MLP execution path cannot run",
        )

    try:
        model = build_mlp(architecture, _import=_import)
    except RuntimeError as exc:
        return _incomplete(TrainingRunStatus.UNAVAILABLE, task_type, architecture_name, str(exc))

    try:
        train_batch = to_tensors(X_train, y_train, task_type, _import=_import)
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
        eval_batch = to_tensors(X_eval, y_eval, task_type, _import=_import)
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


__all__ = ["run_mlp_modeling"]
