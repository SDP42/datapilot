"""Phase 8.4 — deep learning evaluation foundation.

:func:`evaluate_model` evaluates an **already-trained** ``torch.nn.Module``
against explicitly supplied evaluation data — never training it, never
selecting between candidates, never sourcing a held-out partition itself.
Strict separation from training::

    training data -> to_tensors() -> train_model() -> trained model
                                                             |
                                    evaluation data -> to_tensors()
                                                             |
                                                   evaluate_model()
                                                             |
                                              DLEvaluationResult

There is no ``fit_and_evaluate()`` convenience function and this module
never calls :func:`dl_engine.training_loop.train_model`.

Metrics reuse the **exact** Phase-7 vocabulary, semantics, and rounding
computed by :mod:`data_engine.modeling.training` (the same sklearn
functions, the same macro-averaging / ``zero_division=0`` /
`MODEL_TRAINING_METRIC_ROUND` rounding, the same ``roc_auc`` gating on a
binary task with both classes present in the data) — this module does
not import those private functions across the package boundary (they are
internal to ``data_engine.modeling.training``), but it computes the
identical metrics the identical way, so the two never silently diverge.

Like every other ``dl_engine`` module, PyTorch is imported lazily — only
inside :func:`evaluate_model`, only when called.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from data_engine.modeling import MODEL_TRAINING_METRIC_ROUND, TrainingRunStatus
from data_engine.problem_understanding import TaskType

from .availability import torch_availability
from .contracts import DLEvaluationResult
from .tensors import TensorBatch

if TYPE_CHECKING:
    import torch

_CLASSIFICATION_TASKS = (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


def _round(value: float) -> float:
    if not np.isfinite(value):
        return float(value)
    return round(float(value), MODEL_TRAINING_METRIC_ROUND)


def _incomplete_result(
    status: TrainingRunStatus,
    task_type: TaskType,
    architecture_name: str | None,
    reason: str,
) -> DLEvaluationResult:
    return DLEvaluationResult(
        status=status,
        task_type=task_type,
        architecture_name=architecture_name,
        sample_count=0,
        reason=reason,
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics: dict[str, float] = {
        "rmse": _round(float(np.sqrt(mean_squared_error(y_true, y_pred)))),
        "mae": _round(float(mean_absolute_error(y_true, y_pred))),
    }
    if float(np.var(y_true)) > 0.0:
        metrics["r2"] = _round(float(r2_score(y_true, y_pred)))
    return metrics


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> tuple[dict[str, float], list[str]]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics: dict[str, float] = {
        "accuracy": _round(float(accuracy_score(y_true, y_pred))),
        "precision": _round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        ),
        "recall": _round(float(recall_score(y_true, y_pred, average="macro", zero_division=0))),
        "f1": _round(float(f1_score(y_true, y_pred, average="macro", zero_division=0))),
    }
    notes: list[str] = []
    if n_classes == 2:
        if proba is not None and len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score

            metrics["roc_auc"] = _round(float(roc_auc_score(y_true, proba)))
        else:
            notes.append(
                "roc_auc omitted: mathematically undefined — the evaluation data contains only "
                "one class"
            )
    return metrics, notes


def evaluate_model(
    model: torch.nn.Module,
    batch: TensorBatch,
    *,
    architecture_name: str | None = None,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLEvaluationResult:
    """Evaluate ``model`` on ``batch`` and return a :class:`DLEvaluationResult`.

    ``batch`` is evaluation data the caller explicitly built with
    :func:`dl_engine.tensors.to_tensors` — never a partition this function
    sources itself. Row order is never shuffled. The model's mode
    (``training`` / ``eval``) is restored to whatever it was **before**
    this call, whether evaluation succeeds or fails; inference runs
    inside ``torch.no_grad()`` so no computation graph is retained and no
    gradient is ever computed. Model parameters are never written to.

    Returns a structured ``unavailable`` result if PyTorch is not
    installed, or a structured ``failed`` result for an incompatible
    model/data pairing (wrong feature dimension, an out-of-range target
    class index, a non-``nn.Module`` model, non-finite predictions) —
    never a silently wrong or ``NaN`` metric.
    """
    if batch.n_rows == 0:
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            batch.task_type,
            architecture_name,
            "evaluation batch has zero rows; nothing to evaluate",
        )

    availability = torch_availability(_import=_import)
    if not availability.available:
        return _incomplete_result(
            TrainingRunStatus.UNAVAILABLE,
            batch.task_type,
            architecture_name,
            availability.reason or "",
        )

    torch = _import("torch")

    if not isinstance(model, torch.nn.Module):
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            batch.task_type,
            architecture_name,
            f"model must be a torch.nn.Module instance, got {type(model).__name__}",
        )

    was_training = model.training
    try:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        model.eval()
        with torch.no_grad():
            outputs = model(batch.features.to(device))

        if batch.task_type is TaskType.REGRESSION:
            if outputs.shape != batch.targets.shape:
                raise ValueError(
                    f"model output shape {tuple(outputs.shape)} does not match target shape "
                    f"{tuple(batch.targets.shape)} for regression evaluation"
                )
            y_pred = outputs.detach().cpu().numpy().reshape(-1)
            y_true = batch.targets.detach().cpu().numpy().reshape(-1)
            if not np.all(np.isfinite(y_pred)):
                raise ValueError("model predictions contain non-finite values (NaN / Inf)")
            metrics = _regression_metrics(y_true, y_pred)
            primary_metric = "rmse"
            notes: list[str] = []

        elif batch.task_type in _CLASSIFICATION_TASKS:
            if outputs.ndim != 2 or outputs.shape[0] != batch.n_rows:
                raise ValueError(
                    f"model output shape {tuple(outputs.shape)} is not a valid "
                    f"(n_rows, n_classes) logits tensor for {batch.n_rows} evaluation rows"
                )
            n_classes = int(outputs.shape[1])
            y_true = batch.targets.detach().cpu().numpy().reshape(-1)
            if y_true.size and (int(y_true.max()) >= n_classes or int(y_true.min()) < 0):
                raise ValueError(
                    f"target class index out of range: expected [0, {n_classes}), got "
                    f"[{int(y_true.min())}, {int(y_true.max())}]"
                )
            probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            y_pred = probabilities.argmax(axis=1)
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("model output produced non-finite probabilities (NaN / Inf)")
            proba = probabilities[:, 1] if n_classes == 2 else None
            metrics, class_notes = _classification_metrics(y_true, y_pred, proba, n_classes)
            primary_metric = "f1"
            notes = class_notes

        else:
            raise ValueError(f"unsupported task_type for DL evaluation: '{batch.task_type.value}'")
    except (RuntimeError, ValueError, IndexError) as exc:
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            batch.task_type,
            architecture_name,
            f"evaluation failed: {exc}",
        )
    finally:
        model.train(was_training)

    return DLEvaluationResult(
        status=TrainingRunStatus.COMPLETED,
        task_type=batch.task_type,
        architecture_name=architecture_name,
        sample_count=batch.n_rows,
        metrics=metrics,
        primary_metric=primary_metric,
        notes=notes,
    )


__all__ = ["evaluate_model"]
