"""Phase 8.2 — the minimal deterministic PyTorch training-loop foundation.

:func:`train_model` trains an **already-constructed** ``torch.nn.Module``
(Phase 8.2 defines no architecture — no MLP / CNN / LSTM / etc.) on an
already-converted :class:`~dl_engine.tensors.TensorBatch`, using an
existing :class:`~dl_engine.contracts.DLTrainingConfig` for every
hyperparameter. It performs training only:

* **no** evaluation against a held-out / test set (nothing here computes
  accuracy, RMSE, or any other metric),
* **no** model selection between candidates,
* **no** persistence, checkpoint, or artifact path,
* **no** experiment record / MLflow integration,
* **no** hyperparameter search.

Determinism is established by calling
:func:`dl_engine.runtime.seed_everything` with ``config.seed`` /
``config.deterministic_mode`` before the first forward / backward pass,
and by iterating batches in the fixed row order of the supplied
:class:`~dl_engine.tensors.TensorBatch` every epoch — batches are never
shuffled, so no additional RNG draw is needed to make batch order
reproducible. Note this seeding happens **after** the caller constructs
``model`` — this function trains an already-built module, so it cannot
make weight *initialization* reproducible by itself; a caller that also
needs deterministic initial weights must seed (e.g.
``dl_engine.runtime.seed_everything(config.seed)``) before constructing
the model, exactly as this module's own tests do. What ``train_model``
itself guarantees deterministic is everything it controls: optimizer
steps, batch order, and loss computation. No hidden global state is
introduced beyond the RNG seeding that PyTorch itself requires (see
:func:`dl_engine.runtime.seed_everything` for why that is unavoidable);
the loop keeps no module-level state and depends on nothing set outside
this function.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from data_engine.modeling import TrainingRunStatus

from .availability import torch_availability
from .contracts import DLDevice, DLLoss, DLOptimizer, DLTrainingConfig, DLTrainingResult
from .runtime import resolve_device, seed_everything
from .tensors import TensorBatch

if TYPE_CHECKING:
    import torch

_OPTIMIZER_FACTORY: dict[DLOptimizer, str] = {
    DLOptimizer.ADAM: "Adam",
    DLOptimizer.ADAMW: "AdamW",
    DLOptimizer.SGD: "SGD",
}
_LOSS_FACTORY: dict[DLLoss, str] = {
    DLLoss.MSE: "MSELoss",
    DLLoss.MAE: "L1Loss",
    DLLoss.CROSS_ENTROPY: "CrossEntropyLoss",
}


def _incomplete_result(
    status: TrainingRunStatus,
    config: DLTrainingConfig,
    reason: str,
    *,
    epochs_completed: int = 0,
    loss_history: list[float] | None = None,
    device_used: DLDevice | None = None,
) -> DLTrainingResult:
    history = loss_history or []
    return DLTrainingResult(
        status=status,
        family=config.family,
        device_used=device_used,
        epochs_requested=config.epochs,
        epochs_completed=epochs_completed,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        loss=config.loss,
        seed=config.seed,
        deterministic_mode=config.deterministic_mode,
        loss_history=history,
        final_loss=history[-1] if history else None,
        reason=reason,
    )


def train_model(
    model: torch.nn.Module,
    batch: TensorBatch,
    config: DLTrainingConfig,
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> DLTrainingResult:
    """Train ``model`` on ``batch`` for ``config.epochs`` epochs.

    Returns a :class:`DLTrainingResult`. Never raises for an
    environment-level condition (PyTorch missing, requested device
    unavailable, a task-type mismatch) — those are reported as a
    structured ``unavailable`` / ``failed`` result instead. A training
    failure partway through (e.g. the model's output shape is
    incompatible with ``batch.targets`` for the configured loss) is
    caught and reported as ``failed`` with the underlying error message
    as ``reason`` — never swallowed, never generalised away.
    """
    if config.task_type != batch.task_type:
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            config,
            f"config.task_type ('{config.task_type.value}') does not match the training "
            f"batch's task_type ('{batch.task_type.value}')",
        )

    availability = torch_availability(_import=_import)
    if not availability.available:
        return _incomplete_result(TrainingRunStatus.UNAVAILABLE, config, availability.reason or "")

    torch = _import("torch")

    if not isinstance(model, torch.nn.Module):
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            config,
            f"model must be a torch.nn.Module instance, got {type(model).__name__}",
        )

    device_resolution = resolve_device(config.device, _import=_import)
    if not device_resolution.available:
        return _incomplete_result(
            TrainingRunStatus.UNAVAILABLE, config, device_resolution.reason or ""
        )

    seed_everything(config.seed, deterministic=config.deterministic_mode, _import=_import)

    device = torch.device(device_resolution.device_str)
    model = model.to(device)
    features = batch.features.to(device)
    targets = batch.targets.to(device)

    optimizer_cls = getattr(torch.optim, _OPTIMIZER_FACTORY[config.optimizer])
    optimizer = optimizer_cls(model.parameters(), lr=config.learning_rate)
    criterion = getattr(torch.nn, _LOSS_FACTORY[config.loss])()

    n_rows = features.shape[0]
    batch_size = config.batch_size
    loss_history: list[float] = []

    try:
        model.train()
        for _epoch in range(config.epochs):
            epoch_loss_total = 0.0
            for start in range(0, n_rows, batch_size):
                end = min(start + batch_size, n_rows)
                x_batch = features[start:end]
                y_batch = targets[start:end]

                optimizer.zero_grad()
                output = model(x_batch)
                if config.loss in (DLLoss.MSE, DLLoss.MAE) and output.shape != y_batch.shape:
                    # MSELoss / L1Loss silently broadcast a mismatched shape instead of
                    # raising, which would otherwise train "successfully" against a
                    # meaningless broadcast — treat it as the shape-incompatibility
                    # failure it actually is rather than an accidental success.
                    raise ValueError(
                        f"model output shape {tuple(output.shape)} does not match target "
                        f"shape {tuple(y_batch.shape)} for loss '{config.loss.value}'"
                    )
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss_total += float(loss.item()) * (end - start)

            loss_history.append(epoch_loss_total / n_rows)
    except (RuntimeError, ValueError) as exc:
        return _incomplete_result(
            TrainingRunStatus.FAILED,
            config,
            f"training failed after {len(loss_history)} completed epoch(s): {exc}",
            epochs_completed=len(loss_history),
            loss_history=loss_history,
            device_used=device_resolution.resolved_device,
        )

    return DLTrainingResult(
        status=TrainingRunStatus.COMPLETED,
        family=config.family,
        device_used=device_resolution.resolved_device,
        epochs_requested=config.epochs,
        epochs_completed=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        loss=config.loss,
        seed=config.seed,
        deterministic_mode=config.deterministic_mode,
        loss_history=loss_history,
        final_loss=loss_history[-1] if loss_history else None,
    )
