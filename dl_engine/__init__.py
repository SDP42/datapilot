"""Deep Learning (Phase 8) — PyTorch model definitions, training loops, evaluation.

**Phase 8.1** established the foundation: nothing trained.

* :mod:`dl_engine.availability` — a deterministic, lazy probe for whether
  PyTorch is installed (:func:`is_torch_available`,
  :func:`torch_availability`). PyTorch is an optional dependency (the
  ``dl`` extra) — importing ``dl_engine``, or any other DataPilot
  package, never requires it.
* :mod:`dl_engine.contracts` — :class:`DLTrainingConfig`, the
  deterministic, JSON-serialisable configuration contract a training run
  consumes.

**Phase 8.2 (this increment)** adds the deterministic training
*foundation* — still no model architecture is defined here:

* :mod:`dl_engine.runtime` — :func:`seed_everything` (Python / NumPy /
  PyTorch seeding, CPU + CUDA-if-present, deterministic-algorithms mode)
  and :func:`resolve_device` (deterministic ``DLDevice`` → actual torch
  device resolution; a requested, unavailable accelerator is reported as
  a structured failure, never silently swapped for CPU).
* :mod:`dl_engine.tensors` — :func:`to_tensors`, the narrow
  dataset-to-tensor boundary: converts already-prepared numeric modeling
  data (never raw / unprocessed data — Phase 6.5 / 7.4 preprocessing
  remains the boundary) into PyTorch tensors for regression / binary /
  multiclass classification.
* :mod:`dl_engine.training_loop` — :func:`train_model`, the minimal
  training loop for an **already-constructed** ``torch.nn.Module``. No
  evaluation, no model selection, no persistence, no experiment tracking.
* :mod:`dl_engine.contracts` also gains :class:`DLTrainingResult` — the
  structured, JSON-serialisable, deterministic result of one
  :func:`~dl_engine.training_loop.train_model` call.

``dl_engine`` integrates with the existing Phase-7
:class:`~data_engine.modeling.ModelFamily` (``NEURAL``) and Phase-5
:class:`~data_engine.problem_understanding.TaskType` vocabularies rather
than inventing a parallel one; a future increment that wires a completed
DL run into the existing modeling pipeline is expected to populate the
**existing** :class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` contracts, not a second
modeling pipeline or evaluation contract.

    from dl_engine import is_torch_available, DLTrainingConfig, to_tensors, train_model

    if is_torch_available():
        config = DLTrainingConfig(architecture_name="mlp", task_type=TaskType.REGRESSION)
        batch = to_tensors(X, y, TaskType.REGRESSION)
        result = train_model(my_module, batch, config)

Out of scope for Phase 8.2 (and every later increment in this package
until explicitly implemented): any DL architecture (MLP / CNN / LSTM /
etc.), DL evaluation against a test set, model selection, experiment
tracking / ``ExperimentRecord`` (Phase 9), hyperparameter optimization,
SHAP, deployment, and general feature-engineering execution.
"""

from __future__ import annotations

from .availability import TorchAvailability, is_torch_available, torch_availability
from .contracts import (
    DL_ENGINE_VERSION,
    DLDevice,
    DLLoss,
    DLOptimizer,
    DLTrainingConfig,
    DLTrainingResult,
    DLTrainingStatus,
)
from .runtime import DeviceResolution, resolve_device, seed_everything
from .tensors import TensorBatch, to_tensors
from .training_loop import train_model

__all__ = [
    "DL_ENGINE_VERSION",
    "DLDevice",
    "DLLoss",
    "DLOptimizer",
    "DLTrainingConfig",
    "DLTrainingResult",
    "DLTrainingStatus",
    "DeviceResolution",
    "TensorBatch",
    "TorchAvailability",
    "is_torch_available",
    "resolve_device",
    "seed_everything",
    "to_tensors",
    "torch_availability",
    "train_model",
]
