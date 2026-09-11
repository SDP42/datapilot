"""Deep Learning (Phase 8) — PyTorch model definitions, training loops, evaluation.

**Phase 8.1 (this increment) is foundation-only: nothing is trained.**
It establishes:

* :mod:`dl_engine.availability` — a deterministic, lazy probe for whether
  PyTorch is installed (:func:`is_torch_available`,
  :func:`torch_availability`). PyTorch is an optional dependency (the
  ``dl`` extra) — importing ``dl_engine``, or any other DataPilot
  package, never requires it.
* :mod:`dl_engine.contracts` — :class:`DLTrainingConfig`, the
  deterministic, JSON-serialisable configuration contract a future
  training-execution increment will consume. Constructing one trains
  nothing.

``dl_engine`` integrates with the existing Phase-7
:class:`~data_engine.modeling.ModelFamily` (``NEURAL``) and Phase-5
:class:`~data_engine.problem_understanding.TaskType` vocabularies rather
than inventing a parallel one; a future increment that actually trains a
network is expected to populate the **existing**
:class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` contracts, not a second
modeling pipeline or evaluation contract.

    from dl_engine import is_torch_available, DLTrainingConfig

    if is_torch_available():
        config = DLTrainingConfig(architecture_name="mlp", task_type=TaskType.REGRESSION)

Out of scope for Phase 8.1 (and every later increment in this package
until explicitly implemented): model training, an MLP or any other
architecture, training loops, experiment tracking / ``ExperimentRecord``
(Phase 9), hyperparameter optimization, SHAP, deployment, and general
feature-engineering execution.
"""

from __future__ import annotations

from .availability import TorchAvailability, is_torch_available, torch_availability
from .contracts import (
    DL_ENGINE_VERSION,
    DLDevice,
    DLLoss,
    DLOptimizer,
    DLTrainingConfig,
    DLTrainingStatus,
)

__all__ = [
    "DL_ENGINE_VERSION",
    "DLDevice",
    "DLLoss",
    "DLOptimizer",
    "DLTrainingConfig",
    "DLTrainingStatus",
    "TorchAvailability",
    "is_torch_available",
    "torch_availability",
]
