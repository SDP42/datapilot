"""Phase 8.1 — the deterministic DL training configuration contract.

Pydantic v2, JSON round-trip safe, JSON-primitive only — no tensor, NumPy
array, fitted `torch.nn.Module`, timestamp, UUID, or filesystem-specific
runtime state. Importing this module never imports PyTorch (see
:mod:`dl_engine.availability`).

This module defines the **configuration contract only** — Phase 8.1 does
not train anything. It deliberately reuses the existing Phase-7
:class:`~data_engine.modeling.ModelFamily` vocabulary (``NEURAL``) and
Phase-5 :class:`~data_engine.problem_understanding.TaskType` rather than
inventing a parallel classification of models or tasks: a future Phase-8
increment that actually trains a network is expected to populate the
**existing** :class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` contracts (family=`NEURAL`),
not a second evaluation contract living here.

``DLTrainingConfig.status`` defaults to ``not_yet_started`` — constructing
a config is a declaration of intent, never a claim that training ran.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from data_engine.modeling import ModelFamily
from data_engine.problem_understanding import TaskType

DL_ENGINE_VERSION = "1"


class DLTrainingStatus(str, Enum):
    """Lifecycle of one :class:`DLTrainingConfig`.

    Phase 8.1 only ever produces ``NOT_YET_STARTED`` (the default) or
    ``UNAVAILABLE`` (e.g. PyTorch is not installed). ``COMPLETED`` /
    ``FAILED`` are declared now, mirroring the existing Phase-7
    ``TrainingRunStatus`` vocabulary, so a future training-execution
    increment is additive rather than requiring a contract change.
    """

    NOT_YET_STARTED = "not_yet_started"
    UNAVAILABLE = "unavailable"
    COMPLETED = "completed"
    FAILED = "failed"


class DLOptimizer(str, Enum):
    """Supported optimizers for a future Phase-8 training loop."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class DLLoss(str, Enum):
    """Supported loss functions for a future Phase-8 training loop."""

    MSE = "mse"
    MAE = "mae"
    CROSS_ENTROPY = "cross_entropy"


class DLDevice(str, Enum):
    """Supported compute devices for a future Phase-8 training loop."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DLTrainingConfig(BaseModel):
    """Deterministic configuration for a future Phase-8 PyTorch training run.

    Foundation-only: constructing this contract trains nothing. Every
    field is a JSON primitive / enum; no tensor, NumPy array, fitted
    module, timestamp, or UUID is ever stored here.
    """

    model_config = ConfigDict(protected_namespaces=())

    architecture_name: str = Field(
        description="Name of the DL architecture this configuration targets, e.g. 'mlp'. "
        "Declarative only — Phase 8.1 defines no architectures."
    )
    task_type: TaskType = Field(
        description="The Phase-5 task type this configuration targets (reuses TaskType rather "
        "than a parallel vocabulary)."
    )
    family: ModelFamily = Field(
        default=ModelFamily.NEURAL,
        description="Always NEURAL — reuses the existing Phase-7 ModelFamily vocabulary so a "
        "future training run integrates with TrainingRun.family rather than a parallel one.",
    )

    seed: int = Field(
        default=42,
        description="Fixed random seed for reproducibility, mirroring the Phase-7.4 "
        "MODEL_TRAINING_RANDOM_SEED convention.",
    )
    epochs: int = Field(default=1, ge=1, description="Number of training epochs.")
    batch_size: int = Field(default=32, ge=1, description="Mini-batch size.")
    learning_rate: float = Field(default=1e-3, gt=0, description="Optimizer learning rate.")
    optimizer: DLOptimizer = Field(default=DLOptimizer.ADAM)
    loss: DLLoss = Field(default=DLLoss.MSE)
    device: DLDevice = Field(
        default=DLDevice.CPU,
        description="Requested compute device; a future training increment decides whether it "
        "is actually available.",
    )
    deterministic_mode: bool = Field(
        default=True,
        description="Whether a future training loop must run in torch's deterministic-algorithms "
        "mode.",
    )

    status: DLTrainingStatus = DLTrainingStatus.NOT_YET_STARTED
    reason: str | None = Field(
        default=None,
        description="Why the configuration is unavailable (e.g. PyTorch not installed), or why "
        "a later training attempt failed; None otherwise.",
    )


__all__ = [
    "DL_ENGINE_VERSION",
    "DLDevice",
    "DLLoss",
    "DLOptimizer",
    "DLTrainingConfig",
    "DLTrainingStatus",
]
