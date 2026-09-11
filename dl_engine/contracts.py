"""Phase 8.1 / 8.2 — the deterministic DL training contracts.

Pydantic v2, JSON round-trip safe, JSON-primitive only — no tensor, NumPy
array, fitted `torch.nn.Module`, timestamp, UUID, or filesystem-specific
runtime state. Importing this module never imports PyTorch (see
:mod:`dl_engine.availability`).

**Phase 8.1** defined the configuration contract only —
:class:`DLTrainingConfig` — before anything trained. It deliberately
reuses the existing Phase-7 :class:`~data_engine.modeling.ModelFamily`
vocabulary (``NEURAL``) and Phase-5
:class:`~data_engine.problem_understanding.TaskType` rather than
inventing a parallel classification of models or tasks.

**Phase 8.2** adds :class:`DLTrainingResult` — the structured result of
actually running :func:`dl_engine.training_loop.train_model` once. It is
a **new, additive contract**, not a modification of any existing Phase-7
contract: :class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` evaluate a *set* of
candidate model families against a held-out test partition, while
:class:`DLTrainingResult` reports only the raw training-loop execution of
*one already-constructed* model — no evaluation, no candidate ranking, no
test-set metric. Integrating a completed DL run into the existing
`TrainingRun` / `TrainingOutcome` contracts is deliberately deferred to a
later increment; `DLTrainingResult.status` reuses the existing
:class:`~data_engine.modeling.TrainingRunStatus` vocabulary
(``completed`` / ``unavailable`` / ``failed``) for consistency rather
than inventing a fourth status enum.

``DLTrainingConfig.status`` defaults to ``not_yet_started`` — constructing
a config is a declaration of intent, never a claim that training ran.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from data_engine.modeling import ModelFamily, TrainingRunStatus
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


class DLTrainingResult(BaseModel):
    """The structured result of one :func:`dl_engine.training_loop.train_model` call.

    Contains **only** information Phase 8.2 actually produces: no
    evaluation against a held-out / test set, no experiment id, no
    artifact path, no timestamp, no UUID, no fitted model object, no
    tensor. Deterministic and JSON-serialisable — repeated training runs
    with the same `DLTrainingConfig`, model, and data produce a
    byte-identical result (see the Phase-8.2 determinism tests).

    This is a **new, additive** contract — it does not modify
    :class:`~data_engine.modeling.TrainingRun` /
    :class:`~data_engine.modeling.TrainingOutcome`, which remain Phase
    7's evaluation contracts. ``status`` reuses the existing
    :class:`~data_engine.modeling.TrainingRunStatus` enum.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed (trained for the full requested epochs), unavailable (PyTorch or "
        "the requested device was not available — nothing was attempted), or failed (training "
        "raised an error partway through)."
    )
    family: ModelFamily = Field(
        default=ModelFamily.NEURAL,
        description="Always NEURAL — mirrors DLTrainingConfig.family.",
    )
    device_used: DLDevice | None = Field(
        default=None,
        description="The device training actually ran on; None when status is not completed.",
    )
    epochs_requested: int = Field(description="DLTrainingConfig.epochs at the time of the call.")
    epochs_completed: int = Field(
        default=0, description="Number of epochs that finished; < epochs_requested on failure."
    )
    batch_size: int = Field(description="DLTrainingConfig.batch_size at the time of the call.")
    learning_rate: float = Field(
        description="DLTrainingConfig.learning_rate at the time of the call."
    )
    optimizer: DLOptimizer = Field(
        description="DLTrainingConfig.optimizer at the time of the call."
    )
    loss: DLLoss = Field(description="DLTrainingConfig.loss at the time of the call.")
    seed: int = Field(description="DLTrainingConfig.seed at the time of the call.")
    deterministic_mode: bool = Field(
        description="DLTrainingConfig.deterministic_mode at the time of the call."
    )
    loss_history: list[float] = Field(
        default_factory=list,
        description="Mean training loss per completed epoch, in epoch order; empty when no "
        "epoch completed.",
    )
    final_loss: float | None = Field(
        default=None, description="loss_history[-1]; None when no epoch completed."
    )
    reason: str | None = Field(
        default=None, description="Why status is unavailable / failed; None when completed."
    )
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "DL_ENGINE_VERSION",
    "DLDevice",
    "DLLoss",
    "DLOptimizer",
    "DLTrainingConfig",
    "DLTrainingResult",
    "DLTrainingStatus",
]
