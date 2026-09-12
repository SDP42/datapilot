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

**Phase 8.4** adds :class:`DLEvaluationResult` — the structured result of
evaluating an *already-trained* model on explicitly supplied evaluation
data, produced by :func:`dl_engine.evaluation.evaluate_model`. It reuses
the Phase-7 metric vocabulary and semantics
(:mod:`data_engine.modeling.training`'s ``rmse`` / ``mae`` / ``r2`` for
regression and ``accuracy`` / ``precision`` / ``recall`` / ``f1`` /
``roc_auc`` — macro-averaged, ``zero_division=0``, ``roc_auc`` only for a
binary task with both classes present — for classification) rather than
a competing metric framework. A **new, additive** contract: distinct from
:class:`~data_engine.modeling.EvaluationResults` (Phase 7's status
mirror of a *set* of classical candidate runs) and from
:class:`DLTrainingResult` (raw training-loop execution — no metric is
computed there at all). ``status`` reuses
:class:`~data_engine.modeling.TrainingRunStatus` for the same reason
``DLTrainingResult.status`` does.

**Phase 8.5** adds :class:`DLModelingResult` — the small aggregate result
of one complete single-model DL run
(:func:`dl_engine.execution.run_mlp_modeling`: build → train →
evaluate). It **nests** the existing :class:`DLTrainingResult` /
:class:`DLEvaluationResult` rather than duplicating any of their fields —
the same "reference, don't flatten" pattern
:class:`~data_engine.modeling.ModelingSpec` already uses for its own
``training`` / ``evaluation`` sections. Its own ``status`` is resolved
from the two nested statuses (``completed`` only when both training and
evaluation completed), mirroring
:func:`data_engine.modeling.pipeline._resolve_overall_status`'s
stage-naming-the-failure convention without importing that private
function across the package boundary.

**Phase 8.6** adds :class:`DLCandidate`, :class:`DLCandidateRank`, and
:class:`DLSelectionResult` — a deterministic, DL-only selection layer
comparing multiple already-specified Phase-8 configurations against each
other (never against a Phase-7 classical model; that comparison is out
of scope). ``DLCandidate`` pairs an existing
:class:`~dl_engine.architectures.MLPArchitectureConfig` with an existing
:class:`DLTrainingConfig` — no new configuration vocabulary. Its
identity (``DLCandidateRank.candidate_id``) is a deterministic digest of
both configs' own JSON — never a random UUID or an experiment id.
``DLCandidateRank`` **nests** the existing :class:`DLModelingResult`
rather than duplicating any of its fields, the same pattern
``DLModelingResult`` itself already uses for ``DLTrainingResult`` /
``DLEvaluationResult``. The selection metric and direction reuse the
**exact** per-task values :mod:`data_engine.modeling.selection` already
established (``rmse`` / minimize for regression, ``f1`` / maximize for
binary and multiclass classification) — not a separate DL-specific
metric. ``DLSelectionResult.status`` reuses
:class:`~data_engine.modeling.TrainingRunStatus`: ``completed`` once at
least one candidate is eligible and one is selected, ``failed`` when no
supplied candidate was eligible (a deliberate, documented difference
from Phase 7's own :class:`~data_engine.modeling.ModelSelection`, which
reports ``completed`` even with nothing selected — Phase 8's own
:class:`DLTrainingResult` / :class:`DLEvaluationResult` /
:class:`DLModelingResult` already use ``failed`` to mean "this call did
not produce a usable outcome," and this keeps that convention
consistent across every Phase-8 contract rather than introducing Phase
7's separate three-state ``ModelingStatus`` vocabulary for one contract).
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from data_engine.modeling import ModelFamily, TrainingRunStatus
from data_engine.problem_understanding import TaskType

from .architectures import MLPArchitectureConfig

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


class DLEvaluationResult(BaseModel):
    """The structured result of evaluating an already-trained model.

    Produced by :func:`dl_engine.evaluation.evaluate_model` against
    explicitly supplied evaluation data — never a held-out partition the
    evaluator sources itself. Contains **only** information Phase 8.4
    actually produces: no timestamp, UUID, experiment id, artifact path,
    tensor, fitted model object, optimizer object, gradient, or
    explainability information.

    ``metrics`` uses the Phase-7 metric vocabulary and rounding
    (:mod:`data_engine.modeling.training`); a mathematically undefined
    metric (e.g. ``roc_auc`` when the evaluation data contains only one
    class) is **omitted** from ``metrics``, matching the existing Phase-7
    convention — never a fabricated ``NaN`` masquerading as a valid
    score. ``primary_metric`` names — but does not compute a ranking
    from — the metric most representative of the task; it is purely
    descriptive and is never consumed by model selection (Phase 8.4
    performs none).
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed (metrics computed), unavailable (PyTorch was not available — "
        "nothing was attempted), or failed (evaluation raised an error, or the model/data were "
        "incompatible)."
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    architecture_name: str | None = Field(
        default=None,
        description="Caller-supplied architecture label (e.g. 'mlp'), purely descriptive; None "
        "if not supplied.",
    )
    sample_count: int = Field(
        default=0, description="Rows evaluated; 0 when evaluation did not run."
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Task-appropriate metric values in the Phase-7 vocabulary and rounding; "
        "empty when status is not completed. A mathematically undefined metric is omitted, "
        "never NaN.",
    )
    primary_metric: str | None = Field(
        default=None,
        description="'rmse' for regression, 'f1' for binary/multiclass classification — "
        "descriptive only, never used for model selection.",
    )
    reason: str | None = Field(
        default=None, description="Why status is unavailable / failed; None when completed."
    )
    notes: list[str] = Field(default_factory=list)


class DLModelingResult(BaseModel):
    """The aggregate result of one complete single-model DL run.

    Produced by :func:`dl_engine.execution.run_mlp_modeling`: build the
    configured MLP, train it on the supplied training data, evaluate it
    on the supplied (**separate**) evaluation data. Contains no runtime
    object (no fitted model, tensor, optimizer, or gradient) and no
    timestamp, UUID, or MLflow / experiment identifier — experiment
    identity is explicitly Phase 9's concern, out of scope here.

    ``training`` / ``evaluation`` **nest** the existing
    :class:`DLTrainingResult` / :class:`DLEvaluationResult` rather than
    duplicating their fields; either is ``None`` when that stage never
    ran (e.g. evaluation never runs after a failed training stage).
    ``status`` is resolved from the two nested statuses — ``completed``
    only when both training and evaluation completed; otherwise
    ``unavailable`` / ``failed`` with ``reason`` naming which stage
    stopped the run.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed only when both training and evaluation completed; otherwise "
        "mirrors whichever stage first reported unavailable / failed."
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    family: ModelFamily = Field(
        default=ModelFamily.NEURAL,
        description="Always NEURAL — reuses the existing Phase-7 ModelFamily vocabulary.",
    )
    architecture_name: str | None = Field(
        default=None, description="Caller-supplied architecture label (e.g. 'mlp'); descriptive."
    )
    training: DLTrainingResult | None = Field(
        default=None, description="The training-stage result; None if training never ran."
    )
    evaluation: DLEvaluationResult | None = Field(
        default=None,
        description="The evaluation-stage result; None if evaluation never ran (e.g. training "
        "did not complete).",
    )
    reason: str | None = Field(
        default=None,
        description="Which stage stopped the run and why; None when status is completed.",
    )
    notes: list[str] = Field(default_factory=list)


class DLCandidate(BaseModel):
    """One concrete Phase-8 model configuration to execute and compare.

    Pairs an existing :class:`~dl_engine.architectures.MLPArchitectureConfig`
    with an existing :class:`DLTrainingConfig` — no new configuration
    vocabulary is introduced. ``candidate_id`` is a deterministic digest
    of both configs' own JSON (never a random UUID or experiment id), so
    the same configuration always yields the same identity and two
    candidates with identical configuration are indistinguishable by
    design — exactly as they should be.
    """

    model_config = ConfigDict(protected_namespaces=())

    architecture: MLPArchitectureConfig
    training_config: DLTrainingConfig

    @property
    def candidate_id(self) -> str:
        """A deterministic identity derived from this candidate's own configuration."""
        digest_input = (
            f"{self.architecture.model_dump_json()}|{self.training_config.model_dump_json()}"
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]


class DLCandidateRank(BaseModel):
    """One candidate's place in a :class:`DLSelectionResult`'s ranking.

    **Nests** the existing :class:`DLModelingResult` rather than
    duplicating any of its fields — the full training / evaluation
    detail for this candidate lives there. ``rank`` is ``1``-based for
    an eligible candidate and ``None`` for one that is not (failed,
    unavailable, missing the selection metric, or carrying a
    non-finite metric value) — mirroring
    :class:`~data_engine.modeling.ModelSelectionRank`'s exact
    eligible/ineligible convention. ``score`` is populated **only** for
    an eligible candidate; it is the existing evaluation metric value,
    never recomputed.
    """

    model_config = ConfigDict(protected_namespaces=())

    candidate_id: str = Field(description="This candidate's deterministic configuration digest.")
    architecture_name: str | None = Field(
        default=None, description="This candidate's architecture label; purely descriptive."
    )
    status: TrainingRunStatus = Field(description="The candidate's DLModelingResult.status.")
    score: float | None = Field(
        default=None, description="The selection-metric value for an eligible candidate; else None."
    )
    metric: str | None = Field(default=None, description="The selection metric name; else None.")
    rank: int | None = Field(
        default=None, description="1-based rank among eligible candidates; else None."
    )
    reason: str = Field(description="Why the candidate is eligible, or why it is not selectable.")
    modeling_result: DLModelingResult | None = Field(
        default=None, description="The full build/train/evaluate result for this candidate."
    )


class DLSelectionResult(BaseModel):
    """A deterministic DL-only comparison of multiple :class:`DLCandidate` runs.

    Produced by :func:`dl_engine.selection.select_dl_models`, which
    **executes nothing itself** — it calls the existing
    :func:`dl_engine.execution.run_mlp_modeling` once per candidate and
    only compares the results. Selection metric and direction reuse the
    **exact** per-task values already established by
    :mod:`data_engine.modeling.selection` (``rmse`` / minimize for
    regression, ``f1`` / maximize for binary and multiclass
    classification) — never a DL-specific metric or a different
    averaging semantic. Compares Phase-8 DL candidates against each
    other **only** — never against a Phase-7 classical model; that
    comparison is out of scope for this contract.

    ``status`` is ``completed`` once at least one candidate was eligible
    and one was selected; ``failed`` when no supplied candidate was
    eligible (every candidate remains visible in ``ranking`` either way,
    with a ``reason`` explaining its ineligibility).
    """

    model_config = ConfigDict(protected_namespaces=())

    status: TrainingRunStatus = Field(
        description="completed once a candidate was selected; failed when none was eligible."
    )
    task_type: TaskType = Field(
        description="regression, binary_classification, or multiclass_classification."
    )
    family: ModelFamily = Field(
        default=ModelFamily.NEURAL,
        description="Always NEURAL — reuses the existing Phase-7 ModelFamily vocabulary.",
    )
    selection_metric: str | None = Field(
        default=None, description="The fixed selection metric for the task."
    )
    selection_direction: str | None = Field(
        default=None, description="'minimize' or 'maximize' for the selection metric."
    )
    ranking: list[DLCandidateRank] = Field(
        default_factory=list,
        description="Every supplied candidate's selection standing, deterministically ordered "
        "(eligible candidates first, by rank, then ineligible candidates).",
    )
    selected_candidate_id: str | None = Field(
        default=None, description="The winning candidate's id; None when nothing is selectable."
    )
    selected_architecture_name: str | None = Field(
        default=None, description="The winning candidate's architecture label; descriptive only."
    )
    selected_score: float | None = Field(
        default=None, description="The winning candidate's selection-metric value."
    )
    reason: str | None = Field(
        default=None, description="Why status is failed / why nothing was selected."
    )
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "DL_ENGINE_VERSION",
    "DLCandidate",
    "DLCandidateRank",
    "DLDevice",
    "DLEvaluationResult",
    "DLLoss",
    "DLModelingResult",
    "DLOptimizer",
    "DLSelectionResult",
    "DLTrainingConfig",
    "DLTrainingResult",
    "DLTrainingStatus",
]
