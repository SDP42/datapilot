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

**Phase 8.2** added the deterministic training *foundation*:

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
* :mod:`dl_engine.contracts` also gained :class:`DLTrainingResult` — the
  structured, JSON-serialisable, deterministic result of one
  :func:`~dl_engine.training_loop.train_model` call.

**Phase 8.3** added the first Phase-8 neural *architecture*:

* :mod:`dl_engine.architectures` — :class:`MLPArchitectureConfig`, the
  deterministic, JSON-serialisable configuration for a small feed-forward
  MLP (input / output dimensions, hidden layer sizes, activation,
  dropout). A separate, additive contract from ``DLTrainingConfig``: this
  one says *what* to build, ``DLTrainingConfig`` says *how* to train it.
* :mod:`dl_engine.mlp` — :func:`build_mlp`, the **only** architecture
  builder in ``dl_engine`` so far: a fixed ``Linear`` + activation
  (+ optional ``Dropout``) stack, ending in a raw (no softmax/sigmoid)
  output layer sized for regression (``(n, 1)``), binary classification
  (``(n, 2)`` logits), or multiclass classification (``(n, num_classes)``
  logits) — matching the target conventions ``to_tensors`` already
  produces and the losses ``train_model`` already supports. A model built
  here plugs directly into the existing ``to_tensors`` → ``train_model``
  pipeline. A separate implementation from the Phase-7 scikit-learn MLP
  baseline (``data_engine.modeling``, ``ModelFamily.NEURAL``) — Phase 7
  is untouched.

**Phase 8.4 (this increment)** adds a deep learning evaluation
*foundation* — still no full modeling-pipeline integration, no model
selection:

* :mod:`dl_engine.evaluation` — :func:`evaluate_model`, evaluating an
  **already-trained** model against explicitly supplied evaluation data
  (never a held-out partition it sources itself). ``model.eval()`` +
  ``torch.no_grad()``; the model's original training/eval mode is
  restored afterward; parameters are never written to. Reuses the exact
  Phase-7 metric vocabulary / semantics / rounding (``rmse`` / ``mae`` /
  ``r2`` for regression; ``accuracy`` / ``precision`` / ``recall`` /
  ``f1`` / ``roc_auc`` — macro-averaged, binary-only ``roc_auc`` gated on
  both classes present — for classification).
* :mod:`dl_engine.contracts` also gained :class:`DLEvaluationResult` —
  the structured, JSON-serialisable, deterministic result of one
  :func:`~dl_engine.evaluation.evaluate_model` call. Strictly separate
  from training: this module never calls ``train_model`` and there is no
  ``fit_and_evaluate()`` convenience function.

``dl_engine`` integrates with the existing Phase-7
:class:`~data_engine.modeling.ModelFamily` (``NEURAL``) and Phase-5
:class:`~data_engine.problem_understanding.TaskType` vocabularies rather
than inventing a parallel one; a future increment that wires a completed
DL run into the existing modeling pipeline is expected to populate the
**existing** :class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` /
:class:`~data_engine.modeling.EvaluationResults` contracts, not a second
modeling pipeline.

    from dl_engine import (
        is_torch_available, DLTrainingConfig, MLPArchitectureConfig,
        build_mlp, to_tensors, train_model, evaluate_model,
    )

    if is_torch_available():
        arch = MLPArchitectureConfig(
            task_type=TaskType.REGRESSION, input_features=4, output_dim=1,
            hidden_layer_sizes=[16],
        )
        model = build_mlp(arch)
        config = DLTrainingConfig(architecture_name="mlp", task_type=TaskType.REGRESSION)
        train_batch = to_tensors(X_train, y_train, TaskType.REGRESSION)
        training_result = train_model(model, train_batch, config)

        eval_batch = to_tensors(X_eval, y_eval, TaskType.REGRESSION)
        evaluation_result = evaluate_model(model, eval_batch, architecture_name="mlp")

Out of scope for Phase 8.4 (and every later increment in this package
until explicitly implemented): integration into the full Phase-7
modeling pipeline, DL model selection, any architecture beyond the MLP
(CNN / LSTM / Transformer / attention / sequence models), experiment
tracking / ``ExperimentRecord`` (Phase 9), hyperparameter optimization,
SHAP, deployment, and general feature-engineering execution.
"""

from __future__ import annotations

from .architectures import MLPActivation, MLPArchitectureConfig
from .availability import TorchAvailability, is_torch_available, torch_availability
from .contracts import (
    DL_ENGINE_VERSION,
    DLDevice,
    DLEvaluationResult,
    DLLoss,
    DLOptimizer,
    DLTrainingConfig,
    DLTrainingResult,
    DLTrainingStatus,
)
from .evaluation import evaluate_model
from .mlp import build_mlp
from .runtime import DeviceResolution, resolve_device, seed_everything
from .tensors import TensorBatch, to_tensors
from .training_loop import train_model

__all__ = [
    "DL_ENGINE_VERSION",
    "DLDevice",
    "DLEvaluationResult",
    "DLLoss",
    "DLOptimizer",
    "DLTrainingConfig",
    "DLTrainingResult",
    "DLTrainingStatus",
    "DeviceResolution",
    "MLPActivation",
    "MLPArchitectureConfig",
    "TensorBatch",
    "TorchAvailability",
    "build_mlp",
    "evaluate_model",
    "is_torch_available",
    "resolve_device",
    "seed_everything",
    "to_tensors",
    "torch_availability",
    "train_model",
]
