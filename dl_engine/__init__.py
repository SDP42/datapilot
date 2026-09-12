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
  produces and the losses ``train_model`` already supports. A separate
  implementation from the Phase-7 scikit-learn MLP baseline
  (``data_engine.modeling``, ``ModelFamily.NEURAL``) — Phase 7 is
  untouched.

**Phase 8.4** added a deep learning evaluation *foundation*:

* :mod:`dl_engine.evaluation` — :func:`evaluate_model`, evaluating an
  **already-trained** model against explicitly supplied evaluation data
  (never a held-out partition it sources itself). ``model.eval()`` +
  ``torch.no_grad()``; the model's original training/eval mode is
  restored afterward; parameters are never written to. Reuses the exact
  Phase-7 metric vocabulary / semantics / rounding (``rmse`` / ``mae`` /
  ``r2`` for regression; ``accuracy`` / ``precision`` / ``recall`` /
  ``f1`` / ``roc_auc`` — macro-averaged, binary-only ``roc_auc`` gated on
  both classes present — for classification).
* :mod:`dl_engine.contracts` also gained :class:`DLEvaluationResult`.
  Strictly separate from training: this module never calls
  ``train_model`` and there is no ``fit_and_evaluate()`` convenience
  function.

**Phase 8.5** connected the existing components into a coherent
single-model workflow:

* :mod:`dl_engine.execution` — :func:`run_mlp_modeling`, the single
  modeling-facing entry point chaining ``build_mlp`` → ``to_tensors``
  (training data) → ``train_model`` → ``to_tensors`` (evaluation data,
  **always a separate array the caller supplies** — this function never
  splits data itself) → ``evaluate_model`` → one structured result.
  Deterministic (seeds before ``build_mlp``, exactly like
  ``train_model``'s own convention); evaluation is **never** attempted
  after a training stage that did not complete; every environment-level
  or data condition (PyTorch missing, device unavailable, invalid /
  incompatible data, a training or evaluation failure) is reported as a
  structured result naming the stage that stopped the run, never a raw
  exception.
* :mod:`dl_engine.contracts` also gained :class:`DLModelingResult` — the
  small aggregate result *nesting* ``DLTrainingResult`` /
  ``DLEvaluationResult`` (never duplicating their fields), the same
  "reference, don't flatten" pattern ``ModelingSpec`` already uses for
  its own ``training`` / ``evaluation`` sections.
* **Not a replacement for the Phase-7 modeling pipeline.**
  ``run_modeling_pipeline`` and the scikit-learn ``ModelFamily.NEURAL``
  baseline are untouched — ``data_engine.modeling`` has zero diffs from
  this increment. This is a deliberately separate, **opt-in** Phase-8
  entry point a caller reaches by explicitly importing ``dl_engine``,
  not something the existing Phase-7 API triggers automatically.

**Phase 8.6 (this increment)** adds a deterministic, DL-only selection
layer comparing multiple Phase-8 candidates — still no comparison against
classical models, no experiment tracking:

* :mod:`dl_engine.selection` — :func:`select_dl_models`, executing each
  supplied :class:`DLCandidate` exactly once via the existing
  ``run_mlp_modeling`` (no candidate is ever retrained) and ranking the
  results deterministically. The selection metric / direction reuse the
  **exact** per-task values :mod:`data_engine.modeling.selection`
  already established (``rmse`` / minimize for regression, ``f1`` /
  maximize for binary and multiclass classification) — never a
  DL-specific substitute. A candidate is eligible only when its run
  completed and its evaluation carries a finite selection-metric value;
  an ineligible candidate stays visible in the ranking with a reason, it
  is never dropped. The tie-break (metric, then architecture name, then
  the candidate's own serialised training configuration) is fully
  reproducible — never dictionary order, an object id, or a random
  number.
* :mod:`dl_engine.contracts` also gained :class:`DLCandidate` (an
  ``MLPArchitectureConfig`` + ``DLTrainingConfig`` pair, identified by a
  deterministic configuration digest — never a random UUID),
  :class:`DLCandidateRank`, and :class:`DLSelectionResult` (which
  **nests** ``DLModelingResult`` per candidate rather than duplicating
  its fields).
* **Compares Phase-8 DL candidates against each other only** — never
  against a Phase-7 classical candidate; that broader comparison is
  explicitly out of scope and would need its own separate design.

``dl_engine`` integrates with the existing Phase-7
:class:`~data_engine.modeling.ModelFamily` (``NEURAL``) and Phase-5
:class:`~data_engine.problem_understanding.TaskType` vocabularies rather
than inventing a parallel one; wiring a completed DL run into the
existing :class:`~data_engine.modeling.TrainingRun` /
:class:`~data_engine.modeling.TrainingOutcome` /
:class:`~data_engine.modeling.EvaluationResults` contracts,
classical-vs-DL comparison, and experiment tracking remain future work.

    from dl_engine import (
        is_torch_available, DLTrainingConfig, MLPArchitectureConfig,
        DLCandidate, select_dl_models,
    )

    if is_torch_available():
        candidates = [
            DLCandidate(
                architecture=MLPArchitectureConfig(
                    task_type=TaskType.REGRESSION, input_features=4, output_dim=1,
                    hidden_layer_sizes=[16],
                ),
                training_config=DLTrainingConfig(
                    architecture_name="mlp-16", task_type=TaskType.REGRESSION,
                ),
            ),
            DLCandidate(
                architecture=MLPArchitectureConfig(
                    task_type=TaskType.REGRESSION, input_features=4, output_dim=1,
                    hidden_layer_sizes=[32, 16],
                ),
                training_config=DLTrainingConfig(
                    architecture_name="mlp-32-16", task_type=TaskType.REGRESSION,
                ),
            ),
        ]
        result = select_dl_models(X_train, y_train, X_eval, y_eval, candidates)

**Phase 8.7 (this increment)** adds architecture **foundations only** for
three advanced architectures — contracts and lazy-PyTorch builders, with
**no training / evaluation / selection integration**:

* :mod:`dl_engine.architectures` gains :class:`CNNArchitectureConfig`,
  :class:`LSTMArchitectureConfig`, and
  :class:`TransformerArchitectureConfig` — sharing the same ``task_type``
  / ``output_dim`` convention as :class:`MLPArchitectureConfig`
  (unchanged, untouched by this increment).
* :mod:`dl_engine.cnn` — :func:`build_cnn` (a small 1D CNN; input shape
  ``(batch, input_channels, sequence_length)``), :mod:`dl_engine.lstm` —
  :func:`build_lstm` (a standard sequence LSTM; input shape ``(batch,
  seq_len, input_size)``), :mod:`dl_engine.transformer` — :func:`build_transformer`
  (a compact numeric-sequence Transformer encoder; input shape ``(batch,
  seq_len, input_size)``). Each returns raw logits (no softmax/sigmoid),
  rejects a mismatched input shape with ``ValueError`` rather than
  reshaping it, and is deterministic when
  :func:`dl_engine.runtime.seed_everything` seeds immediately beforehand
  — exactly like :func:`dl_engine.mlp.build_mlp`.
* **None of the three are wired into**
  :func:`dl_engine.training_loop.train_model`,
  :func:`dl_engine.execution.run_mlp_modeling`, or
  :func:`dl_engine.selection.select_dl_models` — that integration, and
  classical-vs-DL comparison, remain future work.

Out of scope for Phase 8.7 (and every later increment in this package
until explicitly implemented): CNN / LSTM / Transformer training loops,
modeling execution, evaluation integration, or candidate selection;
comparison between classical and DL models; automatic model selection
*within* the Phase-7 pipeline; attention-based architectures beyond the
compact Transformer foundation; experiment tracking /
``ExperimentRecord`` (Phase 9); hyperparameter optimization; SHAP;
deployment; and general feature-engineering execution.
"""

from __future__ import annotations

from .architectures import (
    CNNArchitectureConfig,
    LSTMArchitectureConfig,
    MLPActivation,
    MLPArchitectureConfig,
    TransformerArchitectureConfig,
)
from .availability import TorchAvailability, is_torch_available, torch_availability
from .cnn import build_cnn
from .contracts import (
    DL_ENGINE_VERSION,
    DLCandidate,
    DLCandidateRank,
    DLDevice,
    DLEvaluationResult,
    DLLoss,
    DLModelingResult,
    DLOptimizer,
    DLSelectionResult,
    DLTrainingConfig,
    DLTrainingResult,
    DLTrainingStatus,
)
from .evaluation import evaluate_model
from .execution import run_mlp_modeling
from .lstm import build_lstm
from .mlp import build_mlp
from .runtime import DeviceResolution, resolve_device, seed_everything
from .selection import select_dl_models
from .tensors import TensorBatch, to_tensors
from .training_loop import train_model
from .transformer import build_transformer

__all__ = [
    "DL_ENGINE_VERSION",
    "CNNArchitectureConfig",
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
    "DeviceResolution",
    "LSTMArchitectureConfig",
    "MLPActivation",
    "MLPArchitectureConfig",
    "TensorBatch",
    "TorchAvailability",
    "TransformerArchitectureConfig",
    "build_cnn",
    "build_lstm",
    "build_mlp",
    "build_transformer",
    "evaluate_model",
    "is_torch_available",
    "resolve_device",
    "run_mlp_modeling",
    "seed_everything",
    "select_dl_models",
    "to_tensors",
    "torch_availability",
    "train_model",
]
