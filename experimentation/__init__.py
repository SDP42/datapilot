"""Experiment Tracking (Phase 9) — definitions, execution, comparison,
history, recommendation interface.

**Phase 9.1** establishes the foundation: :class:`ExperimentRecord`, the
first contract in this codebase that deliberately carries a timestamp and
a random identifier (every prior phase's contract was byte-identical on
repeated calls — see :mod:`experimentation.contracts` for why Phase 9
changes that on purpose). It nests exactly one already-produced result —
a classical Phase-7 :class:`~data_engine.modeling.ModelingSpec`, a
Phase-8 :class:`~dl_engine.DLModelingResult`, or a Phase-8
:class:`~dl_engine.DLSelectionResult` — rather than duplicating any of
their fields. :func:`capture_environment` reads the installed-package
environment (via ``importlib.metadata`` only — never imports a package,
so calling it never imports ``torch``); :func:`record_experiment` is the
only way to produce a ``completed`` record, wrapping an already-produced
result without re-running, re-training, or re-evaluating anything.

**Phase 9.2** adds :class:`~experimentation.store.ExperimentStore` — a
deterministic, filesystem-based registry mirroring
:class:`~data_engine.validation.DatasetVersionStore` exactly: one
read-only JSON file per record under ``data/experiments/``, keyed by
``experiment_id``. ``register`` rejects a non-``completed`` record and a
duplicate id; ``get`` / ``exists`` / ``list_experiments`` read back.

**Phase 9.3** adds :func:`~experimentation.comparison.compare_experiments`
— a deterministic, selection-only comparison of multiple already-recorded
experiments, mirroring :func:`data_engine.modeling.select_model` /
:func:`dl_engine.select_dl_models`'s own boundary: it reads each record's
own already-established selection metric (never recomputes one) and
ranks the records that have one. A ``DL_MODELING`` record (a single run,
no selection) is always ineligible here, visible with an explicit reason.

**Phase 9.4** adds optional MLflow logging:
:func:`~experimentation.mlflow_integration.log_experiment_to_mlflow` logs
an already-recorded :class:`ExperimentRecord` (params + its established
metric) to one MLflow run. MLflow is an **optional** dependency — the
``mlflow`` extra (``pip install 'datapilot[mlflow]'``) — detected
deterministically by :mod:`experimentation.availability`, mirroring
:mod:`dl_engine.availability`'s own ``torch`` boundary exactly; every
other Phase 9 capability works without it installed.

**Not wired into anything automatically**: recording, storing, comparing,
and logging an experiment are all explicit, opt-in calls a caller makes
*after* it already has a result —
:func:`data_engine.modeling.run_modeling_pipeline`,
:func:`dl_engine.run_mlp_modeling` / ``run_cnn_modeling`` /
``run_lstm_modeling`` / ``run_transformer_modeling``, and
:func:`dl_engine.select_dl_models` are all untouched by this package.

Out of scope for Phase 9 (and every later increment in this package until
explicitly implemented): automatic recording from any existing Phase-7/8
entry point; a database-backed store (filesystem only); querying by
anything beyond ``source`` (a caller filters further in Python); an
MLflow *model registry* integration (only run-level param/metric
logging); experiment recommendation.
"""

from __future__ import annotations

from .availability import MLflowAvailability, is_mlflow_available, mlflow_availability
from .comparison import compare_experiments
from .contracts import (
    EXPERIMENTATION_ENGINE_VERSION,
    EnvironmentSnapshot,
    ExperimentComparisonEntry,
    ExperimentComparisonResult,
    ExperimentRecord,
    ExperimentSource,
    ExperimentStatus,
    MLflowLogResult,
    capture_environment,
    record_experiment,
)
from .mlflow_integration import log_experiment_to_mlflow
from .store import (
    DuplicateExperimentError,
    ExperimentNotFoundError,
    ExperimentStore,
    ExperimentStoreError,
)

__all__ = [
    "EXPERIMENTATION_ENGINE_VERSION",
    "DuplicateExperimentError",
    "EnvironmentSnapshot",
    "ExperimentComparisonEntry",
    "ExperimentComparisonResult",
    "ExperimentNotFoundError",
    "ExperimentRecord",
    "ExperimentSource",
    "ExperimentStatus",
    "ExperimentStore",
    "ExperimentStoreError",
    "MLflowAvailability",
    "MLflowLogResult",
    "capture_environment",
    "compare_experiments",
    "is_mlflow_available",
    "log_experiment_to_mlflow",
    "mlflow_availability",
    "record_experiment",
]
