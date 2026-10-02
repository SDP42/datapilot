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

**Not wired into anything automatically**: recording an experiment is an
explicit, opt-in call a caller makes *after* it already has a
``ModelingSpec`` / ``DLModelingResult`` / ``DLSelectionResult`` —
:func:`data_engine.modeling.run_modeling_pipeline`,
:func:`dl_engine.run_mlp_modeling`, and :func:`dl_engine.select_dl_models`
are all untouched by this increment.

Out of scope for Phase 9.1 (and every later increment in this package
until explicitly implemented): a persistent store for records (filesystem
or database), querying / listing / comparing recorded experiments,
automatic recording from any existing entry point, MLflow integration.
"""

from __future__ import annotations

from .contracts import (
    EXPERIMENTATION_ENGINE_VERSION,
    EnvironmentSnapshot,
    ExperimentRecord,
    ExperimentSource,
    ExperimentStatus,
    capture_environment,
    record_experiment,
)

__all__ = [
    "EXPERIMENTATION_ENGINE_VERSION",
    "EnvironmentSnapshot",
    "ExperimentRecord",
    "ExperimentSource",
    "ExperimentStatus",
    "capture_environment",
    "record_experiment",
]
