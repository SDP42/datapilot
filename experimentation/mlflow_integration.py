"""Phase 9.4 — optional MLflow logging for a recorded experiment.

:func:`log_experiment_to_mlflow` is the **only** function in this module
that imports MLflow, and it does so lazily (via :mod:`experimentation.availability`),
exactly like every ``dl_engine`` module's own ``torch`` boundary. It logs
an **already-produced** :class:`~experimentation.contracts.ExperimentRecord`
— params (identity, seed, environment) and, when the record's own source
has one, its established selection metric — to an MLflow run. It never
re-runs, re-trains, or re-evaluates anything, and it is **not** wired into
:func:`experimentation.contracts.record_experiment` or
:class:`experimentation.store.ExperimentStore` — logging to MLflow is a
separate, explicit, opt-in call a caller makes with an already-recorded
``ExperimentRecord``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from data_engine.modeling import ModelSelection, TrainingRunStatus
from dl_engine import DLSelectionResult

from .availability import mlflow_availability
from .contracts import ExperimentRecord, ExperimentSource, MLflowLogResult

_DEFAULT_EXPERIMENT_NAME = "datapilot"


def _metrics_for(record: ExperimentRecord) -> dict[str, float]:
    """The metric(s) worth logging for `record`, read from its own already-produced result.

    Never recomputes a metric. ``DL_MODELING`` logs every metric its own
    evaluation produced (there is no "selection", just one run);
    ``CLASSICAL_MODELING`` / ``DL_SELECTION`` log the single established
    selection metric (``selected_score`` keyed by ``selection_metric``) —
    the same values :func:`experimentation.comparison.compare_experiments`
    itself reads, never a second extraction path.
    """
    if record.source is ExperimentSource.DL_MODELING:
        if record.dl_modeling_result is not None and record.dl_modeling_result.evaluation:
            return dict(record.dl_modeling_result.evaluation.metrics)
        return {}

    selection: ModelSelection | DLSelectionResult | None = None
    if record.source is ExperimentSource.CLASSICAL_MODELING and record.classical_result:
        selection = record.classical_result.selection
    elif record.source is ExperimentSource.DL_SELECTION:
        selection = record.dl_selection_result

    if selection is None or selection.selection_metric is None or selection.selected_score is None:
        return {}
    return {selection.selection_metric: selection.selected_score}


def log_experiment_to_mlflow(
    record: ExperimentRecord,
    *,
    experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
    _import: Callable[[str], ModuleType] = import_module,
) -> MLflowLogResult:
    """Log `record` to MLflow as one run under `experiment_name`.

    Logged **params**: ``experiment_id``, ``source``, ``seed`` (when not
    ``None``), ``python_version``, ``platform``, and each tracked
    package's installed version (prefixed ``pkg_``, e.g. ``pkg_torch``;
    an uninstalled package is logged as the literal string ``"not
    installed"`` — MLflow params are string-only, so ``None`` cannot be
    logged directly). Logged **metrics**: see :func:`_metrics_for`. The
    MLflow run name is ``record.experiment_id``.

    Never raises for an environment-level condition — MLflow not
    installed, or MLflow itself raising while logging — both are reported
    as a structured ``MLflowLogResult`` naming the stage that stopped the
    attempt, matching every other Phase-8/9 entry point's convention.
    """
    availability = mlflow_availability(_import=_import)
    if not availability.available:
        return MLflowLogResult(
            status=TrainingRunStatus.UNAVAILABLE,
            experiment_id=record.experiment_id,
            reason=availability.reason or "",
        )

    mlflow = _import("mlflow")
    metrics = _metrics_for(record)

    params: dict[str, str] = {
        "experiment_id": record.experiment_id,
        "source": record.source.value if record.source else "none",
    }
    if record.seed is not None:
        params["seed"] = str(record.seed)
    if record.environment is not None:
        params["python_version"] = record.environment.python_version
        params["platform"] = record.environment.platform
        for name, version in record.environment.packages.items():
            params[f"pkg_{name}"] = version if version is not None else "not installed"

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=record.experiment_id) as run:
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            run_id = run.info.run_id
    except Exception as exc:  # noqa: BLE001 - MLflow's own exception hierarchy is not public API
        return MLflowLogResult(
            status=TrainingRunStatus.FAILED,
            experiment_id=record.experiment_id,
            reason=f"MLflow logging failed: {exc}",
        )

    return MLflowLogResult(
        status=TrainingRunStatus.COMPLETED,
        experiment_id=record.experiment_id,
        mlflow_run_id=run_id,
        mlflow_experiment_name=experiment_name,
        logged_metrics=metrics,
    )


__all__ = ["log_experiment_to_mlflow"]
