"""MLflow availability detection — the Phase-9.4 optional-dependency boundary.

Mirrors :mod:`dl_engine.availability` exactly, for the same reason: every
other ``experimentation`` capability (``ExperimentRecord``,
``capture_environment``, ``record_experiment``, ``ExperimentStore``,
``compare_experiments``) never requires MLflow. This module is the
**only** place in ``experimentation`` that imports ``mlflow``, and it does
so **lazily** — only when :func:`mlflow_availability` /
:func:`is_mlflow_available` is actually called, never at package import
time.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel, Field

_UNAVAILABLE_REASON = (
    "MLflow is not installed in this environment. Install the optional 'mlflow' extra "
    "(pip install 'datapilot[mlflow]') to enable Phase 9.4 experiment logging; every other "
    "Phase 9 capability (ExperimentRecord, capture_environment, record_experiment, "
    "ExperimentStore, compare_experiments) works without it."
)


class MLflowAvailability(BaseModel):
    """The result of probing whether MLflow is importable right now.

    JSON-primitive only — no module object, no MLflow client, no run handle.
    """

    available: bool = Field(description="True iff `import mlflow` succeeded in this environment.")
    version: str | None = Field(
        default=None, description="`mlflow.__version__`, when available; else None."
    )
    reason: str | None = Field(
        default=None, description="Why MLflow is unavailable; None when available."
    )


def mlflow_availability(
    *, _import: Callable[[str], ModuleType] = import_module
) -> MLflowAvailability:
    """Probe whether MLflow is importable in this environment (lazy, uncached).

    ``_import`` is an injectable import hook for tests only — the public
    call is always ``mlflow_availability()``.
    """
    try:
        mlflow = _import("mlflow")
    except ImportError:
        return MLflowAvailability(available=False, reason=_UNAVAILABLE_REASON)
    return MLflowAvailability(available=True, version=str(mlflow.__version__))


def is_mlflow_available() -> bool:
    """True iff MLflow is importable in this environment right now."""
    return mlflow_availability().available


__all__ = ["MLflowAvailability", "is_mlflow_available", "mlflow_availability"]
