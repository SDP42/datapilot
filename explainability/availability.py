"""SHAP availability detection — the Phase-10.3 optional-dependency boundary.

Mirrors :mod:`dl_engine.availability` / :mod:`experimentation.availability`
exactly: every other ``explainability`` capability (the Phase-10.1
contracts, Phase-10.2 permutation importance, Phase-10.4 partial
dependence — both built on scikit-learn, already a Phase-7.4 dependency)
never requires SHAP. This module is the **only** place in
``explainability`` that imports ``shap``, and it does so **lazily** —
only when :func:`shap_availability` / :func:`is_shap_available` is
actually called, never at package import time.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel, Field

_UNAVAILABLE_REASON = (
    "SHAP is not installed in this environment. Install the optional 'explain' extra "
    "(pip install 'datapilot[explain]') to enable Phase 10.3 SHAP explanations; every other "
    "Phase 10 capability (permutation importance, partial dependence) works without it."
)


class SHAPAvailability(BaseModel):
    """The result of probing whether SHAP is importable right now.

    JSON-primitive only — no module object, no explainer, no SHAP values array.
    """

    available: bool = Field(description="True iff `import shap` succeeded in this environment.")
    version: str | None = Field(
        default=None, description="`shap.__version__`, when available; else None."
    )
    reason: str | None = Field(
        default=None, description="Why SHAP is unavailable; None when available."
    )


def shap_availability(*, _import: Callable[[str], ModuleType] = import_module) -> SHAPAvailability:
    """Probe whether SHAP is importable in this environment (lazy, uncached).

    ``_import`` is an injectable import hook for tests only — the public
    call is always ``shap_availability()``.
    """
    try:
        shap = _import("shap")
    except ImportError:
        return SHAPAvailability(available=False, reason=_UNAVAILABLE_REASON)
    return SHAPAvailability(available=True, version=str(shap.__version__))


def is_shap_available() -> bool:
    """True iff SHAP is importable in this environment right now."""
    return shap_availability().available


__all__ = ["SHAPAvailability", "is_shap_available", "shap_availability"]
