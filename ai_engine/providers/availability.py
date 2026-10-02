"""Anthropic SDK availability detection — the Phase-11.3 optional-dependency boundary.

Mirrors :mod:`dl_engine.availability` / :mod:`experimentation.availability`
/ :mod:`explainability.availability` exactly: the Phase-0
:class:`~ai_engine.providers.base.LLMProvider` interface and every other
``ai_engine`` capability (context building, tool schemas) never require
the ``anthropic`` package. This module is the **only** place in
``ai_engine`` that imports ``anthropic``, and it does so **lazily** —
only when :func:`anthropic_availability` / :func:`is_anthropic_available`
is actually called, never at package import time.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel, Field

_UNAVAILABLE_REASON = (
    "The 'anthropic' package is not installed in this environment. Install the optional 'ai' "
    "extra (pip install 'datapilot[ai]') to enable AnthropicProvider; every other ai_engine "
    "capability (LLMProvider interface, context building, tool schemas) works without it."
)


class AnthropicAvailability(BaseModel):
    """The result of probing whether the `anthropic` package is importable right now.

    JSON-primitive only — no module object, no client, no API key.
    """

    available: bool = Field(
        description="True iff `import anthropic` succeeded in this environment."
    )
    version: str | None = Field(
        default=None, description="`anthropic.__version__`, when available; else None."
    )
    reason: str | None = Field(
        default=None, description="Why the package is unavailable; None when available."
    )


def anthropic_availability(
    *, _import: Callable[[str], ModuleType] = import_module
) -> AnthropicAvailability:
    """Probe whether `anthropic` is importable in this environment (lazy, uncached).

    ``_import`` is an injectable import hook for tests only — the public
    call is always ``anthropic_availability()``. This probes only whether
    the SDK is installed, never whether an API key is valid or the
    network is reachable — those are a provider-construction-time or
    `complete()`-time concern, not an availability concern (the same
    distinction `torch_availability` / `mlflow_availability` /
    `shap_availability` already draw).
    """
    try:
        anthropic = _import("anthropic")
    except ImportError:
        return AnthropicAvailability(available=False, reason=_UNAVAILABLE_REASON)
    return AnthropicAvailability(available=True, version=str(anthropic.__version__))


def is_anthropic_available() -> bool:
    """True iff `anthropic` is importable in this environment right now."""
    return anthropic_availability().available


__all__ = ["AnthropicAvailability", "anthropic_availability", "is_anthropic_available"]
