"""PyTorch availability detection — the Phase-8 optional-dependency boundary.

DataPilot's classical stack (Phases 0-7, including the scikit-learn
baselines in :mod:`data_engine.modeling`) never requires PyTorch. This
module is the **only** place in ``dl_engine`` that imports ``torch``, and
it does so **lazily** — only when :func:`torch_availability` /
:func:`is_torch_available` is actually called, never at package import
time. Every other ``dl_engine`` module (and every Phase 0-7 module)
therefore stays importable in an environment with no PyTorch installed.

The probe is deterministic for a given environment: it reports exactly
whether ``import torch`` succeeds right now, never a cached guess from a
previous process or a filesystem check.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel, Field

_UNAVAILABLE_REASON = (
    "PyTorch is not installed in this environment. Install the optional 'dl' extra "
    "(pip install 'datapilot[dl]') to enable Phase 8 deep-learning training; every "
    "Phase 0-7 (classical) capability works without it."
)


class TorchAvailability(BaseModel):
    """The result of probing whether PyTorch is importable right now.

    JSON-primitive only — no module object, no fitted model, no tensor.
    """

    available: bool = Field(description="True iff `import torch` succeeded in this environment.")
    version: str | None = Field(
        default=None, description="`torch.__version__`, when available; else None."
    )
    reason: str | None = Field(
        default=None, description="Why PyTorch is unavailable; None when available."
    )


def torch_availability(
    *, _import: Callable[[str], ModuleType] = import_module
) -> TorchAvailability:
    """Probe whether PyTorch is importable in this environment (lazy, uncached).

    ``_import`` is an injectable import hook for tests only — the public
    call is always ``torch_availability()``.
    """
    try:
        torch = _import("torch")
    except ImportError:
        return TorchAvailability(available=False, reason=_UNAVAILABLE_REASON)
    return TorchAvailability(available=True, version=str(torch.__version__))


def is_torch_available() -> bool:
    """True iff PyTorch is importable in this environment right now."""
    return torch_availability().available
