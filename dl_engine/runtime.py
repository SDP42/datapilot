"""Phase 8.2 — deterministic PyTorch runtime: seeding + device resolution.

Mirrors :mod:`dl_engine.availability`'s lazy-import discipline: every
function here imports ``torch`` only when called (via the same
injectable ``_import`` seam used throughout ``dl_engine``, so tests can
simulate CUDA / MPS presence without real hardware), never at module
import time. CPU-only environments are the default, fully-supported
case — this module never requires CUDA or MPS to work correctly.

No distributed training, no multiprocessing, no GPU orchestration, no
mixed precision, and no silent device fallback live here: a requested
accelerator that is not available is reported as a structured
``DeviceResolution(available=False, ...)``, never silently swapped for
CPU.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from importlib import import_module
from types import ModuleType

import numpy as np
from pydantic import BaseModel, Field

from .availability import torch_availability
from .contracts import DLDevice


class DeviceResolution(BaseModel):
    """The deterministic outcome of resolving a requested :class:`DLDevice`.

    JSON-primitive only. ``resolved_device`` / ``device_str`` are ``None``
    whenever ``available`` is ``False`` — a requested-but-missing
    accelerator never silently falls back to another device.
    """

    requested_device: DLDevice
    available: bool
    resolved_device: DLDevice | None = Field(
        default=None, description="The device training will actually run on; None if unavailable."
    )
    device_str: str | None = Field(
        default=None,
        description="The torch device string (e.g. 'cpu', 'cuda', 'mps'); None if unavailable.",
    )
    reason: str | None = Field(
        default=None, description="Why the requested device is unavailable; None when available."
    )


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    _import: Callable[[str], ModuleType] = import_module,
) -> bool:
    """Seed Python's ``random``, NumPy's global RNG, and PyTorch (CPU + CUDA).

    PyTorch initializes an arbitrary caller-supplied ``nn.Module`` from its
    own global RNG, so — unlike the rest of DataPilot's modeling code,
    which threads a local ``np.random.default_rng`` instance instead of
    touching global state — a process-wide seed is the only way to make
    that initialization reproducible. This function mutates global RNG
    state **only when called**, never at import time, and the call site
    (:func:`dl_engine.training_loop.train_model`) is explicit about it.

    ``deterministic=True`` (the default, matching ``DLTrainingConfig``'s
    own default) additionally enables ``torch.use_deterministic_algorithms``
    (``warn_only=True`` — a kernel with no deterministic implementation
    warns rather than raising, since Phase 8.2 defines no architecture and
    cannot know in advance what operations a caller-supplied model uses).

    Returns ``True`` once seeding completed; ``False`` when PyTorch is not
    installed (nothing is partially seeded in that case — Python's
    ``random`` and NumPy's global RNG are left untouched too, so a caller
    can tell "seeding happened" from "seeding could not happen" without
    inspecting global state itself).
    """
    availability = torch_availability(_import=_import)
    if not availability.available:
        return False

    torch = _import("torch")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return True


def resolve_device(
    requested: DLDevice, *, _import: Callable[[str], ModuleType] = import_module
) -> DeviceResolution:
    """Deterministically resolve a requested :class:`DLDevice`.

    CPU is always available and never requires importing ``torch``. CUDA /
    MPS are resolved only when PyTorch itself reports them present in this
    process — an unavailable accelerator returns a structured
    ``available=False`` result with a reason; it never silently resolves
    to CPU instead (that would silently change reproducibility / the
    device the caller thinks they asked for).
    """
    if requested is DLDevice.CPU:
        return DeviceResolution(
            requested_device=requested,
            available=True,
            resolved_device=DLDevice.CPU,
            device_str="cpu",
        )

    availability = torch_availability(_import=_import)
    if not availability.available:
        return DeviceResolution(
            requested_device=requested,
            available=False,
            reason=f"PyTorch is not installed; cannot resolve device '{requested.value}'. "
            f"{availability.reason}",
        )

    torch = _import("torch")

    if requested is DLDevice.CUDA:
        if bool(torch.cuda.is_available()):
            return DeviceResolution(
                requested_device=requested,
                available=True,
                resolved_device=DLDevice.CUDA,
                device_str="cuda",
            )
        return DeviceResolution(
            requested_device=requested,
            available=False,
            reason="CUDA was requested but is not available in this environment.",
        )

    if requested is DLDevice.MPS:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            return DeviceResolution(
                requested_device=requested,
                available=True,
                resolved_device=DLDevice.MPS,
                device_str="mps",
            )
        return DeviceResolution(
            requested_device=requested,
            available=False,
            reason="MPS was requested but is not available in this environment.",
        )

    return DeviceResolution(
        requested_device=requested,
        available=False,
        reason=f"unrecognised device: '{requested}'",
    )
