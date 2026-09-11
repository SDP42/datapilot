"""Phase 8.2 — the deterministic PyTorch runtime (`dl_engine.runtime`).

CUDA / MPS presence is simulated via the injectable `_import` seam so
these tests never require real accelerator hardware.
"""

from __future__ import annotations

import pytest

from dl_engine.contracts import DLDevice
from dl_engine.runtime import resolve_device, seed_everything


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available
        self.manual_seed_all_calls: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)


class _FakeMPSBackend:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeBackends:
    def __init__(self, mps_available: bool):
        self.mps = _FakeMPSBackend(mps_available)


class _FakeTorch:
    __version__ = "0.0.0-fake"

    def __init__(self, cuda_available: bool = False, mps_available: bool = False):
        self.cuda = _FakeCuda(cuda_available)
        self.backends = _FakeBackends(mps_available)
        self.manual_seed_calls: list[int] = []
        self.deterministic_calls: list[tuple] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def use_deterministic_algorithms(self, *args, **kwargs) -> None:
        self.deterministic_calls.append((args, kwargs))


def _fake_import(fake_torch: _FakeTorch):
    def _import(name: str):
        assert name == "torch"
        return fake_torch

    return _import


# --- resolve_device: CPU is always available, no torch needed ----------


def test_cpu_is_always_available_without_torch():
    result = resolve_device(DLDevice.CPU, _import=_raise_import_error)
    assert result.available is True
    assert result.resolved_device is DLDevice.CPU
    assert result.device_str == "cpu"
    assert result.reason is None


# --- resolve_device: PyTorch missing entirely ---------------------------


def test_cuda_unavailable_when_torch_not_installed():
    result = resolve_device(DLDevice.CUDA, _import=_raise_import_error)
    assert result.available is False
    assert result.resolved_device is None
    assert result.device_str is None
    assert result.reason is not None


def test_mps_unavailable_when_torch_not_installed():
    result = resolve_device(DLDevice.MPS, _import=_raise_import_error)
    assert result.available is False
    assert result.resolved_device is None


# --- resolve_device: torch present, accelerator absent (no hardware needed) --


def test_cuda_unavailable_when_torch_reports_no_cuda():
    fake = _FakeTorch(cuda_available=False)
    result = resolve_device(DLDevice.CUDA, _import=_fake_import(fake))
    assert result.available is False
    assert "CUDA" in (result.reason or "")


def test_mps_unavailable_when_torch_reports_no_mps():
    fake = _FakeTorch(mps_available=False)
    result = resolve_device(DLDevice.MPS, _import=_fake_import(fake))
    assert result.available is False
    assert "MPS" in (result.reason or "")


# --- resolve_device: torch present, accelerator reported available -----


def test_cuda_available_when_torch_reports_cuda():
    fake = _FakeTorch(cuda_available=True)
    result = resolve_device(DLDevice.CUDA, _import=_fake_import(fake))
    assert result.available is True
    assert result.resolved_device is DLDevice.CUDA
    assert result.device_str == "cuda"
    assert result.reason is None


def test_mps_available_when_torch_reports_mps():
    fake = _FakeTorch(mps_available=True)
    result = resolve_device(DLDevice.MPS, _import=_fake_import(fake))
    assert result.available is True
    assert result.resolved_device is DLDevice.MPS
    assert result.device_str == "mps"


def test_no_silent_fallback_to_cpu_when_accelerator_unavailable():
    fake = _FakeTorch(cuda_available=False)
    result = resolve_device(DLDevice.CUDA, _import=_fake_import(fake))
    # never silently resolves to CPU instead of the requested device
    assert result.resolved_device is None
    assert result.resolved_device is not DLDevice.CPU


# --- resolve_device: JSON serialisable + deterministic ------------------


def test_device_resolution_is_json_serialisable_and_deterministic():
    a = resolve_device(DLDevice.CPU)
    b = resolve_device(DLDevice.CPU)
    assert a.model_dump_json() == b.model_dump_json()


# --- seed_everything: PyTorch missing -----------------------------------


def test_seed_everything_returns_false_when_torch_missing():
    assert seed_everything(42, _import=_raise_import_error) is False


# --- seed_everything: PyTorch present (simulated) -----------------------


def test_seed_everything_seeds_torch_when_available():
    fake = _FakeTorch()
    assert seed_everything(123, _import=_fake_import(fake)) is True
    assert fake.manual_seed_calls == [123]


def test_seed_everything_seeds_cuda_only_when_cuda_available():
    fake_no_cuda = _FakeTorch(cuda_available=False)
    seed_everything(7, _import=_fake_import(fake_no_cuda))
    assert fake_no_cuda.cuda.manual_seed_all_calls == []

    fake_cuda = _FakeTorch(cuda_available=True)
    seed_everything(7, _import=_fake_import(fake_cuda))
    assert fake_cuda.cuda.manual_seed_all_calls == [7]


def test_seed_everything_deterministic_flag_controls_use_deterministic_algorithms():
    fake = _FakeTorch()
    seed_everything(1, deterministic=True, _import=_fake_import(fake))
    assert len(fake.deterministic_calls) == 1

    fake2 = _FakeTorch()
    seed_everything(1, deterministic=False, _import=_fake_import(fake2))
    assert len(fake2.deterministic_calls) == 0


def test_seed_everything_also_seeds_python_and_numpy_globals():
    import random

    import numpy as np

    fake = _FakeTorch()
    seed_everything(99, _import=_fake_import(fake))
    a = random.random()
    b = np.random.rand()

    seed_everything(99, _import=_fake_import(_FakeTorch()))
    c = random.random()
    d = np.random.rand()

    assert a == c
    assert b == d


# --- real-torch path (skipped when unavailable) -------------------------


def test_real_torch_cpu_resolution_when_installed():
    pytest.importorskip("torch")
    result = resolve_device(DLDevice.CPU)
    assert result.available is True
    assert result.device_str == "cpu"


def test_real_torch_seed_everything_when_installed():
    pytest.importorskip("torch")
    assert seed_everything(42) is True
