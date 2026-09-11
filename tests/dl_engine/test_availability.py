"""Phase 8.1 — the PyTorch availability probe (`dl_engine.availability`)."""

from __future__ import annotations

import json

import pytest

from dl_engine.availability import TorchAvailability, is_torch_available, torch_availability


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_unavailable_path_is_deterministic_and_explicit():
    result = torch_availability(_import=_raise_import_error)
    assert result.available is False
    assert result.version is None
    assert result.reason is not None
    assert "torch" in result.reason.lower() or "pytorch" in result.reason.lower()


def test_unavailable_path_repeated_calls_are_identical():
    a = torch_availability(_import=_raise_import_error)
    b = torch_availability(_import=_raise_import_error)
    assert a == b
    assert a.model_dump_json() == b.model_dump_json()


def test_is_torch_available_matches_torch_availability():
    assert is_torch_available() == torch_availability().available


def test_current_environment_reflects_real_torch_state():
    try:
        import torch  # noqa: F401
    except ImportError:
        assert is_torch_available() is False
        result = torch_availability()
        assert result.reason is not None
        assert result.version is None
    else:
        assert is_torch_available() is True
        result = torch_availability()
        assert result.version is not None
        assert result.reason is None


def test_available_path_only_when_torch_is_installed():
    pytest.importorskip("torch")
    result = torch_availability()
    assert result.available is True
    assert result.version is not None
    assert result.reason is None


def test_torch_availability_is_json_serialisable_and_round_trips():
    result = torch_availability(_import=_raise_import_error)
    payload = result.model_dump_json()
    json.loads(payload)  # valid JSON
    restored = TorchAvailability.model_validate_json(payload)
    assert restored == result
