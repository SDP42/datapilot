"""Phase 11.3 — the anthropic optional-dependency boundary
(`ai_engine.providers.availability`).
"""

from __future__ import annotations

from ai_engine.providers.availability import (
    AnthropicAvailability,
    anthropic_availability,
    is_anthropic_available,
)


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_unavailable_when_anthropic_missing():
    result = anthropic_availability(_import=_raise_import_error)
    assert result.available is False
    assert result.version is None
    assert "anthropic" in (result.reason or "").lower()


def test_is_anthropic_available_matches_probe():
    assert is_anthropic_available() == anthropic_availability().available


def test_availability_is_json_serialisable():
    result = anthropic_availability(_import=_raise_import_error)
    restored = AnthropicAvailability.model_validate_json(result.model_dump_json())
    assert restored == result
