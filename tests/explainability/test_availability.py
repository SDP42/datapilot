"""Phase 10.3 — the SHAP optional-dependency boundary
(`explainability.availability`).
"""

from __future__ import annotations

from explainability.availability import SHAPAvailability, is_shap_available, shap_availability


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_unavailable_when_shap_missing():
    result = shap_availability(_import=_raise_import_error)
    assert result.available is False
    assert result.version is None
    assert "shap" in (result.reason or "").lower()


def test_is_shap_available_matches_probe():
    assert is_shap_available() == shap_availability().available


def test_availability_is_json_serialisable():
    result = shap_availability(_import=_raise_import_error)
    restored = SHAPAvailability.model_validate_json(result.model_dump_json())
    assert restored == result
