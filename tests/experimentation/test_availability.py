"""Phase 9.4 — the MLflow optional-dependency boundary
(`experimentation.availability`).
"""

from __future__ import annotations

from experimentation.availability import is_mlflow_available, mlflow_availability


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_unavailable_when_mlflow_missing():
    result = mlflow_availability(_import=_raise_import_error)
    assert result.available is False
    assert result.version is None
    assert "mlflow" in (result.reason or "").lower()


def test_is_mlflow_available_matches_probe():
    assert is_mlflow_available() == mlflow_availability().available


def test_availability_is_json_serialisable():
    result = mlflow_availability(_import=_raise_import_error)
    from experimentation.availability import MLflowAvailability

    restored = MLflowAvailability.model_validate_json(result.model_dump_json())
    assert restored == result
