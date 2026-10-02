"""Phase 13.4 — the DuckDB optional-dependency boundary (`backend.availability`)."""

from __future__ import annotations

from backend.availability import DuckDBAvailability, duckdb_availability, is_duckdb_available


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_unavailable_when_duckdb_missing():
    result = duckdb_availability(_import=_raise_import_error)
    assert result.available is False
    assert result.version is None
    assert "duckdb" in (result.reason or "").lower()


def test_is_duckdb_available_matches_probe():
    assert is_duckdb_available() == duckdb_availability().available


def test_availability_is_json_serialisable():
    result = duckdb_availability(_import=_raise_import_error)
    restored = DuckDBAvailability.model_validate_json(result.model_dump_json())
    assert restored == result
