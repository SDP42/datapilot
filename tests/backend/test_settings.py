"""Phase 13.1 — typed backend settings (`backend.settings`)."""

from __future__ import annotations

from backend.settings import Settings, get_settings


def test_defaults_need_no_environment():
    settings = Settings()
    assert settings.database_url == "sqlite:///./datapilot_backend.db"
    assert settings.api_title == "DataPilot API"
    assert settings.max_upload_bytes > 0


def test_environment_variable_override(monkeypatch):
    monkeypatch.setenv("DATAPILOT_DATABASE_URL", "postgresql://user:pass@host/db")
    settings = Settings()
    assert settings.database_url == "postgresql://user:pass@host/db"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
