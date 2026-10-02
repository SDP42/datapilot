"""Phase 13.1 — typed backend settings.

Replaces the Phase-0 YAML-only loader (``datapilot.config.load_config``)
**for the backend specifically** — every non-backend engine still reads
``configs/default.yaml`` exactly as before; this module is additive, not
a replacement of that loader (``datapilot/config.py``'s own docstring
already named Phase 13 as where a typed settings model would arrive).

All values are environment-variable driven (``DATAPILOT_`` prefix, e.g.
``DATAPILOT_DATABASE_URL``) with safe, working defaults for local
development and tests — a SQLite file needs no running database server,
so ``pytest`` never depends on PostgreSQL actually being installed or
reachable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend configuration, overridable via `DATAPILOT_*` environment variables."""

    model_config = SettingsConfigDict(env_prefix="DATAPILOT_", extra="ignore")

    database_url: str = "sqlite:///./datapilot_backend.db"
    api_title: str = "DataPilot API"
    api_version: str = "0.1.0"
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB


@lru_cache
def get_settings() -> Settings:
    """A process-wide cached `Settings` instance.

    `lru_cache` means environment variables are read once, the first
    time this is called — matching the standard FastAPI-settings
    convention, and keeping every request handler's dependency
    injection (`Depends(get_settings)`) cheap.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
