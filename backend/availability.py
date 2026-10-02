"""DuckDB availability detection — the Phase-13.4 optional-dependency boundary.

Mirrors every other lazy-import boundary in this codebase
(`dl_engine.availability`, `experimentation.availability`,
`explainability.availability`, `ai_engine.providers.availability`)
exactly: the FastAPI app, SQLAlchemy job persistence, and job
orchestration never require `duckdb`. This module is the **only** place
in `backend` that imports it, and does so **lazily** — only when
:func:`duckdb_availability` / :func:`is_duckdb_available` is actually
called, never at package import time.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from pydantic import BaseModel, Field

_UNAVAILABLE_REASON = (
    "DuckDB is not installed in this environment. Install the optional 'analytics' extra "
    "(pip install 'datapilot[analytics]') to enable Phase 13.4 ad-hoc analytical queries; "
    "every other Phase 13 capability (the FastAPI app, job persistence, job orchestration) "
    "works without it."
)


class DuckDBAvailability(BaseModel):
    """The result of probing whether `duckdb` is importable right now."""

    available: bool = Field(description="True iff `import duckdb` succeeded in this environment.")
    version: str | None = Field(default=None, description="`duckdb.__version__`, when available.")
    reason: str | None = Field(default=None, description="Why unavailable; None when available.")


def duckdb_availability(
    *, _import: Callable[[str], ModuleType] = import_module
) -> DuckDBAvailability:
    """Probe whether `duckdb` is importable in this environment (lazy, uncached)."""
    try:
        duckdb = _import("duckdb")
    except ImportError:
        return DuckDBAvailability(available=False, reason=_UNAVAILABLE_REASON)
    return DuckDBAvailability(available=True, version=str(duckdb.__version__))


def is_duckdb_available() -> bool:
    """True iff `duckdb` is importable in this environment right now."""
    return duckdb_availability().available


__all__ = ["DuckDBAvailability", "duckdb_availability", "is_duckdb_available"]
