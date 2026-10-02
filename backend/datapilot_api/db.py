"""Phase 13.2 — SQLAlchemy engine / session setup.

The **first** database-backed store in this codebase — every Phase 1-12
store (`DatasetVersionStore`, `ExperimentStore`) is filesystem-based
(one JSON file per record). Job records need concurrent-safe,
queryable-by-status access across potentially many simultaneous API
requests, which a filesystem JSON-file store is not well suited for —
Phase 13's own roadmap scope is exactly where SQLAlchemy / PostgreSQL
enter this codebase for the first time (see `docs/decisions.md`).

SQLite by default (`backend.settings.Settings.database_url`'s own
default) needs no running database server — `pytest` never depends on
PostgreSQL actually being installed or reachable; PostgreSQL is
selected in production purely by setting `DATAPILOT_DATABASE_URL` to a
`postgresql://...` URL, with zero code change.
"""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """The declarative base every Phase-13 ORM model inherits from."""


def make_engine(database_url: str) -> Engine:
    """Build a SQLAlchemy engine for `database_url`.

    SQLite needs `check_same_thread=False` for FastAPI's threaded test
    client / worker model (a single SQLite file connection is otherwise
    bound to the thread that opened it); every other backend (notably
    PostgreSQL) takes no extra connect arguments.
    """
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@lru_cache
def get_engine() -> Engine:
    """The process-wide engine, built from `backend.settings.get_settings()`.

    `lru_cache` means this is built once per process — matching
    `get_settings`'s own caching convention, and avoiding a new
    connection pool per request.
    """
    from backend.settings import get_settings

    return make_engine(get_settings().database_url)


def create_all_tables(engine: Engine | None = None) -> None:
    """Create every Phase-13 ORM table that doesn't already exist.

    Idempotent — safe to call on every app startup. No migration tool
    (Alembic) is introduced in this increment; schema changes after this
    point are explicitly deferred (see `docs/decisions.md`).
    """
    from . import job_models  # noqa: F401 - imported for its side effect of registering the table

    Base.metadata.create_all(bind=engine or get_engine())


def get_session() -> Generator[Session, None, None]:
    """A FastAPI dependency yielding one request-scoped `Session`."""
    session_factory = make_session_factory(get_engine())
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "Base",
    "create_all_tables",
    "get_engine",
    "get_session",
    "make_engine",
    "make_session_factory",
]
