"""Phase 15.1 — seeds the configured dev account into `users` on startup.

`backend.settings.Settings.auth_username` / `auth_password` used to *be*
the entire auth system (Phase 13.5); now that `users` is a real table,
that configured account is instead the one row this seeds on first
startup (idempotent — a no-op once it exists), purely so the documented
local-dev login (`admin` / `datapilot-dev-only`) keeps working without
every fresh clone needing to register first.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from .user_store import UserStore


def seed_dev_account(engine: Engine | None = None) -> None:
    from .db import get_engine, make_session_factory

    session_factory = make_session_factory(engine or get_engine())
    session = session_factory()
    try:
        from backend.settings import get_settings

        settings = get_settings()
        store = UserStore()
        if store.get_by_username(session, settings.auth_username) is not None:
            return
        store.register(
            session,
            username=settings.auth_username,
            password=settings.auth_password,
            experience_level="advanced",
            primary_goal="explore_the_platform",
            full_name="Dev Admin",
            role="administrator",
        )
    finally:
        session.close()


__all__ = ["seed_dev_account"]
