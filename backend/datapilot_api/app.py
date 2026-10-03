"""Phase 13.1 — the FastAPI application factory.

:func:`create_app` wires every router together. Deliberately a
**factory function, not a module-level app instance** — tests construct
a fresh app (optionally with overridden settings / a test database) via
:func:`create_app`, never importing a shared mutable global.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .bootstrap import seed_dev_account
from .db import create_all_tables
from .routes import analytics, auth, datasets, dl, health, history, jobs, modeling, predictions


def create_app() -> FastAPI:
    """Build and return a new DataPilot API application instance.

    Creates every Phase-13.2 table that doesn't already exist
    (idempotent) before returning — so the first request against a
    fresh SQLite dev database, or a `TestClient` wrapping a freshly
    constructed app, never 500s on a missing `jobs` table. Also seeds
    the configured dev account (Phase 15.1) into `users` so the
    documented local-dev login keeps working on a fresh database.
    """
    from backend.settings import get_settings

    settings = get_settings()
    app = FastAPI(title=settings.api_title, version=settings.api_version)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    create_all_tables()
    seed_dev_account()

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(datasets.router)
    app.include_router(modeling.router)
    app.include_router(dl.router)
    app.include_router(predictions.router)
    app.include_router(jobs.router)
    app.include_router(analytics.router)
    app.include_router(history.router)

    return app


__all__ = ["create_app"]
