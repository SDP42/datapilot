"""API routes, schemas, dependency wiring.

:func:`~backend.datapilot_api.app.create_app` builds the FastAPI
application. See :mod:`backend` for the package-level overview of every
Phase-13 increment.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
