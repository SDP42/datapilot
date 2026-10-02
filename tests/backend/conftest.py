"""Shared fixtures for the Phase-13 backend test suite.

Every test gets its own isolated SQLite file (never the shared
`get_settings()` / `get_engine()` `lru_cache`d singletons leaking
between tests) by clearing those caches and pointing
`DATAPILOT_DATABASE_URL` at a fresh `tmp_path` file before building the
app.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DATAPILOT_DATABASE_URL", f"sqlite:///{db_path}")

    from backend.settings import get_settings
    from backend.datapilot_api.db import get_engine

    get_settings.cache_clear()
    get_engine.cache_clear()

    from backend.datapilot_api import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()
    get_engine.cache_clear()


@pytest.fixture
def sample_csv_bytes():
    import numpy as np

    rng = np.random.default_rng(0)
    lines = ["x1,x2,y"]
    for _ in range(80):
        x1, x2 = rng.normal(), rng.normal()
        y = x1 * 2 + x2 + rng.normal() * 0.1
        lines.append(f"{x1},{x2},{y}")
    return ("\n".join(lines)).encode()
