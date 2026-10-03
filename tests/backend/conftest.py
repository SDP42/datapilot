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
    monkeypatch.setenv("DATAPILOT_TRAINED_MODEL_DIR", str(tmp_path / "models"))

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
def auth_headers(client):
    """A valid `Authorization: Bearer <token>` header for the default dev account."""
    from backend.settings import get_settings

    settings = get_settings()
    response = client.post(
        "/api/v1/auth/login",
        json={"username": settings.auth_username, "password": settings.auth_password},
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


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


@pytest.fixture
def sample_xlsx_bytes(tmp_path):
    """The same shape as `sample_csv_bytes`, serialized as a real .xlsx
    workbook — Phase 14.10's Excel ingestion path."""
    import io

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x1 = rng.normal(size=80)
    x2 = rng.normal(size=80)
    y = x1 * 2 + x2 + rng.normal(size=80) * 0.1
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


@pytest.fixture
def sample_blobs_csv_bytes():
    """Three well-separated 2D blobs — for Phase 14.13's clustering search,
    which needs real cluster structure, not a regression/classification
    target."""
    import numpy as np

    rng = np.random.default_rng(0)
    lines = ["x1,x2"]
    for cx, cy in ((0.0, 0.0), (10.0, 10.0), (0.0, 10.0)):
        for _ in range(40):
            x1 = rng.normal(cx, 1.0)
            x2 = rng.normal(cy, 1.0)
            lines.append(f"{x1},{x2}")
    return ("\n".join(lines)).encode()
