"""Phase 13.4 — the optional DuckDB analytics endpoint
(`backend.datapilot_api.routes.analytics`).
"""

from __future__ import annotations


def test_query_returns_503_when_duckdb_not_installed(client, auth_headers):
    response = client.post(
        "/api/v1/analytics/experiments/query",
        json={"experiment_ids": [], "sql": "select 1"},
        headers=auth_headers,
    )
    # This environment does not have duckdb installed by default; the
    # route must report 503 with an explicit reason, never a 404 (the
    # route itself always exists) or an opaque 500.
    assert response.status_code == 503
    assert "duckdb" in response.json()["detail"].lower()
