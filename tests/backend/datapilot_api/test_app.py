"""Phase 13.1 — the FastAPI app's stateless endpoints
(`backend.datapilot_api.routes.health` / `datasets` / `modeling`).
"""

from __future__ import annotations

import io


def test_health(client, auth_headers):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_route_without_token_returns_401(client, auth_headers):
    response = client.get("/api/v1/jobs/does-not-exist")
    assert response.status_code == 401


def test_ingest_returns_reference_and_profile(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/datasets/ingest", files=files, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["reference"]["dataset_id"]
    assert body["profile"]["n_rows"] == 80


def test_ingest_empty_file_returns_400(client, auth_headers):
    files = {"file": ("data.csv", io.BytesIO(b""), "text/csv")}
    response = client.post("/api/v1/datasets/ingest", files=files, headers=auth_headers)
    assert response.status_code == 400


def test_quality_endpoint_returns_real_report(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/datasets/quality", files=files, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert "findings" in body
    assert "dataset_id" in body


def test_quality_endpoint_accepts_target_column(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/datasets/quality",
        files=files,
        data={"target_column": "y"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["target_column"] == "y"


def test_eda_endpoint_returns_real_report(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/datasets/eda", files=files, headers=auth_headers)
    assert response.status_code == 200
    assert "dataset_id" in response.json()


def test_modeling_run_requires_objective(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/modeling/run", files=files, headers=auth_headers)
    assert response.status_code == 422  # objective is a required form field


def test_modeling_run_completes_on_real_data(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/run",
        files=files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["selection"]["selected_family"] is not None


def test_modeling_run_rejects_invalid_forecast_horizon(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/run",
        files=files,
        data={"objective": "predict y", "forecast_horizon": 0},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_modeling_search_returns_more_than_20_ranked_candidates(
    client, auth_headers, sample_csv_bytes
):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/search",
        files=files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["candidate_count"] > 20
    assert len(body["candidates"]) == body["candidate_count"]
    assert body["candidates"][0]["rank"] == 1


def test_modeling_search_requires_auth(client, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/modeling/search", files=files, data={"objective": "predict y"})
    assert response.status_code == 401
