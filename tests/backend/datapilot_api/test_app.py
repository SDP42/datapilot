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


def test_ingest_accepts_xlsx(client, auth_headers, sample_xlsx_bytes):
    files = {
        "file": (
            "data.xlsx",
            io.BytesIO(sample_xlsx_bytes),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    }
    response = client.post("/api/v1/datasets/ingest", files=files, headers=auth_headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reference"]["source_format"] == "xlsx"
    assert body["profile"]["n_rows"] == 80
    assert body["profile"]["column_names"] == ["x1", "x2", "y"]


def test_modeling_run_completes_on_xlsx_upload(client, auth_headers, sample_xlsx_bytes):
    files = {"file": ("data.xlsx", io.BytesIO(sample_xlsx_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/run",
        files=files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["selection"]["selected_family"] is not None


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


def test_modeling_search_returns_more_than_100_ranked_candidates(
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
    assert body["candidate_count"] > 100
    assert len(body["candidates"]) == body["candidate_count"]
    assert body["candidates"][0]["rank"] == 1


def test_modeling_search_requires_auth(client, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/modeling/search", files=files, data={"objective": "predict y"})
    assert response.status_code == 401


def test_modeling_search_cross_validate_adds_cv_metrics(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/search",
        files=files,
        data={"objective": "predict y", "cross_validate": "true"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["cross_validation_enabled"] is True
    with_cv = [c for c in body["candidates"] if "cv_rmse_mean" in c["metrics"]]
    assert len(with_cv) > 0
    assert all("cv_rmse_std" in c["metrics"] for c in with_cv)


def test_modeling_tune_deep_tunes_a_named_candidate(client, auth_headers, sample_csv_bytes):
    search_files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    search_response = client.post(
        "/api/v1/modeling/search",
        files=search_files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert search_response.status_code == 200, search_response.text
    best = search_response.json()["candidates"][0]

    tune_files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/tune",
        files=tune_files,
        data={
            "objective": "predict y",
            "family": best["family"],
            "estimator_name": best["estimator_name"],
        },
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in ("completed", "unavailable")
    if body["status"] == "completed":
        assert body["estimator_name"] == best["estimator_name"]
        assert body["metrics"]


def test_modeling_tune_rejects_unknown_family(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/tune",
        files=files,
        data={"objective": "predict y", "family": "not_a_family", "estimator_name": "Ridge"},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_modeling_cluster_finds_real_cluster_structure(
    client, auth_headers, sample_blobs_csv_bytes
):
    files = {"file": ("blobs.csv", io.BytesIO(sample_blobs_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/cluster",
        files=files,
        data={"objective": "cluster customers into segments"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["task_type"] == "clustering"
    assert body["selection_metric"] == "silhouette_score"
    assert body["candidate_count"] > 50
    assert body["candidates"][0]["rank"] == 1
    assert body["candidates"][0]["hyperparameters"].get("n_clusters") == 3


def test_modeling_cluster_rejects_non_clustering_objective(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/cluster",
        files=files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "unavailable"


def test_modeling_cluster_requires_auth(client, sample_blobs_csv_bytes):
    files = {"file": ("blobs.csv", io.BytesIO(sample_blobs_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/modeling/cluster",
        files=files,
        data={"objective": "cluster customers into segments"},
    )
    assert response.status_code == 401
