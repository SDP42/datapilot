"""Phase 13.6 — train-and-save / list / predict endpoint tests."""

from __future__ import annotations

import io


def _upload(client, auth_headers, csv_bytes, objective="predict y"):
    return client.post(
        "/api/v1/predict/train",
        headers=auth_headers,
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"objective": objective},
    )


def test_train_and_save_persists_a_model(client, auth_headers, sample_csv_bytes):
    response = _upload(client, auth_headers, sample_csv_bytes)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["model"] is not None
    assert body["model"]["model_id"].startswith("model-")
    assert body["model"]["target_column"] == "y"
    assert set(body["model"]["feature_cols"]) == {"x1", "x2"}


def test_models_list_reflects_trained_models(client, auth_headers, sample_csv_bytes):
    assert client.get("/api/v1/predict/models", headers=auth_headers).json() == []

    response = _upload(client, auth_headers, sample_csv_bytes)
    model_id = response.json()["model"]["model_id"]

    listed = client.get("/api/v1/predict/models", headers=auth_headers).json()
    assert len(listed) == 1
    assert listed[0]["model_id"] == model_id


def test_predict_returns_a_prediction_per_row(client, auth_headers, sample_csv_bytes):
    train_response = _upload(client, auth_headers, sample_csv_bytes)
    model_id = train_response.json()["model"]["model_id"]

    new_rows = b"x1,x2\n0.5,0.2\n-1.0,0.3\n"
    response = client.post(
        f"/api/v1/predict/models/{model_id}/predict",
        headers=auth_headers,
        files={"file": ("new.csv", io.BytesIO(new_rows), "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["row_count"] == 2
    assert len(body["predictions"]) == 2
    assert body["missing_columns"] == []


def test_predict_reports_missing_required_columns(client, auth_headers, sample_csv_bytes):
    train_response = _upload(client, auth_headers, sample_csv_bytes)
    model_id = train_response.json()["model"]["model_id"]

    new_rows = b"x1\n0.5\n"
    response = client.post(
        f"/api/v1/predict/models/{model_id}/predict",
        headers=auth_headers,
        files={"file": ("new.csv", io.BytesIO(new_rows), "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predictions"] == []
    assert body["missing_columns"] == ["x2"]


def test_predict_unknown_model_id_returns_404(client, auth_headers):
    response = client.post(
        "/api/v1/predict/models/model-does-not-exist/predict",
        headers=auth_headers,
        files={"file": ("new.csv", io.BytesIO(b"x1,x2\n1,2\n"), "text/csv")},
    )
    assert response.status_code == 404


def test_predict_routes_require_auth(client):
    assert client.get("/api/v1/predict/models").status_code == 401
    assert (
        client.post(
            "/api/v1/predict/train",
            files={"file": ("data.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
            data={"objective": "predict b"},
        ).status_code
        == 401
    )


def test_train_with_too_little_data_completes_with_no_model(client, auth_headers):
    tiny_csv = b"x1,x2,y\n1,2,3\n4,5,6\n"
    response = _upload(client, auth_headers, tiny_csv)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["model"] is None
