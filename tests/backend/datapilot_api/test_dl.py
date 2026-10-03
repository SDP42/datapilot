"""Phase 14.14 — the synchronous deep-learning training endpoint
(`backend.datapilot_api.routes.dl`).
"""

from __future__ import annotations

import io

import pytest

from dl_engine import is_torch_available

pytestmark = pytest.mark.skipif(
    not is_torch_available(), reason="PyTorch is not installed in this environment"
)


def test_dl_train_completes_on_real_data(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/dl/train",
        files=files,
        data={"objective": "predict y", "hidden_layer_sizes": "32,16", "epochs": "50"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["evaluation"]["metrics"]


def test_dl_train_requires_auth(client, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/dl/train", files=files, data={"objective": "predict y"})
    assert response.status_code == 401


def test_dl_train_defaults_hidden_layers_on_invalid_input(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post(
        "/api/v1/dl/train",
        files=files,
        data={"objective": "predict y", "hidden_layer_sizes": "not,numbers", "epochs": "20"},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
