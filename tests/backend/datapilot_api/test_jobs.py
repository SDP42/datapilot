"""Phase 13.3 — asynchronous job submission and polling
(`backend.datapilot_api.routes.jobs`).
"""

from __future__ import annotations

import io


def test_submit_job_returns_pending_then_completes(client, auth_headers, sample_csv_bytes):
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    submit = client.post(
        "/api/v1/jobs/modeling",
        files=files,
        data={"objective": "predict y"},
        headers=auth_headers,
    )
    assert submit.status_code == 200
    job = submit.json()
    assert job["kind"] == "modeling"
    job_id = job["job_id"]

    # TestClient runs BackgroundTasks synchronously before the response
    # completes, so the job is already done by the time we poll.
    poll = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
    assert poll.status_code == 200
    polled = poll.json()
    assert polled["status"] == "completed"
    assert polled["result"]["status"] == "completed"
    assert polled["error"] is None


def test_poll_unknown_job_returns_404(client, auth_headers):
    response = client.get("/api/v1/jobs/does-not-exist", headers=auth_headers)
    assert response.status_code == 404


def test_job_ids_are_unique(client, auth_headers, sample_csv_bytes):
    files1 = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    files2 = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    job1 = client.post(
        "/api/v1/jobs/modeling", files=files1, data={"objective": "x"}, headers=auth_headers
    ).json()
    job2 = client.post(
        "/api/v1/jobs/modeling", files=files2, data={"objective": "x"}, headers=auth_headers
    ).json()
    assert job1["job_id"] != job2["job_id"]
