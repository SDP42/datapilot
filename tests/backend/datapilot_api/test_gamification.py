"""Phase 15.3/15.4 — per-user history scoping and the XP / streak / badge
state awarded for instrumented actions."""

from __future__ import annotations

import io


def _register_and_login(client, username):
    payload = {
        "username": username,
        "password": "correcthorsebattery",
        "confirm_password": "correcthorsebattery",
        "experience_level": "beginner",
        "primary_goal": "learn_data_science",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_ingest_awards_xp_and_first_steps_badge(client, sample_csv_bytes):
    headers = _register_and_login(client, "xpuser")
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/datasets/ingest", files=files, headers=headers)
    assert response.status_code == 200

    me = client.get("/api/v1/auth/me", headers=headers)
    body = me.json()
    assert body["xp"] == 5
    assert body["current_streak"] == 1
    assert "First Steps" in body["badges"]


def test_history_is_scoped_per_user(client, sample_csv_bytes):
    headers_a = _register_and_login(client, "userA")
    headers_b = _register_and_login(client, "userB")

    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    response = client.post("/api/v1/datasets/ingest", files=files, headers=headers_a)
    assert response.status_code == 200

    history_a = client.get("/api/v1/history", headers=headers_a)
    history_b = client.get("/api/v1/history", headers=headers_b)
    assert response.status_code == 200
    assert len(history_a.json()) == 1
    assert len(history_b.json()) == 0


def test_repeated_actions_same_day_count_streak_once(client, sample_csv_bytes):
    headers = _register_and_login(client, "samedayuser")
    files = {"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")}
    client.post("/api/v1/datasets/ingest", files=files, headers=headers)
    client.post(
        "/api/v1/datasets/ingest",
        files={"file": ("data.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
        headers=headers,
    )
    me = client.get("/api/v1/auth/me", headers=headers)
    body = me.json()
    assert body["xp"] == 10
    assert body["current_streak"] == 1
