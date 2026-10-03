"""Phase 13.5 — the single-operator JWT auth boundary
(`backend.datapilot_api.routes.auth`, `backend.datapilot_api.auth`).
"""

from __future__ import annotations


def test_login_with_correct_credentials_returns_token(client):
    from backend.settings import get_settings

    settings = get_settings()
    response = client.post(
        "/api/v1/auth/login",
        json={"username": settings.auth_username, "password": settings.auth_password},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["username"] == settings.auth_username
    assert body["access_token"]


def test_login_with_wrong_password_returns_401(client):
    from backend.settings import get_settings

    response = client.post(
        "/api/v1/auth/login",
        json={"username": get_settings().auth_username, "password": "wrong"},
    )
    assert response.status_code == 401


def test_login_with_unknown_username_returns_401(client):
    response = client.post(
        "/api/v1/auth/login", json={"username": "nobody", "password": "whatever"}
    )
    assert response.status_code == 401


def test_me_with_valid_token_returns_username(client, auth_headers):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["username"]


def test_me_without_token_returns_401(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_with_malformed_token_returns_401(client):
    response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_me_with_token_from_different_secret_returns_401(client, monkeypatch):
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone

    bad_token = pyjwt.encode(
        {"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "a-completely-different-secret",
        algorithm="HS256",
    )
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {bad_token}"})
    assert response.status_code == 401


def test_expired_token_returns_401(client):
    import jwt as pyjwt
    from datetime import datetime, timedelta, timezone
    from backend.settings import get_settings

    settings = get_settings()
    expired_token = pyjwt.encode(
        {"sub": "admin", "exp": datetime.now(timezone.utc) - timedelta(minutes=5)},
        settings.jwt_secret,
        algorithm="HS256",
    )
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401


def _register(client, username="newuser", password="correcthorsebattery", **overrides):
    payload = {
        "username": username,
        "password": password,
        "confirm_password": password,
        "experience_level": "beginner",
        "primary_goal": "learn_data_science",
    }
    payload.update(overrides)
    return client.post("/api/v1/auth/register", json=payload)


def test_register_creates_account_and_returns_token(client):
    response = _register(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["username"] == "newuser"
    assert body["access_token"]


def test_register_duplicate_username_returns_409(client):
    _register(client, username="dupe")
    response = _register(client, username="dupe")
    assert response.status_code == 409


def test_register_mismatched_passwords_returns_422(client):
    response = _register(client, password="aaaaaaaa", confirm_password="bbbbbbbb")
    assert response.status_code == 422


def test_register_short_password_returns_422(client):
    response = _register(client, password="short", confirm_password="short")
    assert response.status_code == 422


def test_register_then_login_with_new_account_works(client):
    _register(client, username="loginlater", password="correcthorsebattery")
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "loginlater", "password": "correcthorsebattery"},
    )
    assert response.status_code == 200


def test_me_includes_onboarding_profile_and_gamification_state(client):
    response = _register(
        client,
        username="profileuser",
        full_name="Ada Lovelace",
        experience_level="intermediate",
        primary_goal="build_ml_models",
    )
    token = response.json()["access_token"]
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["full_name"] == "Ada Lovelace"
    assert body["experience_level"] == "intermediate"
    assert body["primary_goal"] == "build_ml_models"
    assert body["xp"] == 0
    assert body["level"] == 1
    assert body["current_streak"] == 0
    assert body["badges"] == []
