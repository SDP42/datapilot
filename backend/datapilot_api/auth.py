"""Phase 13.5 — a single-operator JWT auth boundary.

Deliberately **not** a user-management system — one configured account
(`backend.settings.Settings.auth_username` / `auth_password`), issuing a
short-lived signed JWT on successful login. This exists so the frontend's
login page gates something real, not a cosmetic form; a multi-user /
role-based system is explicitly out of scope for this increment (see
`docs/decisions.md`).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.settings import get_settings

_security = HTTPBearer(auto_error=False)
_ALGORITHM = "HS256"


def verify_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    return username == settings.auth_username and password == settings.auth_password


def create_access_token(username: str) -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": username, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Return the username encoded in `token`. Raises `HTTPException(401)` for anything invalid."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="invalid token")
    return username


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> str:
    """A FastAPI dependency: the authenticated username, or a `401`."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return decode_access_token(credentials.credentials)


__all__ = ["create_access_token", "decode_access_token", "get_current_user", "verify_credentials"]
