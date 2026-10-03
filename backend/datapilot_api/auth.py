"""Phase 15.1 — a real multi-user JWT auth boundary.

Replaces Phase 13.5's single configured operator account: credentials are
now verified against the `users` table (`..user_store.UserStore`), and
the signed JWT carries both the user's id and username so every
downstream dependency can scope data (history, gamification) to the
actual caller rather than a single shared account. A `DATAPILOT_AUTH_*`
dev account is still bootstrapped on startup (see `..bootstrap`) purely
for local-dev convenience — it is a real row in `users`, not a special
case here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.settings import get_settings

_security = HTTPBearer(auto_error=False)
_ALGORITHM = "HS256"


@dataclass(frozen=True)
class AuthenticatedUser:
    """The identity carried by a verified bearer token."""

    user_id: str
    username: str


def create_access_token(user_id: str, username: str) -> str:
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": username, "uid": user_id, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> AuthenticatedUser:
    """Return the identity encoded in `token`. Raises `HTTPException(401)` for anything invalid."""
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc
    username = payload.get("sub")
    user_id = payload.get("uid")
    if not username or not user_id:
        raise HTTPException(status_code=401, detail="invalid token")
    return AuthenticatedUser(user_id=user_id, username=username)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> AuthenticatedUser:
    """A FastAPI dependency: the authenticated identity, or a `401`."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    return decode_access_token(credentials.credentials)


__all__ = ["AuthenticatedUser", "create_access_token", "decode_access_token", "get_current_user"]
