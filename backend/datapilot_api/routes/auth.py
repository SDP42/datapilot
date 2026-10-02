"""Phase 13.5 — login and the current-user endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import create_access_token, get_current_user, verify_credentials

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class MeResponse(BaseModel):
    username: str


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    if not verify_credentials(request.username, request.password):
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = create_access_token(request.username)
    return LoginResponse(access_token=token, username=request.username)


@router.get("/me", response_model=MeResponse)
def me(username: str = Depends(get_current_user)) -> MeResponse:
    return MeResponse(username=username)


__all__ = ["router"]
