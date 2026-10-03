"""Phase 15.1 — registration, login, and the current-user profile."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from ..auth import AuthenticatedUser, create_access_token, get_current_user
from ..db import get_session
from ..user_store import UserRecord, UsernameTakenError, UserStore

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_users = UserStore()

ExperienceLevel = Literal["beginner", "intermediate", "advanced"]
PrimaryGoal = Literal[
    "learn_data_science",
    "analyze_business_data",
    "build_ml_models",
    "research",
    "explore_the_platform",
    "other",
]


class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str
    experience_level: ExperienceLevel
    primary_goal: PrimaryGoal
    full_name: str | None = None
    role: str | None = None

    @field_validator("username")
    @classmethod
    def _username_shape(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("username must be at least 3 characters")
        return value

    @field_validator("password")
    @classmethod
    def _password_shape(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        return value


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


@router.post("/register", response_model=LoginResponse, status_code=201)
def register(request: RegisterRequest, session: Session = Depends(get_session)) -> LoginResponse:
    if request.password != request.confirm_password:
        raise HTTPException(status_code=422, detail="password and confirm_password do not match")
    try:
        record = _users.register(
            session,
            username=request.username,
            password=request.password,
            experience_level=request.experience_level,
            primary_goal=request.primary_goal,
            full_name=request.full_name,
            role=request.role,
        )
    except UsernameTakenError as exc:
        raise HTTPException(status_code=409, detail="username is already taken") from exc
    token = create_access_token(record.user_id, record.username)
    return LoginResponse(access_token=token, username=record.username)


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)) -> LoginResponse:
    row = _users.authenticate(session, request.username, request.password)
    if row is None:
        raise HTTPException(status_code=401, detail="invalid username or password")
    token = create_access_token(row.user_id, row.username)
    return LoginResponse(access_token=token, username=row.username)


@router.get("/me", response_model=UserRecord)
def me(
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> UserRecord:
    record = _users.get_record(session, current_user.user_id)
    if record is None:
        raise HTTPException(status_code=401, detail="user no longer exists")
    return record


__all__ = ["router"]
