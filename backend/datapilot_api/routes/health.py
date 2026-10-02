"""Phase 13.1 — the health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = "ok"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


__all__ = ["router"]
