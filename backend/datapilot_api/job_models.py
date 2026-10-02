"""Phase 13.2 — the `jobs` table ORM model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class JobRow(Base):
    """One row per submitted asynchronous job (Phase 13.3).

    `result_json` / `error` are plain `Text` columns holding an already-
    serialised JSON string (the same `model_dump_json()` every other
    Phase result contract in this codebase already produces) — never a
    database-specific JSON column type, so the same schema works
    identically on SQLite (dev/test) and PostgreSQL (production).
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


__all__ = ["JobRow"]
