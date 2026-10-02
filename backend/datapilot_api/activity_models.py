"""Phase 13.7 — the `activity_log` table ORM model.

Every prior Phase-13 synchronous endpoint (`/datasets/ingest`, `/quality`,
`/eda`, `/modeling/run`, `/predict/train`, `/predict/*/predict`) ran and
returned its result without leaving any record behind — only
`routes.jobs`' asynchronous path (Phase 13.3) was ever persisted. This
table is the first place a synchronous run's own history is recorded, so
"what did I run, and when" survives a page refresh or a new session.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class ActivityRow(Base):
    """One row per completed synchronous API call worth remembering.

    `summary` is a short, already-human-readable string (never a full
    result payload — those can be large, and the full `ModelingSpec` /
    `EDAReport` / etc. a run produced is not re-derivable from this row
    by design; this is a log, not a second copy of every result).
    """

    __tablename__ = "activity_log"

    activity_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    dataset_id: Mapped[str] = mapped_column(String(64))
    dataset_filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


__all__ = ["ActivityRow"]
