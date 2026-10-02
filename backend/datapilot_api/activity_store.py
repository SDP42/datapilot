"""Phase 13.7 — the `ActivityStore`: SQLAlchemy-backed run-history persistence.

Mirrors `JobStore`'s own shape (a thin wrapper around one already-
constructed `Session`, a Pydantic record type, never a raw ORM row
leaking out) — `record()` is the one call every instrumented route makes
after it has a real result to report; `list_recent()` is read-only and
never raises for an empty table.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .activity_models import ActivityRow


class ActivityRecord(BaseModel):
    """The JSON-serialisable view of one `ActivityRow`."""

    activity_id: str
    kind: str
    dataset_id: str
    dataset_filename: str | None = None
    summary: str
    created_at: datetime


def _to_record(row: ActivityRow) -> ActivityRecord:
    return ActivityRecord(
        activity_id=row.activity_id,
        kind=row.kind,
        dataset_id=row.dataset_id,
        dataset_filename=row.dataset_filename,
        summary=row.summary,
        created_at=row.created_at,
    )


class ActivityStore:
    """A thin wrapper around one SQLAlchemy `Session` for activity-log CRUD."""

    def record(
        self,
        session: Session,
        *,
        kind: str,
        dataset_id: str,
        summary: str,
        dataset_filename: str | None = None,
    ) -> ActivityRecord:
        row = ActivityRow(
            activity_id=str(uuid4()),
            kind=kind,
            dataset_id=dataset_id,
            dataset_filename=dataset_filename,
            summary=summary,
            created_at=datetime.now(timezone.utc),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_record(row)

    def list_recent(self, session: Session, *, limit: int = 50) -> list[ActivityRecord]:
        stmt = select(ActivityRow).order_by(ActivityRow.created_at.desc()).limit(limit)
        rows = session.execute(stmt).scalars().all()
        return [_to_record(row) for row in rows]


__all__ = ["ActivityRecord", "ActivityStore"]
