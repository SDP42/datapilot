"""Phase 13.2 — the `JobStore`: SQLAlchemy-backed job-record persistence.

Mirrors the filesystem stores' own read/write boundary (`register` /
`get` / `list`-shaped methods, never a raw ORM session leaking out) —
just backed by a database table instead of JSON files, since job
records need concurrent-safe, queryable-by-status access (see
`backend.datapilot_api.db`'s own docstring for why this phase is where
that distinction first matters).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from .job_models import JobRow


class JobStatus(str, Enum):
    """Lifecycle of one asynchronous job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(BaseModel):
    """The JSON-serialisable view of one `JobRow` — never the ORM row itself."""

    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    kind: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: dict | None = Field(default=None, description="Populated iff status is completed.")
    error: str | None = Field(default=None, description="Populated iff status is failed.")


def _to_record(row: JobRow) -> JobRecord:
    return JobRecord(
        job_id=row.job_id,
        kind=row.kind,
        status=JobStatus(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        result=json.loads(row.result_json) if row.result_json is not None else None,
        error=row.error,
    )


class JobNotFoundError(Exception):
    """No job with the requested id is registered."""


class JobStore:
    """A thin wrapper around one SQLAlchemy `Session` for job-record CRUD.

    Takes an already-constructed `Session` per call (the same per-request
    dependency-injection pattern `backend.datapilot_api.db.get_session`
    already establishes) — never opens or manages its own connection.
    """

    def create(self, session: Session, *, kind: str) -> JobRecord:
        """Register a new job with `status = PENDING`. Returns the created record."""
        now = datetime.now(timezone.utc)
        row = JobRow(
            job_id=str(uuid4()),
            kind=kind,
            status=JobStatus.PENDING.value,
            created_at=now,
            updated_at=now,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_record(row)

    def get(self, session: Session, job_id: str) -> JobRecord:
        row = session.get(JobRow, job_id)
        if row is None:
            raise JobNotFoundError(f"no registered job {job_id!r}")
        return _to_record(row)

    def mark_running(self, session: Session, job_id: str) -> JobRecord:
        return self._update(session, job_id, status=JobStatus.RUNNING)

    def mark_completed(self, session: Session, job_id: str, *, result: dict) -> JobRecord:
        return self._update(
            session, job_id, status=JobStatus.COMPLETED, result_json=json.dumps(result)
        )

    def mark_failed(self, session: Session, job_id: str, *, error: str) -> JobRecord:
        return self._update(session, job_id, status=JobStatus.FAILED, error=error)

    def _update(
        self,
        session: Session,
        job_id: str,
        *,
        status: JobStatus,
        result_json: str | None = None,
        error: str | None = None,
    ) -> JobRecord:
        row = session.get(JobRow, job_id)
        if row is None:
            raise JobNotFoundError(f"no registered job {job_id!r}")
        row.status = status.value
        row.updated_at = datetime.now(timezone.utc)
        if result_json is not None:
            row.result_json = result_json
        if error is not None:
            row.error = error
        session.commit()
        session.refresh(row)
        return _to_record(row)


__all__ = ["JobNotFoundError", "JobRecord", "JobStatus", "JobStore"]
