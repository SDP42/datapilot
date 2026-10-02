"""Phase 13.2 — the SQLAlchemy-backed job store
(`backend.datapilot_api.job_store.JobStore`).
"""

from __future__ import annotations

import pytest

from backend.datapilot_api.db import create_all_tables, make_engine, make_session_factory
from backend.datapilot_api.job_store import JobNotFoundError, JobStatus, JobStore


@pytest.fixture
def session_factory():
    engine = make_engine("sqlite:///:memory:")
    create_all_tables(engine)
    return make_session_factory(engine)


def test_create_returns_pending_job(session_factory):
    store = JobStore()
    with session_factory() as session:
        job = store.create(session, kind="modeling")
    assert job.status is JobStatus.PENDING
    assert job.kind == "modeling"
    assert job.result is None
    assert job.error is None


def test_get_roundtrips_created_job(session_factory):
    store = JobStore()
    with session_factory() as session:
        created = store.create(session, kind="modeling")
        fetched = store.get(session, created.job_id)
    assert fetched == created


def test_get_unknown_job_raises():
    store = JobStore()
    engine = make_engine("sqlite:///:memory:")
    create_all_tables(engine)
    with make_session_factory(engine)() as session:
        with pytest.raises(JobNotFoundError):
            store.get(session, "does-not-exist")


def test_mark_running_then_completed(session_factory):
    store = JobStore()
    with session_factory() as session:
        job = store.create(session, kind="modeling")
        running = store.mark_running(session, job.job_id)
        assert running.status is JobStatus.RUNNING

        completed = store.mark_completed(session, job.job_id, result={"status": "completed"})
        assert completed.status is JobStatus.COMPLETED
        assert completed.result == {"status": "completed"}


def test_mark_failed_records_error(session_factory):
    store = JobStore()
    with session_factory() as session:
        job = store.create(session, kind="modeling")
        failed = store.mark_failed(session, job.job_id, error="boom")
    assert failed.status is JobStatus.FAILED
    assert failed.error == "boom"
    assert failed.result is None


def test_updated_at_advances_on_status_change(session_factory):
    store = JobStore()
    with session_factory() as session:
        job = store.create(session, kind="modeling")
        completed = store.mark_completed(session, job.job_id, result={})
    assert completed.updated_at >= job.updated_at
