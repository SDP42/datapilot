"""Phase 13.3 — asynchronous job submission and polling.

`POST /jobs/modeling` returns a `job_id` immediately (`status =
pending`) and runs the actual Phase-7 modeling pipeline via
`BackgroundTasks` — for a dataset large enough that
`routes.modeling.run`'s synchronous endpoint would block too long.
`GET /jobs/{job_id}` polls the same job record Phase 13.2's `JobStore`
persists. The background task opens its **own** database session
(`backend.datapilot_api.db.make_session_factory`) rather than reusing
the request-scoped one `Depends(get_session)` provides, since that
session is closed once the HTTP response is sent — before the
background task actually runs.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from data_engine.modeling import ModelingRequest, run_modeling_pipeline

from ..db import get_engine, get_session, make_session_factory
from ..dependencies import ingest_upload
from ..job_store import JobNotFoundError, JobRecord, JobStore

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
_store = JobStore()


def _run_modeling_job(
    job_id: str, dataset_id: str, df_csv: str, objective: str, forecast_horizon: int
) -> None:
    """The actual background work for one modeling job. Opens its own DB session."""
    import io

    import pandas as pd

    session_factory = make_session_factory(get_engine())
    with session_factory() as session:
        _store.mark_running(session, job_id)
        try:
            df = pd.read_csv(io.StringIO(df_csv))
            request = ModelingRequest(
                dataset_id=dataset_id, objective=objective, forecast_horizon=forecast_horizon
            )
            spec = run_modeling_pipeline(df, request)
            _store.mark_completed(session, job_id, result=spec.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 - any underlying phase's own exception is reported, not swallowed
            _store.mark_failed(session, job_id, error=str(exc))


@router.post("/modeling", response_model=JobRecord)
async def submit_modeling_job(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    objective: str = Form(...),
    forecast_horizon: int = Form(default=1, ge=1),
    session: Session = Depends(get_session),
) -> JobRecord:
    """Ingest an uploaded CSV and submit a Phase-7 modeling run as a background job."""
    reference, df = await ingest_upload(file)
    job = _store.create(session, kind="modeling")
    background_tasks.add_task(
        _run_modeling_job,
        job.job_id,
        reference.dataset_id,
        df.to_csv(index=False),
        objective,
        forecast_horizon,
    )
    return job


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobRecord:
    """Poll one job's current status / result."""
    try:
        return _store.get(session, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


__all__ = ["router"]
