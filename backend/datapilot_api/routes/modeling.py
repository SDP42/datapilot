"""Phase 13.1 — the synchronous modeling endpoint.

Stateless, exactly like `routes.datasets`: ingest an uploaded CSV
(Phase 1) and run the existing Phase-7 `run_modeling_pipeline` end to
end, returning the real `ModelingSpec` directly. For a dataset large
enough that this blocks too long for a single HTTP request, see
Phase 13.3's asynchronous job endpoint (`routes.jobs`) instead — this
endpoint is deliberately kept for the common, fast case.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, UploadFile

from data_engine.modeling import ModelingRequest, ModelingSpec, run_modeling_pipeline

from ..auth import get_current_user
from ..dependencies import ingest_upload

router = APIRouter(
    prefix="/api/v1/modeling", tags=["modeling"], dependencies=[Depends(get_current_user)]
)


@router.post("/run", response_model=ModelingSpec)
async def run(
    file: UploadFile,
    objective: str = Form(...),
    forecast_horizon: int = Form(default=1, ge=1),
) -> ModelingSpec:
    """Ingest an uploaded CSV and run the full Phase-7 modeling pipeline on it."""
    reference, df = await ingest_upload(file)
    request = ModelingRequest(
        dataset_id=reference.dataset_id, objective=objective, forecast_horizon=forecast_horizon
    )
    return run_modeling_pipeline(df, request)


__all__ = ["router"]
