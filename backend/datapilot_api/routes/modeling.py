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
from sqlalchemy.orm import Session

from data_engine.modeling import (
    ExpandedSearchResult,
    ModelingRequest,
    ModelingSpec,
    run_expanded_model_search,
    run_modeling_pipeline,
)

from ..auth import AuthenticatedUser, get_current_user
from ..db import get_session
from ..dependencies import ingest_upload
from ..instrumentation import record_activity

router = APIRouter(
    prefix="/api/v1/modeling", tags=["modeling"], dependencies=[Depends(get_current_user)]
)


@router.post("/run", response_model=ModelingSpec)
async def run(
    file: UploadFile,
    objective: str = Form(...),
    forecast_horizon: int = Form(default=1, ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ModelingSpec:
    """Ingest an uploaded CSV and run the full Phase-7 modeling pipeline on it."""
    reference, df = await ingest_upload(file)
    request = ModelingRequest(
        dataset_id=reference.dataset_id, objective=objective, forecast_horizon=forecast_horizon
    )
    spec = run_modeling_pipeline(df, request)
    record_activity(
        session,
        current_user,
        kind="modeling",
        dataset_id=reference.dataset_id,
        dataset_filename=reference.original_filename,
        summary=(
            f"status={spec.status.value}; selected={spec.selection.selected_estimator or 'none'}"
        ),
    )
    return spec


@router.post("/search", response_model=ExpandedSearchResult)
async def search(
    file: UploadFile,
    objective: str = Form(...),
    cross_validate: bool = Form(default=False),
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ExpandedSearchResult:
    """Ingest an uploaded CSV and fit + rank every candidate in the Phase 7.7
    expanded catalog (100+ (estimator, hyperparameter) combinations), not
    just one baseline per family. Slower than `/run` — every candidate is
    its own fit — but returns the full ranked field for comparison.

    `cross_validate=true` additionally scores every candidate with 5-fold
    cross-validation and ranks by that more reliable estimate instead of
    the single train/test split score — roughly 5x slower, since each
    candidate is fit that many more times.
    """
    reference, df = await ingest_upload(file)
    request = ModelingRequest(dataset_id=reference.dataset_id, objective=objective)
    result = run_expanded_model_search(df, request, use_cross_validation=cross_validate)
    record_activity(
        session,
        current_user,
        kind="search",
        dataset_id=reference.dataset_id,
        dataset_filename=reference.original_filename,
        summary=(
            f"status={result.status.value}; {result.candidate_count} candidates ranked"
            f"{' (cross-validated)' if cross_validate else ''}"
        ),
    )
    return result


__all__ = ["router"]
