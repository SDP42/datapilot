"""Phase 13.6 — train-and-persist, list, and predict with a saved model.

Phase 7.6 (`data_engine.modeling.persistence`) is the first part of this
codebase allowed to write a fitted estimator to disk and load it back for
inference on new data. These three endpoints are the HTTP surface for
that: train a model and keep it (`POST /train`), see what is saved
(`GET /models`), and run it against new, unseen rows
(`POST /models/{model_id}/predict`). Stateless and synchronous, exactly
like `routes.modeling` — for a dataset large enough that training blocks
too long for one request, submit it as a background job instead
(`routes.jobs`), which is unaffected by this router.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from data_engine.modeling import (
    ModelingRequest,
    ModelingSpec,
    PersistedModelMetadata,
    PredictionResult,
    list_models,
    predict_with_model,
    train_and_persist_model,
)

from backend.settings import get_settings

from ..auth import AuthenticatedUser, get_current_user
from ..db import get_session
from ..dependencies import ingest_upload
from ..instrumentation import record_activity

router = APIRouter(
    prefix="/api/v1/predict", tags=["predict"], dependencies=[Depends(get_current_user)]
)


def _store_root() -> Path:
    return Path(get_settings().trained_model_dir)


class TrainAndSaveResponse(ModelingSpec):
    """`ModelingSpec` plus the persisted model, when training succeeded."""

    model: PersistedModelMetadata | None = None


@router.post("/train", response_model=TrainAndSaveResponse)
async def train_and_save(
    file: UploadFile,
    objective: str = Form(...),
    forecast_horizon: int = Form(default=1, ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> TrainAndSaveResponse:
    """Ingest a CSV, run the full Phase-7 pipeline, and persist the selected model.

    Returns the same `ModelingSpec` `POST /api/v1/modeling/run` would, plus
    `model` — populated only when `status == "completed"` and persistence
    actually succeeded; `null` otherwise, with the `ModelingSpec` itself
    explaining why (its own `reason` field).
    """
    reference, df = await ingest_upload(file)
    request = ModelingRequest(
        dataset_id=reference.dataset_id, objective=objective, forecast_horizon=forecast_horizon
    )
    spec, metadata = train_and_persist_model(df, request, root=_store_root())
    record_activity(
        session,
        current_user,
        kind="train",
        dataset_id=reference.dataset_id,
        dataset_filename=reference.original_filename,
        summary=(f"status={spec.status.value}; model={metadata.model_id if metadata else 'none'}"),
    )
    return TrainAndSaveResponse(**spec.model_dump(), model=metadata)


@router.get("/models", response_model=list[PersistedModelMetadata])
async def models() -> list[PersistedModelMetadata]:
    """Every persisted model, newest first."""
    return list_models(root=_store_root())


@router.post("/models/{model_id}/predict", response_model=PredictionResult)
async def predict(
    model_id: str,
    file: UploadFile,
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> PredictionResult:
    """Run a persisted model against new, unseen rows from an uploaded CSV."""
    reference, df = await ingest_upload(file)
    try:
        result = predict_with_model(model_id, df, root=_store_root())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    record_activity(
        session,
        current_user,
        kind="predict",
        dataset_id=reference.dataset_id,
        dataset_filename=reference.original_filename,
        summary=f"predicted {result.row_count} row(s) with model {model_id}",
    )
    return result


__all__ = ["router"]
