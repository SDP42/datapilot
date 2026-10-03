"""Phase 14.14 — the synchronous deep-learning (MLP) training endpoint.

Mirrors `routes.modeling`'s own shape exactly: ingest an uploaded CSV or
Excel file, run `dl_engine.run_mlp_pipeline` end to end, return the real
`DLModelingResult` directly. PyTorch missing or the requested device
unavailable is reported as a structured `unavailable` result by
`run_mlp_pipeline` itself — never a raw `500`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, UploadFile
from sqlalchemy.orm import Session

from data_engine.modeling import ModelingRequest
from dl_engine import DLModelingResult, run_mlp_pipeline

from ..auth import AuthenticatedUser, get_current_user
from ..db import get_session
from ..dependencies import ingest_upload
from ..instrumentation import record_activity

router = APIRouter(
    prefix="/api/v1/dl", tags=["deep-learning"], dependencies=[Depends(get_current_user)]
)


@router.post("/train", response_model=DLModelingResult)
async def train(
    file: UploadFile,
    objective: str = Form(...),
    hidden_layer_sizes: str = Form(default="64,32"),
    epochs: int = Form(default=100, ge=1, le=1000),
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> DLModelingResult:
    """Ingest an uploaded dataset and train+evaluate one MLP on it.

    `hidden_layer_sizes` is a comma-separated list of positive integers
    (e.g. `"64,32"` for two hidden layers of 64 then 32 units) — invalid
    entries are ignored and the default `[64, 32]` is used instead.
    """
    parsed = [int(p) for p in hidden_layer_sizes.split(",") if p.strip().isdigit()]
    sizes: list[int] | None = parsed or None

    reference, df = await ingest_upload(file)
    request = ModelingRequest(dataset_id=reference.dataset_id, objective=objective)
    result = run_mlp_pipeline(df, request, hidden_layer_sizes=sizes, epochs=epochs)
    record_activity(
        session,
        current_user,
        kind="dl_train",
        dataset_id=reference.dataset_id,
        dataset_filename=reference.original_filename,
        summary=f"status={result.status.value}; MLP {'x'.join(map(str, sizes or [64, 32]))}",
    )
    return result


__all__ = ["router"]
