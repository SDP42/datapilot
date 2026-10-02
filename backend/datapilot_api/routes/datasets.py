"""Phase 13.1 — dataset ingestion, profiling, quality, and EDA endpoints.

Every endpoint here is **stateless**: it accepts a CSV upload, runs the
existing deterministic Phase 1/2/4 functions exactly as any other
caller in this codebase would, and returns the real result contract
directly as the response — never a paraphrase, never a second response
shape. Nothing is persisted (no job tracking, no dataset registry) —
that is Phase 13.2/13.3's concern; this module is the thin HTTP layer
over Phase 1/2/4 alone.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, UploadFile

from data_engine.eda import EDAReport, analyze_dataframe
from data_engine.profiling import DatasetProfile, profile_dataset
from data_engine.quality import QualityReport, analyze_quality
from datapilot.contracts import DatasetReference
from pydantic import BaseModel

from ..auth import get_current_user
from ..dependencies import ingest_upload

router = APIRouter(
    prefix="/api/v1/datasets", tags=["datasets"], dependencies=[Depends(get_current_user)]
)


class IngestResponse(BaseModel):
    reference: DatasetReference
    profile: DatasetProfile


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    """Ingest an uploaded CSV (Phase 1) and return its reference + profile."""
    reference, _df = await ingest_upload(file)
    profile = profile_dataset(reference)
    return IngestResponse(reference=reference, profile=profile)


@router.post("/quality", response_model=QualityReport)
async def quality(
    file: UploadFile, target_column: str | None = Form(default=None)
) -> QualityReport:
    """Ingest an uploaded CSV and run Phase-2 deterministic quality analysis on it."""
    reference, _df = await ingest_upload(file)
    return analyze_quality(reference, target_column=target_column)


@router.post("/eda", response_model=EDAReport)
async def eda(file: UploadFile) -> EDAReport:
    """Ingest an uploaded CSV and run Phase-4 deterministic EDA on it."""
    reference, df = await ingest_upload(file)
    return analyze_dataframe(df, dataset_id=reference.dataset_id)


__all__ = ["router"]
