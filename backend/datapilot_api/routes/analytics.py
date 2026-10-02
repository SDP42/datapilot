"""Phase 13.4 — the optional DuckDB analytics endpoint.

Always registered (so the API surface stays discoverable / documented
even without the `analytics` extra installed) — reports `503` with an
explicit reason when DuckDB isn't available, rather than the route
silently not existing.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.analytics import AnalyticsQueryError, query_experiments
from backend.availability import is_duckdb_available
from experimentation import ExperimentStore
from experimentation.store import ExperimentNotFoundError

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


class AnalyticsQueryRequest(BaseModel):
    experiment_ids: list[str]
    sql: str


class AnalyticsQueryResponse(BaseModel):
    rows: list[dict]


@router.post("/experiments/query", response_model=AnalyticsQueryResponse)
def query(request: AnalyticsQueryRequest) -> AnalyticsQueryResponse:
    """Run a read-only SQL query (table name `experiments`) over the named, already-recorded experiments."""
    if not is_duckdb_available():
        raise HTTPException(
            status_code=503,
            detail="DuckDB is not installed; install the optional 'analytics' extra "
            "(pip install 'datapilot[analytics]')",
        )

    store = ExperimentStore.default()
    try:
        records = [store.get(experiment_id) for experiment_id in request.experiment_ids]
    except ExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        rows = query_experiments(records, request.sql)
    except AnalyticsQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalyticsQueryResponse(rows=rows)


__all__ = ["router"]
