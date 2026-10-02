"""Phase 13.7 — read-only access to the activity log.

Every synchronous run worth remembering (`routes.datasets`,
`routes.modeling`, `routes.predictions`) records one row here after it
completes (see `..activity_store.ActivityStore`). This router is the one
read endpoint over that table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..activity_store import ActivityRecord, ActivityStore
from ..auth import get_current_user
from ..db import get_session

router = APIRouter(
    prefix="/api/v1/history", tags=["history"], dependencies=[Depends(get_current_user)]
)

_activity = ActivityStore()


@router.get("", response_model=list[ActivityRecord])
async def history(
    limit: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> list[ActivityRecord]:
    """The most recent runs across ingestion, quality, EDA, modeling, training, and prediction."""
    return _activity.list_recent(session, limit=limit)


__all__ = ["router"]
