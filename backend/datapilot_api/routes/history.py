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
from ..auth import AuthenticatedUser, get_current_user
from ..db import get_session

router = APIRouter(
    prefix="/api/v1/history", tags=["history"], dependencies=[Depends(get_current_user)]
)

_activity = ActivityStore()


@router.get("", response_model=list[ActivityRecord])
async def history(
    limit: int = Query(default=50, ge=1, le=500),
    current_user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[ActivityRecord]:
    """The signed-in user's own most recent runs across ingestion, quality,
    EDA, modeling, training, and prediction — each account's history is its
    own; nothing here is shared across profiles."""
    return _activity.list_recent(session, limit=limit, user_id=current_user.user_id)


__all__ = ["router"]
