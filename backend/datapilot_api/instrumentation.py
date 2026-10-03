"""Phase 15.3/15.4 — the one call every instrumented route makes after a
real result: records the run to `ActivityStore` *and* awards the
corresponding XP / streak update via `UserStore`, scoped to the actual
caller. A single helper so every route does this identically rather than
reimplementing the pairing.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from .activity_store import ActivityRecord, ActivityStore
from .auth import AuthenticatedUser
from .user_store import UserStore

_activity = ActivityStore()
_users = UserStore()


def record_activity(
    session: Session,
    current_user: AuthenticatedUser,
    *,
    kind: str,
    dataset_id: str,
    summary: str,
    dataset_filename: str | None = None,
) -> ActivityRecord:
    record = _activity.record(
        session,
        kind=kind,
        dataset_id=dataset_id,
        dataset_filename=dataset_filename,
        summary=summary,
        user_id=current_user.user_id,
    )
    _users.record_action(session, current_user.user_id, kind)
    return record


__all__ = ["record_activity"]
