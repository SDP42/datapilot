"""Phase 15.1/15.4 — the `UserStore`: registration, authentication, and
gamification state, mirroring `ActivityStore`'s own shape (a thin wrapper
around one already-constructed `Session`, a Pydantic record type, never a
raw ORM row leaking out).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .security import hash_password, verify_password
from .user_models import UserRow

#: XP awarded once per instrumented action, by `ActivityRow.kind`. Deliberately
#: small, fixed integers — never derived from a model's own metrics (accuracy,
#: fit time, …), so XP measures *engagement with the platform*, not model
#: quality, and can never be gamed by submitting a worse model.
XP_BY_KIND: dict[str, int] = {
    "ingest": 5,
    "quality": 5,
    "eda": 10,
    "modeling": 15,
    "search": 15,
    "tune": 20,
    "train": 20,
    "predict": 5,
    "cluster": 15,
}

#: XP thresholds for level N -> N+1; level = 1 + (how many thresholds `xp` clears).
_XP_PER_LEVEL = 100


def level_for_xp(xp: int) -> int:
    return 1 + xp // _XP_PER_LEVEL


def compute_badges(xp: int, current_streak: int, longest_streak: int) -> list[str]:
    """Badges are *computed* from `xp` / streak state, never stored — they can
    never drift out of sync with the numbers that earned them."""
    badges: list[str] = []
    if xp >= 1:
        badges.append("First Steps")
    if xp >= 50:
        badges.append("Getting Started")
    if xp >= 200:
        badges.append("Data Explorer")
    if xp >= 500:
        badges.append("Data Scientist")
    if xp >= 1000:
        badges.append("DataPilot Master")
    if current_streak >= 3:
        badges.append("3-Day Streak")
    if current_streak >= 7:
        badges.append("Week Warrior")
    if longest_streak >= 30:
        badges.append("Consistency Champion")
    return badges


class UserRecord(BaseModel):
    """The JSON-serialisable, password-free view of one `UserRow`."""

    user_id: str
    username: str
    full_name: str | None = None
    experience_level: str
    primary_goal: str
    role: str | None = None
    created_at: datetime
    xp: int
    level: int
    current_streak: int
    longest_streak: int
    badges: list[str]


def _to_record(row: UserRow) -> UserRecord:
    return UserRecord(
        user_id=row.user_id,
        username=row.username,
        full_name=row.full_name,
        experience_level=row.experience_level,
        primary_goal=row.primary_goal,
        role=row.role,
        created_at=row.created_at,
        xp=row.xp,
        level=level_for_xp(row.xp),
        current_streak=row.current_streak,
        longest_streak=row.longest_streak,
        badges=compute_badges(row.xp, row.current_streak, row.longest_streak),
    )


class UsernameTakenError(Exception):
    """Raised by `UserStore.register` when the username already exists."""


class UserStore:
    """A thin wrapper around one SQLAlchemy `Session` for user CRUD."""

    def get_by_username(self, session: Session, username: str) -> UserRow | None:
        stmt = select(UserRow).where(UserRow.username == username)
        return session.execute(stmt).scalar_one_or_none()

    def register(
        self,
        session: Session,
        *,
        username: str,
        password: str,
        experience_level: str,
        primary_goal: str,
        full_name: str | None = None,
        role: str | None = None,
    ) -> UserRecord:
        if self.get_by_username(session, username) is not None:
            raise UsernameTakenError(username)
        password_hash, salt = hash_password(password)
        row = UserRow(
            user_id=str(uuid4()),
            username=username,
            password_hash=password_hash,
            password_salt=salt,
            full_name=full_name,
            experience_level=experience_level,
            primary_goal=primary_goal,
            role=role,
            created_at=datetime.now(timezone.utc),
            xp=0,
            current_streak=0,
            longest_streak=0,
            last_active_date=None,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return _to_record(row)

    def authenticate(self, session: Session, username: str, password: str) -> UserRow | None:
        row = self.get_by_username(session, username)
        if row is None:
            return None
        if not verify_password(password, row.password_hash, row.password_salt):
            return None
        return row

    def get_record(self, session: Session, user_id: str) -> UserRecord | None:
        row = session.get(UserRow, user_id)
        return _to_record(row) if row is not None else None

    def record_action(self, session: Session, user_id: str, kind: str) -> None:
        """Award XP for one instrumented action and update the daily streak.
        Unknown `kind`s award 0 XP but still count toward the streak — any
        real action on the platform counts as "active today"."""
        row = session.get(UserRow, user_id)
        if row is None:
            return
        today = datetime.now(timezone.utc).date()
        yesterday = (today - timedelta(days=1)).isoformat()
        today_str = today.isoformat()
        if row.last_active_date != today_str:
            row.current_streak = row.current_streak + 1 if row.last_active_date == yesterday else 1
            row.longest_streak = max(row.longest_streak, row.current_streak)
            row.last_active_date = today_str
        row.xp = row.xp + XP_BY_KIND.get(kind, 0)
        session.add(row)
        session.commit()


__all__ = [
    "XP_BY_KIND",
    "UserRecord",
    "UsernameTakenError",
    "UserStore",
    "compute_badges",
    "level_for_xp",
]
