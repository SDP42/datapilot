"""Phase 15.1 — the `users` table ORM model.

Replaces Phase 13.5's single configured operator account with real,
independent accounts: each row is one registered user, with their own
password hash, onboarding profile, and gamification state. Multiple
people can now sign up and each see only their own history
(`ActivityRow.user_id`), matching Phase 15.3.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class UserRow(Base):
    """One registered account.

    `password_hash` / `password_salt` are PBKDF2-HMAC-SHA256 output
    (`..security.hash_password`) — the plaintext password is never
    stored or logged. `experience_level` / `primary_goal` / `role` are
    free-text onboarding answers (validated at the request layer, not a
    DB-level enum — the same convention `ActivityRow.kind` already
    uses), asked once at signup so the product can eventually tailor
    itself to a beginner vs. a practitioner. `xp` / `current_streak` /
    `longest_streak` / `last_active_date` are Phase 15.4's gamification
    state, updated once per instrumented action
    (`..user_store.UserStore.record_action`); badges are *computed* from
    this state on read (`..user_store.compute_badges`), never stored, so
    they can never drift from the numbers that justify them.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    password_salt: Mapped[str] = mapped_column(String(32))
    full_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    experience_level: Mapped[str] = mapped_column(String(32))
    primary_goal: Mapped[str] = mapped_column(String(64))
    role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    xp: Mapped[int] = mapped_column(Integer, default=0)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_date: Mapped[str | None] = mapped_column(String(10), nullable=True)


__all__ = ["UserRow"]
