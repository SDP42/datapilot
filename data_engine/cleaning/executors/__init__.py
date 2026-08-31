"""One executor module per operation family.

Each exposes ``execute(df, op, ctx, overrides) -> ExecResult``. Registered
in ``data_engine.cleaning.executor.EXECUTORS``.
"""

from __future__ import annotations

from .base import ExecResult, ExecutionContext

__all__ = ["ExecResult", "ExecutionContext"]
