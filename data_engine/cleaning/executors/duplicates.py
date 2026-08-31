"""Executor for exact duplicate-row removal."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, ok

Overrides = dict[str, dict[str, Any]]


def execute_remove_exact_duplicates(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    """Remove rows that are exact, full-row duplicates of an earlier row.

    ONLY ``df.duplicated(keep="first")`` across every column. No partial
    duplicates, no key guessing, no fuzzy matching.
    """
    dup_mask = df.duplicated(keep="first")
    n_removed = int(dup_mask.sum())
    out = df.loc[~dup_mask].reset_index(drop=True)
    return ok(
        out,
        f"Removed {n_removed} exact duplicate row(s) (kept first occurrence).",
        affected_rows=n_removed,
        parameters_used={
            "scope": "exact_full_row_duplicates",
            "keep": "first",
            "rows_before": len(df),
            "rows_after": len(out),
            "duplicates_removed": n_removed,
        },
    )
