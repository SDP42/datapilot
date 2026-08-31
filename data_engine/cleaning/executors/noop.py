"""Executors for non-transforming operations.

These are recorded as ``skipped`` — they never touch the data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, skipped

Overrides = dict[str, dict[str, Any]]


def execute_review_outliers(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return skipped(
        df,
        "Investigation-only operation; no transformation was requested. Outliers are NOT "
        "deleted, capped, winsorized, clipped, or replaced.",
        parameters_used={"outlier_detected": True, "confirmed_error": False},
    )


def execute_recommend_imbalance_strategy(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return skipped(
        df,
        "Modeling recommendation; the dataset is not modified. No oversampling, "
        "undersampling, SMOTE, class-weight change, or target edit. The strategy belongs to "
        "the future modeling / experiment layer.",
        parameters_used={"is_data_transformation": False},
    )
