"""Executor for the natural-log distribution transform.

Strict: aborts on any value <= 0. NEVER silently substitutes log1p or
any other transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, aborted, failed, ok

Overrides = dict[str, dict[str, Any]]


def execute_log_transform(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    column = op.target_columns[0]
    series = df[column]

    if not ptypes.is_numeric_dtype(series) or ptypes.is_bool_dtype(series):
        return aborted(df, f"'{column}' is not a numeric column; cannot log-transform.")

    non_null = series.dropna()
    if non_null.empty:
        return aborted(df, f"'{column}' has no non-null values.")

    minimum_before = float(non_null.min())
    non_positive = int((non_null <= 0).sum())
    if non_positive > 0:
        return aborted(
            df,
            f"{non_positive} value(s) in '{column}' are <= 0 (minimum {minimum_before}). "
            "Natural log is undefined there. Aborting — the executor will NOT substitute "
            "log1p, Yeo-Johnson, or any other transform.",
            parameters_used={"transform": "log", "minimum_before": minimum_before},
        )

    out = df.copy()
    out[column] = np.log(out[column])
    transformed = out[column].dropna()
    if not np.isfinite(transformed).all():
        return failed(df, f"Log transform produced non-finite values in '{column}'.")

    return ok(
        out,
        f"Applied natural log to '{column}'.",
        values_changed=int(non_null.shape[0]),
        fit_details={"fit_on": ctx.fit_on_label},
        parameters_used={
            "transform": "log",
            "base": "e",
            "minimum_before": minimum_before,
            "minimum_after": float(transformed.min()),
        },
    )


def execute_review_distribution_transform(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return failed(
        df,
        "review_distribution_transform is a review placeholder — no concrete transform has "
        "been approved. Re-plan / approve a specific transform operation.",
    )
