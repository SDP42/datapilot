"""Executors for missing-value operations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, aborted, failed, ok

Overrides = dict[str, dict[str, Any]]


def _fit_column(df: pd.DataFrame, column: str, ctx: ExecutionContext) -> pd.Series:
    return ctx.fit_slice(df)[column].dropna()


def _impute(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, *, numeric: bool
) -> ExecResult:
    column = op.target_columns[0]
    fit_values = _fit_column(df, column, ctx)
    if fit_values.empty:
        return aborted(
            df, f"No non-null values for '{column}' within the fit scope; cannot impute."
        )

    if numeric:
        fit_value: Any = float(fit_values.median())
        strategy = "median"
    else:
        counts = fit_values.value_counts()
        fit_value = min(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0]
        strategy = "mode"

    out = df.copy()
    n_filled = int(out[column].isna().sum())
    if n_filled == 0:
        return aborted(df, f"'{column}' has no missing values to impute.")
    out[column] = out[column].fillna(fit_value)

    return ok(
        out,
        f"Imputed {n_filled} missing value(s) in '{column}' with the {strategy} "
        f"({fit_value!r}), fitted on {ctx.fit_on_label}.",
        affected_rows=n_filled,
        values_changed=n_filled,
        fit_details={
            "strategy": strategy,
            "fit_on": ctx.fit_on_label,
            "fit_rows": len(fit_values),
            "fit_value": fit_value if numeric else str(fit_value),
        },
    )


def execute_numeric(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return _impute(df, op, ctx, numeric=True)


def execute_categorical(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return _impute(df, op, ctx, numeric=False)


def execute_datetime(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return failed(
        df,
        "Datetime imputation has no plan-supplied strategy. The executor will NOT invent "
        "one (no forward-fill / back-fill / arbitrary default). Re-plan with an explicit, "
        "safe strategy.",
    )


def execute_generic(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    return failed(
        df,
        "Column type could not be determined by the planner, so no concrete imputation "
        "strategy exists. Re-run the planner with a DatasetProfile.",
    )


def execute_drop_high_missing_column(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    """Implemented, but the approval layer refuses this (plan status
    ``not_safe_to_automate``). Kept so a future explicitly-safe path can reuse it."""
    column = op.target_columns[0]
    if column not in df.columns:
        return failed(df, f"Column '{column}' is not in the dataset.")
    out = df.drop(columns=[column])
    return ok(
        out,
        f"Dropped column '{column}'.",
        columns_removed=[column],
        parameters_used={"missing_percentage": op.parameters.get("missing_percentage")},
    )
