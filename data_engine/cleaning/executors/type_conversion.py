"""Executors for text -> numeric and text -> datetime conversion.

Both follow the plan's safety contract: validate first, and never let an
invalid value silently become NaN / NaT.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, aborted, ok, skipped

Overrides = dict[str, dict[str, Any]]


def _merged(op: CleaningOperation, overrides: Overrides) -> dict[str, Any]:
    return {**op.parameters, **overrides.get(op.operation_id, {})}


def execute_to_numeric(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    column = op.target_columns[0]
    params = _merged(op, overrides)
    on_unparseable = params.get("on_unparseable", "abort_and_report")

    original = df[column]
    non_null = original.dropna()
    parsed = pd.to_numeric(non_null, errors="coerce")
    unparseable = int(parsed.isna().sum())
    parse_ratio = float(parsed.notna().mean()) if len(non_null) else 1.0
    bad_examples = non_null[parsed.isna()].astype(str).unique()[:5].tolist()

    meta = {
        "original_dtype": str(original.dtype),
        "parse_ratio": round(parse_ratio, 4),
        "unparseable_count": unparseable,
        "example_unparseable": bad_examples,
        "on_unparseable": on_unparseable,
    }

    if unparseable > 0 and on_unparseable == "abort_and_report":
        return aborted(
            df,
            f"{unparseable} non-null value(s) in '{column}' do not parse as numbers "
            f"(e.g. {bad_examples}). Aborting per on_unparseable=abort_and_report — the "
            "column is left completely unchanged.",
            parameters_used=meta,
        )

    out = df.copy()
    converted = pd.to_numeric(out[column], errors="coerce")
    introduced_na = int(converted.isna().sum() - original.isna().sum())
    if introduced_na > 0:
        return aborted(
            df,
            f"Conversion would introduce {introduced_na} new missing value(s) in '{column}'. "
            "Aborting; column unchanged.",
            parameters_used=meta,
        )
    out[column] = converted
    meta["new_dtype"] = str(out[column].dtype)
    return ok(
        out,
        f"Converted '{column}' from {original.dtype} to {out[column].dtype}.",
        parameters_used=meta,
    )


def execute_to_datetime(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    column = op.target_columns[0]
    params = _merged(op, overrides)
    fmt = params.get("format")
    on_unparseable = params.get("on_unparseable", "report_do_not_coerce")

    if not fmt:
        return skipped(
            df,
            f"No explicit date format supplied for '{column}'. The executor refuses to guess "
            "an ambiguous format (e.g. 01/02/2026 could be 1 Feb or 2 Jan). Supply "
            "operation_parameter_overrides={'"
            + op.operation_id
            + "': {'format': '<strftime>'}} to proceed.",
            parameters_used={"format_used": None},
        )

    original = df[column]
    non_null = original.dropna()
    parsed = pd.to_datetime(non_null, format=fmt, errors="coerce")
    unparseable = int(parsed.isna().sum())
    meta = {
        "original_dtype": str(original.dtype),
        "format_used": fmt,
        "unparseable_count": unparseable,
        "on_unparseable": on_unparseable,
    }

    if unparseable > 0 and on_unparseable == "report_do_not_coerce":
        return aborted(
            df,
            f"{unparseable} value(s) in '{column}' do not match format {fmt!r}. Aborting per "
            "on_unparseable=report_do_not_coerce — nothing is coerced to NaT.",
            parameters_used=meta,
        )

    out = df.copy()
    converted = pd.to_datetime(out[column], format=fmt, errors="coerce")
    introduced_nat = int(converted.isna().sum() - original.isna().sum())
    if introduced_nat > 0:
        return aborted(
            df,
            f"Conversion would introduce {introduced_nat} new NaT value(s) in '{column}'. "
            "Aborting; column unchanged.",
            parameters_used=meta,
        )
    out[column] = converted
    meta["new_dtype"] = str(out[column].dtype)
    return ok(
        out,
        f"Converted '{column}' to datetime using format {fmt!r}.",
        parameters_used=meta,
    )
