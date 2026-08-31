"""Executors for categorical formatting cleanup.

Never applies semantic mappings (``USA -> United States``). Only the
transformations the operation explicitly represents.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from data_engine.cleaning.models import CleaningOperation

from .base import ExecResult, ExecutionContext, aborted, ok, skipped

Overrides = dict[str, dict[str, Any]]
_WS = re.compile(r"\s+")


def _merged(op: CleaningOperation, overrides: Overrides) -> dict[str, Any]:
    return {**op.parameters, **overrides.get(op.operation_id, {})}


def execute_trim_whitespace(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    column = op.target_columns[0]
    params = _merged(op, overrides)
    normalization = params.get("normalization", ["strip"])

    original = df[column]

    def _clean(value: object) -> object:
        if not isinstance(value, str):
            return value
        result = value
        if "strip" in normalization:
            result = result.strip()
        if "collapse_internal_whitespace" in normalization:
            result = _WS.sub(" ", result)
        return result

    cleaned = original.map(_clean)
    changed = int(((cleaned != original) & original.notna()).sum())

    out = df.copy()
    out[column] = cleaned
    return ok(
        out,
        f"Trimmed whitespace in '{column}' ({changed} value(s) changed). No case changes, "
        "no label merging beyond whitespace.",
        values_changed=changed,
        parameters_used={
            "normalization": normalization,
            "unique_values_before": int(original.nunique(dropna=True)),
            "unique_values_after": int(cleaned.nunique(dropna=True)),
        },
    )


def execute_standardize_formatting(
    df: pd.DataFrame, op: CleaningOperation, ctx: ExecutionContext, overrides: Overrides
) -> ExecResult:
    column = op.target_columns[0]
    params = _merged(op, overrides)

    if params.get("semantic_mapping"):
        explicit = params.get("explicit_mapping")
        if not explicit:
            return aborted(
                df,
                "semantic_mapping is requested but no explicit_mapping was supplied. The "
                "executor will not invent semantic mappings.",
            )
        mapping: dict[Any, Any] = dict(explicit)
    else:
        groups: dict[str, list[str]] = params.get("variant_groups") or {}
        if not groups:
            return skipped(df, "No variant_groups supplied; nothing to standardize.")
        canonical_choice = params.get("canonical_choice", "most_frequent_variant")
        counts = df[column].value_counts()
        mapping = {}
        for variants in groups.values():
            if canonical_choice == "most_frequent_variant":
                canonical = min(variants, key=lambda v: (-int(counts.get(v, 0)), str(v)))
            elif canonical_choice == "first":
                canonical = variants[0]
            else:
                return aborted(df, f"Unknown canonical_choice {canonical_choice!r}.")
            for variant in variants:
                if variant != canonical:
                    mapping[variant] = canonical

    out = df.copy()
    changed = int(out[column].isin(list(mapping)).sum())
    out[column] = out[column].replace(mapping)
    return ok(
        out,
        f"Standardized {len(mapping)} formatting variant(s) in '{column}' "
        f"({changed} value(s) changed). No semantic mapping.",
        values_changed=changed,
        parameters_used={
            "mapping": {str(k): str(v) for k, v in mapping.items()},
            "semantic_mapping": bool(params.get("semantic_mapping")),
            "unique_values_before": int(df[column].nunique(dropna=True)),
            "unique_values_after": int(out[column].nunique(dropna=True)),
        },
    )
