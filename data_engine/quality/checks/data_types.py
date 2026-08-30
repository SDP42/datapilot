"""Check: potentially incorrect storage types.

Two patterns, both read from the profile + a *non-mutating* probe of the
data:

* numeric values stored as text  (e.g. "1,203", "45.0" in an object column)
* datetime-like values stored as text / categorical

The engine never coerces the column — it only reports that a conversion
is worth considering.
"""

from __future__ import annotations

import pandas as pd

from datapilot.contracts import ColumnType

from ..context import CheckContext
from ..models import FindingType, QualityFinding, Severity, SuggestedAction
from ..thresholds import TYPE_MISMATCH_MIN_PARSE_RATIO

CHECK_NAME = "data_types"

_TEXT_DTYPE_PREFIXES = ("object", "str", "category")


def _is_text_stored(pandas_dtype: str) -> bool:
    return pandas_dtype.lower().startswith(_TEXT_DTYPE_PREFIXES)


def _datetime_as_text(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.inferred_type is ColumnType.DATETIME and _is_text_stored(col.pandas_dtype):
            findings.append(
                QualityFinding(
                    finding_id=f"{FindingType.POTENTIAL_TYPE_MISMATCH.value}:{col.name}",
                    finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
                    severity=Severity.MEDIUM,
                    columns=[col.name],
                    observed={
                        "stored_dtype": col.pandas_dtype,
                        "looks_like": "datetime",
                    },
                    description=(
                        f"Column '{col.name}' is stored as {col.pandas_dtype} but its values "
                        "parse as dates/timestamps. It is likely a datetime column."
                    ),
                    recommended_action=SuggestedAction.CONVERT_COLUMN_TYPE,
                    confidence=0.9,
                )
            )
    return findings


def _numeric_as_text(ctx: CheckContext) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for col in ctx.profile.columns:
        if col.inferred_type is not ColumnType.CATEGORICAL:
            continue
        if not _is_text_stored(col.pandas_dtype):
            continue
        series = ctx.df[col.name].dropna()
        if series.empty:
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        parse_ratio = float(coerced.notna().mean())
        if parse_ratio < TYPE_MISMATCH_MIN_PARSE_RATIO:
            continue
        non_parsing = series[coerced.isna()].astype(str).unique()[:5].tolist()
        findings.append(
            QualityFinding(
                finding_id=f"{FindingType.POTENTIAL_TYPE_MISMATCH.value}:{col.name}",
                finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
                severity=Severity.MEDIUM,
                columns=[col.name],
                observed={
                    "stored_dtype": col.pandas_dtype,
                    "looks_like": "numeric",
                    "parse_ratio": round(parse_ratio, 4),
                    "example_non_numeric_values": non_parsing,
                },
                description=(
                    f"Column '{col.name}' is stored as text but {parse_ratio:.1%} of its "
                    "non-null values are numeric. It may be a number column stored as text."
                ),
                recommended_action=SuggestedAction.CONVERT_COLUMN_TYPE,
                confidence=round(parse_ratio, 4),
            )
        )
    return findings


def check(ctx: CheckContext) -> list[QualityFinding]:
    return _datetime_as_text(ctx) + _numeric_as_text(ctx)
