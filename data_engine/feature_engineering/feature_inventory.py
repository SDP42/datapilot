"""Phase 6.2 — deterministic structural feature inventory.

:func:`inventory_features` inspects a DataFrame **structurally** and
classifies every column as either a plausible input feature or a column
that should be excluded from feature consideration (the declared target,
a constant column, an entirely-missing column, or an identifier-like
column).

It is an **inventory / column-classification** step only. It never
transforms, encodes, scales, imputes, selects, or generates a feature;
never infers a task type; never re-selects a target; never uses
correlation, mutual information, feature importance, a model, an LLM, or
an embedding; and never decides whether a column is *predictively*
useful — only whether it is *structurally* usable.

Analysis-only: it never mutates ``df``, writes a file, creates a figure,
or touches lineage / versions.
"""

from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
)

# --- tunables (documented in docs/feature-engineering.md) -------------------

FRACTION_ROUND = 6

# A categorical / integer column this unique (or more) behaves like a row id.
# A high-uniqueness *float* column is a plausible continuous measurement and is
# never called an identifier on uniqueness alone.
HIGH_UNIQUE_ID_THRESHOLD = 0.99

_SEPARATORS = re.compile(r"[\s_\-]+")

# Column-name tokens (whole name, or first / last token) that mark an identifier.
_IDENTIFIER_NAME_TOKENS = frozenset(
    {"id", "idx", "index", "key", "uuid", "guid", "pk", "rowid", "sk", "hash"}
)


def _name_is_identifier_like(column: str) -> bool:
    tokens = [t for t in _SEPARATORS.split(str(column).strip().lower()) if t]
    if not tokens:
        return False
    if tokens[-1] in _IDENTIFIER_NAME_TOKENS:
        return True
    return len(tokens) == 1 and tokens[0] in _IDENTIFIER_NAME_TOKENS


def _identifier_like(
    column: str, column_type: ColumnType, unique_fraction: float, is_float: bool
) -> tuple[bool, str | None]:
    """Transparent, deterministic identifier detection. Returns (flag, reason)."""
    if _name_is_identifier_like(column):
        return True, f"column name '{column}' matches an identifier naming pattern"
    if unique_fraction < HIGH_UNIQUE_ID_THRESHOLD:
        return False, None
    if column_type is ColumnType.CATEGORICAL:
        return True, (
            f"near-unique categorical column (unique fraction {unique_fraction}); "
            "behaves like a row identifier"
        )
    if column_type is ColumnType.NUMERIC and not is_float:
        return True, (
            f"near-unique integer column (unique fraction {unique_fraction}); "
            "behaves like a row identifier"
        )
    # a near-unique float column is a plausible continuous measurement
    return False, None


def _unavailable(reason: str, *, objective_used: bool) -> FeatureInventory:
    return FeatureInventory(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        candidate_features=[],
        excluded_features=[],
        candidates=[],
        objective_used=objective_used,
        notes=[],
    )


def inventory_features(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    objective: str | None = None,
) -> FeatureInventory:
    """Deterministically classify every column as feature-candidate or excluded.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    target:
        The caller-declared prediction target, if any. When it names a
        column present in ``df`` that column is excluded from candidates.
        A ``target`` that does not name an existing column yields
        ``status = unavailable``. ``None`` means "no declared target" —
        one is never invented.
    objective:
        The user's objective, **verbatim and optional** — accepted as
        context only. Phase 6.2 does not use it to fabricate predictive
        usefulness; ``objective_used`` stays ``False``.

    Returns
    -------
    FeatureInventory
        ``status = completed`` with per-column ``candidates`` (alphabetical
        by name), ``candidate_features``, and ``excluded_features``;
        otherwise ``status = unavailable`` with an explicit ``reason``
        (non-string / zero columns / zero rows / unknown target).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"inventory_features expects a pandas DataFrame, got {type(df).__name__}")

    objective_used = False  # Phase 6.2 never lets the objective change a decision

    if df.shape[1] == 0:
        return _unavailable("the DataFrame has no columns", objective_used=objective_used)

    column_names = [str(c) for c in df.columns]
    non_string = [c for c in df.columns if not isinstance(c, str)]

    if target is not None and target not in column_names:
        return _unavailable(
            f"the declared target column '{target}' is not in the DataFrame",
            objective_used=objective_used,
        )

    if df.shape[0] == 0:
        return _unavailable(
            "the DataFrame has no rows; structural feature statistics cannot be computed",
            objective_used=objective_used,
        )

    n_rows = int(df.shape[0])
    records: list[FeatureInventoryCandidate] = []

    # process columns in a deterministic, positional pass; sort the output below
    for position, raw_name in enumerate(df.columns):
        name = column_names[position]
        series = df.iloc[:, position]

        non_null = series.dropna()
        n_missing = int(n_rows - non_null.shape[0])
        n_observations = int(non_null.shape[0])
        n_unique = int(non_null.nunique())
        missing_fraction = round(n_missing / n_rows, FRACTION_ROUND) if n_rows else 0.0
        unique_fraction = (
            round(n_unique / n_observations, FRACTION_ROUND) if n_observations else 0.0
        )

        column_type = infer_column_type(series)
        is_float = bool(ptypes.is_float_dtype(series))
        all_missing = n_observations == 0
        constant = (not all_missing) and n_unique <= 1

        is_target = target is not None and name == target
        identifier_like, id_reason = _identifier_like(name, column_type, unique_fraction, is_float)

        reasons: list[str] = [f"structural type: {column_type.value}"]
        candidate = True

        if is_target:
            candidate = False
            reasons.append("excluded: this is the caller-declared prediction target")
        elif all_missing:
            candidate = False
            reasons.append("excluded: the column is entirely missing")
        elif constant:
            candidate = False
            reasons.append(f"excluded: the column is constant ({n_unique} distinct non-null value)")
        elif identifier_like:
            candidate = False
            reasons.append(f"excluded: identifier-like — {id_reason}")
        else:
            if column_type is ColumnType.UNKNOWN:
                reasons.append(
                    "structurally usable but the column type is UNKNOWN; treat conservatively "
                    "in later increments"
                )
            else:
                reasons.append("structurally usable as an input feature")
            if n_missing > 0:
                reasons.append(
                    f"note: {missing_fraction:.1%} missing — retained (Phase 6.2 excludes only "
                    "entirely-missing columns)"
                )
            reasons.append(
                "structural candidate only — Phase 6.2 does not assess predictive usefulness"
            )

        if not isinstance(raw_name, str):
            reasons.append(
                f"note: non-string column name of type {type(raw_name).__name__}; "
                "coerced to str for reporting only"
            )

        records.append(
            FeatureInventoryCandidate(
                column=name,
                column_type=column_type,
                n_observations=n_observations,
                n_missing=n_missing,
                missing_fraction=missing_fraction,
                n_unique=n_unique,
                unique_fraction=unique_fraction,
                identifier_like=identifier_like,
                constant=constant,
                all_missing=all_missing,
                is_target=is_target,
                candidate=candidate,
                reasons=reasons,
            )
        )

    records.sort(key=lambda r: r.column)
    candidate_features = sorted(r.column for r in records if r.candidate)
    excluded_features = sorted(r.column for r in records if not r.candidate)

    notes: list[str] = [
        (
            f"{len(records)} column(s) inspected; {len(candidate_features)} structural feature "
            f"candidate(s), {len(excluded_features)} excluded"
        ),
        (
            "structural inventory only — Phase 6.2 identifies structurally plausible feature "
            "columns and does not determine predictive usefulness"
        ),
    ]
    if target is not None:
        notes.append(f"declared target '{target}' excluded from candidates")
    else:
        notes.append("no target declared; none was inferred")
    if objective is not None and objective.strip() != "":
        notes.append("objective recorded as context only; it did not affect any decision")
    if non_string:
        notes.append(
            f"{len(non_string)} non-string column name(s) coerced to str for reporting; "
            "the DataFrame is unchanged"
        )

    return FeatureInventory(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None,
        candidate_features=candidate_features,
        excluded_features=excluded_features,
        candidates=records,
        objective_used=objective_used,
        notes=notes,
    )
