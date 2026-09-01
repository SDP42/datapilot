"""Phase 5.2 — deterministic, structural + objective-aware target identification.

:func:`identify_target` answers "which column(s) are plausible prediction
targets?" using **only** deterministic evidence:

* the column's dtype (via the shared ``infer_column_type`` classifier),
  missingness, cardinality, and identifier-like name / behaviour;
* the user's **explicit** objective string, matched with transparent
  rules (exact phrase / separator-insensitive / significant token) — no
  LLM, no embeddings, no external call, no fuzzy matching.

It never uses correlation, mutual information, feature importance, a
model, or predictive performance. It never mutates the DataFrame, writes
a file, or touches lineage / versions.

If the evidence does not justify a single target it returns a ranked
candidate list plus an explicit ``reason`` rather than guessing.
"""

from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    ObjectiveMatchKind,
    ProblemUnderstandingStatus,
    TargetCandidate,
    TargetIdentification,
)

# --- tunables (documented) ---------------------------------------------------

TARGET_ID_ROUND = 4
FRACTION_ROUND = 6

# A single target is pinned only when the top candidate leads the second by
# at least this many score points (and scores positive), unless the objective
# already selects exactly one column.
TARGET_SELECTION_MARGIN = 20.0

# A near-perfectly-unique categorical or integer column looks like a row id.
HIGH_UNIQUE_ID_THRESHOLD = 0.99

# Column-name tokens (after splitting on space / _ / -) that mark an identifier
# when they are the whole name or its last token.
_IDENTIFIER_NAME_TOKENS = frozenset(
    {"id", "idx", "index", "key", "uuid", "guid", "pk", "rowid", "sk", "hash"}
)

# Objective words treated as filler for *token*-level matching only (they never
# block an exact / normalized match).
_OBJECTIVE_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "with",
        "from",
        "in",
        "on",
        "is",
        "are",
        "will",
        "be",
        "this",
        "that",
        "predict",
        "predicting",
        "forecast",
        "forecasting",
        "classify",
        "classifying",
        "estimate",
        "estimating",
        "determine",
        "identify",
        "detect",
        "detecting",
        "whether",
        "future",
        "target",
        "value",
        "values",
        "data",
        "dataset",
    }
)

_SEPARATORS = re.compile(r"[\s_\-]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# --- name / objective helpers ---------------------------------------------------


def _normalize_name(name: str) -> str:
    return _SEPARATORS.sub(" ", name.strip().lower()).strip()


def _compact(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


class _Objective:
    """Pre-computed objective forms (built once per call)."""

    def __init__(self, objective: str) -> None:
        self.normalized = _normalize_name(objective)
        self.padded = f" {self.normalized} "
        self.tokens = frozenset(t for t in self.normalized.split() if t)
        self.compact = _compact(objective)


def _match_objective(column: str, objective: _Objective) -> ObjectiveMatchKind:
    norm_col = _normalize_name(column)
    if not norm_col:
        return ObjectiveMatchKind.NONE
    if f" {norm_col} " in objective.padded:
        return ObjectiveMatchKind.EXACT

    compact_col = _compact(column)
    col_tokens = [t for t in norm_col.split() if t]
    if len(compact_col) >= 3 and compact_col in objective.compact:
        return ObjectiveMatchKind.NORMALIZED
    if (
        col_tokens
        and any(len(t) >= 3 for t in col_tokens)
        and all(t in objective.tokens for t in col_tokens if len(t) >= 2)
    ):
        return ObjectiveMatchKind.NORMALIZED

    significant = [t for t in col_tokens if len(t) >= 3 and t not in _OBJECTIVE_STOPWORDS]
    objective_significant = [
        t for t in objective.tokens if len(t) >= 3 and t not in _OBJECTIVE_STOPWORDS
    ]
    for col_token in significant:
        if col_token in objective.tokens:
            return ObjectiveMatchKind.TOKEN
        # transparent shared-prefix rule — one token is a prefix of the other and
        # they share at least 4 leading characters (handles churn/churned,
        # sale/sales, price/pricing). No stemmer, no fuzzy edit distance.
        for objective_token in objective_significant:
            shorter, longer = sorted((col_token, objective_token), key=len)
            if len(shorter) >= 4 and longer.startswith(shorter):
                return ObjectiveMatchKind.TOKEN
    return ObjectiveMatchKind.NONE


def _is_identifier_like(
    column: str, column_type: ColumnType, unique_fraction: float, is_float: bool
) -> bool:
    tokens = [t for t in _SEPARATORS.split(column.strip().lower()) if t]
    if tokens:
        candidate_token = tokens[0] if len(tokens) == 1 else tokens[-1]
        if candidate_token in _IDENTIFIER_NAME_TOKENS:
            return True
    if unique_fraction < HIGH_UNIQUE_ID_THRESHOLD:
        return False
    # near-unique: a categorical or *integer* column behaves like an id; a
    # high-uniqueness float column is a plausible continuous target.
    if column_type is ColumnType.CATEGORICAL:
        return True
    return column_type is ColumnType.NUMERIC and not is_float


# --- scoring ------------------------------------------------------------------


def _score_candidate(
    column_type: ColumnType,
    missing_fraction: float,
    n_unique: int,
    unique_fraction: float,
    identifier_like: bool,
    objective_match: ObjectiveMatchKind,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if identifier_like:
        score -= 40.0
        reasons.append("identifier-like column name/behaviour (-40)")
    else:
        score += 15.0
        reasons.append("not identifier-like (+15)")

    if missing_fraction == 0.0:
        score += 12.0
        reasons.append("no missing values (+12)")
    elif missing_fraction <= 0.20:
        score += 6.0
        reasons.append(f"low missingness {missing_fraction:.2%} (+6)")
    elif missing_fraction <= 0.50:
        reasons.append(f"moderate missingness {missing_fraction:.2%} (+0)")
    else:
        score -= 25.0
        reasons.append(f"mostly missing {missing_fraction:.2%} (-25)")

    if column_type is ColumnType.BOOLEAN:
        score += 18.0
        reasons.append("boolean column — classification-target shaped (+18)")
    elif column_type is ColumnType.CATEGORICAL:
        if 2 <= n_unique <= 20:
            score += 18.0
            reasons.append(f"categorical with {n_unique} classes (+18)")
        elif n_unique <= 50:
            score += 6.0
            reasons.append(f"categorical with {n_unique} classes (+6)")
        else:
            score -= 10.0
            reasons.append(f"high-cardinality categorical ({n_unique} classes) (-10)")
    elif column_type is ColumnType.NUMERIC:
        if 2 <= n_unique <= 20:
            score += 14.0
            reasons.append(f"numeric with {n_unique} distinct values — discrete target (+14)")
        elif unique_fraction > 0.5:
            score += 12.0
            reasons.append("numeric, high uniqueness — continuous target (+12)")
        else:
            score += 4.0
            reasons.append("numeric column (+4)")
    elif column_type is ColumnType.DATETIME:
        score += 4.0
        reasons.append("datetime column — eligible target (+4)")
    else:
        score -= 5.0
        reasons.append("unrecognised column type (-5)")

    if objective_match is ObjectiveMatchKind.EXACT:
        score += 60.0
        reasons.append("objective names this column exactly (+60)")
    elif objective_match is ObjectiveMatchKind.NORMALIZED:
        score += 45.0
        reasons.append("objective matches this column name (normalised) (+45)")
    elif objective_match is ObjectiveMatchKind.TOKEN:
        score += 18.0
        reasons.append("objective shares a significant token with this column (+18)")

    return round(score, TARGET_ID_ROUND), reasons


# --- public API --------------------------------------------------------------


def _unavailable(reason: str, objective_used: bool) -> TargetIdentification:
    return TargetIdentification(
        status=ProblemUnderstandingStatus.UNAVAILABLE,
        reason=reason,
        objective_used=objective_used,
    )


def identify_target(
    df: pd.DataFrame,
    *,
    objective: str | None = None,
) -> TargetIdentification:
    """Deterministically identify plausible prediction target column(s).

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    objective:
        The user's objective, **verbatim and optional**. It is used only
        for the transparent name-matching rules; it is never parsed for
        meaning and never stored back into the ``ProblemSpec``.

    Returns
    -------
    TargetIdentification
        * ``status = unavailable`` — ``df`` has no columns / no rows, or
          every column is constant or entirely missing.
        * ``status = completed`` with ``target_column`` set — a single
          defensible target (the objective named exactly one column, or
          the top candidate leads by ``TARGET_SELECTION_MARGIN``, or only
          one candidate exists).
        * ``status = completed`` with ``target_column = None`` — ranked
          ``candidates`` with an explicit ``reason`` for the ambiguity.

        ``candidates`` is always ordered best-first; ties break on the
        column name (ascending).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"identify_target expects a pandas DataFrame, got {type(df).__name__}")

    objective_ctx: _Objective | None = None
    if objective is not None and objective.strip() != "":
        objective_ctx = _Objective(objective)
    objective_used = objective_ctx is not None

    columns = [str(c) for c in df.columns]
    if not columns:
        return _unavailable("the DataFrame has no columns", objective_used)
    if len(df) == 0:
        return _unavailable("the DataFrame has no rows", objective_used)

    notes: list[str] = []
    candidates: list[TargetCandidate] = []

    for position, raw in enumerate(df.columns):
        column = str(raw)
        series = df.iloc[:, position]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_observations = n_total - n_missing
        missing_fraction = round(n_missing / n_total, FRACTION_ROUND) if n_total else 0.0

        if n_observations == 0:
            notes.append(f"'{column}' excluded: entirely missing")
            continue
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        if n_unique <= 1:
            notes.append(f"'{column}' excluded: constant value")
            continue

        column_type = infer_column_type(series)
        unique_fraction = round(n_unique / n_observations, FRACTION_ROUND)
        is_float = bool(ptypes.is_float_dtype(series))
        identifier_like = _is_identifier_like(column, column_type, unique_fraction, is_float)
        objective_match = (
            _match_objective(column, objective_ctx)
            if objective_ctx is not None
            else ObjectiveMatchKind.NONE
        )
        score, reasons = _score_candidate(
            column_type,
            missing_fraction,
            n_unique,
            unique_fraction,
            identifier_like,
            objective_match,
        )
        candidates.append(
            TargetCandidate(
                column=column,
                rank=0,  # assigned after sorting
                score=score,
                column_type=column_type,
                n_observations=n_observations,
                n_missing=n_missing,
                missing_fraction=missing_fraction,
                n_unique=n_unique,
                unique_fraction=unique_fraction,
                identifier_like=identifier_like,
                objective_match=objective_match,
                reasons=reasons,
            )
        )

    if not candidates:
        return _unavailable(
            "every column is constant or entirely missing; no plausible target", objective_used
        )

    candidates.sort(key=lambda c: (-c.score, c.column))
    ranked = [c.model_copy(update={"rank": i}) for i, c in enumerate(candidates, start=1)]
    candidate_columns = [c.column for c in ranked]

    target_column, reason = _decide_target(ranked, objective_used)
    return TargetIdentification(
        status=ProblemUnderstandingStatus.COMPLETED,
        reason=reason,
        target_column=target_column,
        candidate_columns=candidate_columns,
        candidates=ranked,
        objective_used=objective_used,
        notes=notes,
    )


def _decide_target(
    ranked: list[TargetCandidate], objective_used: bool
) -> tuple[str | None, str | None]:
    objective_hits = [
        c
        for c in ranked
        if c.objective_match in (ObjectiveMatchKind.EXACT, ObjectiveMatchKind.NORMALIZED)
    ]
    if len(objective_hits) == 1:
        return objective_hits[0].column, None
    if len(objective_hits) > 1:
        names = ", ".join(f"'{c.column}'" for c in objective_hits)
        return None, (
            f"the objective matches {len(objective_hits)} columns ({names}); "
            "cannot disambiguate a single target"
        )

    # exactly one *non-identifier* column matched the objective at any level
    # (incl. token) and it is also the top-ranked candidate -> that is the target.
    # Identifier-like columns are excluded here: an id column sharing a token with
    # the objective ("customer" in "customer_id") is not a target signal.
    any_match = [
        c
        for c in ranked
        if c.objective_match is not ObjectiveMatchKind.NONE and not c.identifier_like
    ]
    if len(any_match) == 1 and any_match[0].column == ranked[0].column:
        return ranked[0].column, None

    if len(ranked) == 1:
        return ranked[0].column, None

    top, second = ranked[0], ranked[1]
    if top.score > 0.0 and (top.score - second.score) >= TARGET_SELECTION_MARGIN:
        return top.column, None

    hint = (
        "supply an explicit objective to disambiguate"
        if not objective_used
        else "the objective did not decisively name a column"
    )
    if top.score <= 0.0:
        return None, (
            f"no candidate has positive structural evidence (top score {top.score}); {hint}"
        )
    return None, (
        f"the top candidates '{top.column}' ({top.score}) and '{second.column}' ({second.score}) "
        f"are within the selection margin of {TARGET_SELECTION_MARGIN}; {hint}"
    )
