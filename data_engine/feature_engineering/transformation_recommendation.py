"""Phase 6.3 — deterministic, rule-based transformation recommendations.

:func:`recommend_transformations` examines the DataFrame structure and the
**Phase-6.2** :class:`FeatureInventory` and recommends transformations
that the *observed structure* makes worth considering — a log transform
for a strictly-positive feature spanning a large multiplicative range, a
set of calendar derivations for a datetime feature, and so on.

It **recommends only**. It never creates, replaces, renames, encodes,
scales, imputes, bins, or otherwise modifies a column; never selects
features; never infers a target or a task type; never uses a model,
correlation, mutual information, feature importance, cross-validation,
statistical test, embedding, LLM, or external call. A recommendation
never means "this will improve model performance" — only "the observed
structure makes this transformation worth considering".

Analysis-only: ``df`` and ``inventory`` are never mutated; no file,
figure, lineage, version, database, or network access.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    TransformationRecommendation,
    TransformationRecommendations,
)

# --- tunables (documented in docs/feature-engineering.md) ------------------

# |skewness| at or above this is treated as evidence for a monotonic transform.
# A deterministic engineering heuristic — NOT a statistically optimal value.
TRANSFORMATION_SKEW_THRESHOLD = 1.0
# |skewness| at or above this is "strong" skew (log / log1p / reciprocal tier).
TRANSFORMATION_STRONG_SKEW_THRESHOLD = 2.0
# A strictly-positive feature whose max / min reaches this spans a large
# multiplicative range — evidence for log scaling regardless of skew.
TRANSFORMATION_LOG_RANGE_RATIO = 1000.0
# A numeric feature whose largest absolute value exceeds this is flagged for
# scaling / standardisation as a *recommendation category* (never executed).
TRANSFORMATION_SCALING_MAGNITUDE = 1000.0
# |mean| <= this * std is treated as "distributed around zero" (abs-value tier).
TRANSFORMATION_ABS_SYMMETRY_RATIO = 0.1
# Minimum non-missing finite observations before any numeric rule fires.
TRANSFORMATION_MIN_OBS = 3

_SEPARATORS = re.compile(r"[\s_\-/]+")

# stable ordering of the operation categories Phase 6.3 emits
_OPERATION_PRIORITY: dict[FeatureOperationType, int] = {
    FeatureOperationType.TRANSFORMATION: 0,
    FeatureOperationType.DATETIME_DERIVATION: 1,
    FeatureOperationType.NUMERICAL_SCALING: 2,
}

_SKEW_INTENT_TOKENS = frozenset({"skew", "skewed", "skewness", "lognormal"})
_SKEW_INTENT_PHRASES = ("reduce skew", "log transform", "log scale", "normal distribution")
_CYCLICAL_INTENT_TOKENS = frozenset({"cyclical", "seasonal", "seasonality", "periodic"})
_CYCLICAL_INTENT_PHRASES = ("time of day", "day of week", "day of the week")
_SCALING_INTENT_TOKENS = frozenset(
    {"scale", "scaling", "standardize", "standardise", "normalize", "normalise"}
)


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


class _Objective:
    def __init__(self, objective: str) -> None:
        self.normalized = _normalize(objective)
        self.padded = f" {self.normalized} "
        self.tokens = frozenset(t for t in self.normalized.split() if t)

    @property
    def skew_intent(self) -> bool:
        return bool(self.tokens & _SKEW_INTENT_TOKENS) or any(
            p in self.padded for p in _SKEW_INTENT_PHRASES
        )

    @property
    def cyclical_intent(self) -> bool:
        return bool(self.tokens & _CYCLICAL_INTENT_TOKENS) or any(
            p in self.padded for p in _CYCLICAL_INTENT_PHRASES
        )

    @property
    def scaling_intent(self) -> bool:
        return bool(self.tokens & _SCALING_INTENT_TOKENS)


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _skew(values: np.ndarray) -> float | None:
    if values.size < TRANSFORMATION_MIN_OBS:
        return None
    skew = float(pd.Series(values).skew())
    return None if math.isnan(skew) else skew


def _unavailable(reason: str, *, objective_used: bool) -> TransformationRecommendations:
    return TransformationRecommendations(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        recommended_operations=[],
        recommendations=[],
        objective_used=objective_used,
        notes=[],
    )


def _numeric_recommendations(
    column: str,
    values: np.ndarray,
    is_integer: bool,
    has_missing: bool,
    objective: _Objective | None,
) -> list[TransformationRecommendation]:
    n = int(values.size)
    if n < 2:
        return []

    vmin = float(values.min())
    vmax = float(values.max())
    has_zero = bool((values == 0.0).any())
    has_negative = vmin < 0.0
    has_positive = vmax > 0.0
    strictly_positive = vmin > 0.0
    strictly_negative = vmax < 0.0
    non_negative = vmin >= 0.0
    skew = _skew(values)
    skew_mag = abs(skew) if skew is not None else 0.0
    range_ratio = (vmax / vmin) if strictly_positive else None
    skew_intent = objective is not None and objective.skew_intent
    strong_bar = (
        TRANSFORMATION_SKEW_THRESHOLD if skew_intent else TRANSFORMATION_STRONG_SKEW_THRESHOLD
    )

    recs: list[TransformationRecommendation] = []
    monotonic_chosen = False

    def _missing_evidence() -> list[str]:
        if not has_missing:
            return []
        return [
            (
                "recommendation is based on observed non-missing values; missing-value "
                "handling is deferred to Phase 6.5"
            )
        ]

    # --- monotonic transform: at most one, strict priority ----------------
    big_range = range_ratio is not None and range_ratio >= TRANSFORMATION_LOG_RANGE_RATIO
    strong_skew = skew is not None and skew >= strong_bar

    if strictly_positive and (big_range or strong_skew):
        evidence = [f"all {n} usable values are strictly positive (min {vmin:g})"]
        if big_range:
            evidence.append(
                f"max / min multiplicative range is {range_ratio:g} "
                f">= {TRANSFORMATION_LOG_RANGE_RATIO:g}"
            )
        if strong_skew:
            evidence.append(f"right skew {skew:.3f} >= {strong_bar:g}")
        if skew_intent:
            evidence.append("objective requests skew reduction")
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.TRANSFORMATION,
                description="log transform",
                reason=(
                    "positive numeric feature spans a large multiplicative range and/or is "
                    "strongly right-skewed; a log transform may reduce scale dominance"
                ),
                evidence=evidence + _missing_evidence(),
            )
        )
        monotonic_chosen = True
    elif strictly_negative and skew_mag >= strong_bar and not has_zero:
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.TRANSFORMATION,
                description="reciprocal transform",
                reason=(
                    "strictly-negative, strongly-skewed feature with no zero values; a "
                    "reciprocal transform maps it to a more balanced range"
                ),
                evidence=[
                    f"all {n} usable values are strictly negative (max {vmax:g})",
                    f"skew magnitude {skew_mag:.3f} >= {strong_bar:g}",
                    "no zero values — reciprocal domain is satisfied",
                ]
                + _missing_evidence(),
            )
        )
        monotonic_chosen = True
    elif (not strictly_positive) and vmin > -1.0 and (has_zero or has_negative) and strong_skew:
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.TRANSFORMATION,
                description="log1p transform",
                reason=(
                    "right-skewed non-negative-ish feature containing zero; log1p is defined "
                    "for every usable value and may reduce scale dominance"
                ),
                evidence=[
                    f"minimum usable value {vmin:g} > -1 — log1p domain is satisfied",
                    "contains zero values, so a plain log transform is not applicable"
                    if has_zero
                    else "contains small negative values above -1",
                    f"right skew {skew:.3f} >= {strong_bar:g}",
                ]
                + _missing_evidence(),
            )
        )
        monotonic_chosen = True
    elif (
        non_negative
        and skew is not None
        and TRANSFORMATION_SKEW_THRESHOLD <= skew < TRANSFORMATION_STRONG_SKEW_THRESHOLD
    ):
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.TRANSFORMATION,
                description="square-root transform",
                reason=(
                    "non-negative feature with moderate right skew; a square-root transform "
                    "is a milder alternative to log worth considering"
                ),
                evidence=[
                    f"all {n} usable values are non-negative (min {vmin:g})",
                    (
                        f"moderate right skew {skew:.3f} in "
                        f"[{TRANSFORMATION_SKEW_THRESHOLD:g}, "
                        f"{TRANSFORMATION_STRONG_SKEW_THRESHOLD:g})"
                    ),
                ]
                + (["integer-valued (count-like data)"] if is_integer else [])
                + _missing_evidence(),
            )
        )
        monotonic_chosen = True

    # --- absolute value: signed feature centred on zero -------------------
    if has_negative and has_positive:
        std = float(values.std())
        mean = float(values.mean())
        if std > 0.0 and abs(mean) <= TRANSFORMATION_ABS_SYMMETRY_RATIO * std:
            recs.append(
                TransformationRecommendation(
                    column=column,
                    operation=FeatureOperationType.TRANSFORMATION,
                    description="absolute-value transform",
                    reason=(
                        "feature has both positive and negative values distributed around "
                        "zero; an absolute-value transform captures magnitude"
                    ),
                    evidence=[
                        f"values span {vmin:g} to {vmax:g} (both signs present)",
                        (
                            f"|mean| {abs(mean):g} <= "
                            f"{TRANSFORMATION_ABS_SYMMETRY_RATIO:g} * std {std:g}"
                        ),
                    ]
                    + _missing_evidence(),
                )
            )

    # --- scaling: recommendation category only (never executed) ----------
    largest_magnitude = max(abs(vmin), abs(vmax))
    scaling_intent = objective is not None and objective.scaling_intent
    if not monotonic_chosen and (
        largest_magnitude > TRANSFORMATION_SCALING_MAGNITUDE or scaling_intent
    ):
        evidence = [f"largest absolute value {largest_magnitude:g}"]
        if scaling_intent:
            evidence.append("objective mentions scaling / standardisation")
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.NUMERICAL_SCALING,
                description="scaling / standardisation (recommendation only)",
                reason=(
                    "feature values span a wide numeric magnitude; scaling or standardisation "
                    "is worth considering — Phase 6.3 does not perform it"
                ),
                evidence=evidence + _missing_evidence(),
            )
        )

    return recs


_BASE_DATETIME_PARTS = ("year", "month", "day", "day_of_week", "day_of_year", "quarter")


def _datetime_recommendations(
    column: str, series: pd.Series, has_missing: bool, objective: _Objective | None
) -> list[TransformationRecommendation]:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return []

    has_time_of_day = bool(
        (parsed.dt.hour != 0).any()
        or (parsed.dt.minute != 0).any()
        or (parsed.dt.second != 0).any()
    )
    parts = list(_BASE_DATETIME_PARTS)
    if has_time_of_day:
        parts.append("hour")

    missing_evidence: list[str] = (
        [
            (
                "derivations use observed non-missing timestamps; missing-value handling is "
                "deferred to Phase 6.5"
            )
        ]
        if has_missing
        else []
    )

    recs: list[TransformationRecommendation] = []
    for part in parts:
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.DATETIME_DERIVATION,
                description=f"derive {part}",
                reason="datetime feature with usable values; calendar components are worth "
                "considering as separate features",
                evidence=[f"{int(parsed.shape[0])} usable timestamp(s)"] + missing_evidence,
            )
        )

    cyclical_parts = ["month", "day_of_week"]
    if has_time_of_day:
        cyclical_parts.append("hour")
    for part in cyclical_parts:
        evidence = [f"{part} is periodic"]
        if objective is not None and objective.cyclical_intent:
            evidence.append("objective mentions cyclical / seasonal structure")
        recs.append(
            TransformationRecommendation(
                column=column,
                operation=FeatureOperationType.DATETIME_DERIVATION,
                description=f"cyclical (sin/cos) {part}",
                reason="datetime feature; a cyclical (sin/cos) encoding preserves periodic "
                "continuity across the wrap-around",
                evidence=evidence + missing_evidence,
            )
        )

    return recs


def recommend_transformations(
    df: pd.DataFrame,
    inventory: FeatureInventory,
    *,
    objective: str | None = None,
) -> TransformationRecommendations:
    """Deterministically recommend transformations worth considering.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    inventory:
        The **Phase-6.2** :class:`FeatureInventory` — the sole authority
        for which columns are candidate features and which are excluded. A
        non-model raises ``TypeError``; it is **not mutated** and never
        rebuilt.
    objective:
        The user's objective, **verbatim and optional** — matched against
        a small fixed vocabulary to refine (never override) recommendation
        priority. Never parsed for meaning.

    Returns
    -------
    TransformationRecommendations
        ``status = completed`` with structured ``recommendations`` and the
        aligned ``recommended_operations`` (both deterministically
        ordered); ``status = unavailable`` when the inventory is not
        completed. A completed inventory with no candidate features yields
        ``status = completed`` with empty lists and an explicit ``reason``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recommend_transformations expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(inventory, FeatureInventory):
        raise TypeError(
            f"recommend_transformations expects a FeatureInventory, got {type(inventory).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""
    objective_ctx = _Objective(objective) if objective_used and objective is not None else None

    if inventory.status is not FeatureEngineeringStatus.COMPLETED:
        return _unavailable(
            "transformation recommendations require a completed feature inventory "
            f"(inventory status = {inventory.status.value})",
            objective_used=objective_used,
        )

    column_names = [str(c) for c in df.columns]
    candidate_records = [c for c in inventory.candidates if c.candidate]

    if not candidate_records:
        return TransformationRecommendations(
            status=FeatureEngineeringStatus.COMPLETED,
            reason=(
                "no structurally eligible feature columns are available for transformation "
                "recommendations"
            ),
            recommended_operations=[],
            recommendations=[],
            objective_used=objective_used,
            notes=[
                (
                    "recommendations are structural heuristics only and do not establish "
                    "predictive benefit"
                ),
                "transformations are recommendations only; the DataFrame has not been modified",
            ],
        )

    recommendations: list[TransformationRecommendation] = []
    n_numeric = n_datetime = n_categorical = n_boolean = 0
    any_missing_used = False
    skipped_missing_columns: list[str] = []

    for record in candidate_records:
        name = record.column
        if name not in column_names:
            skipped_missing_columns.append(name)
            continue
        series = df.iloc[:, column_names.index(name)]
        has_missing = int(record.n_missing) > 0

        if record.column_type is ColumnType.NUMERIC:
            n_numeric += 1
            values = _finite(series)
            is_integer = bool(pd.api.types.is_integer_dtype(series))
            col_recs = _numeric_recommendations(
                name, values, is_integer, has_missing, objective_ctx
            )
            if col_recs and has_missing:
                any_missing_used = True
            recommendations.extend(col_recs)
        elif record.column_type is ColumnType.DATETIME:
            n_datetime += 1
            col_recs = _datetime_recommendations(name, series, has_missing, objective_ctx)
            if col_recs and has_missing:
                any_missing_used = True
            recommendations.extend(col_recs)
        elif record.column_type is ColumnType.CATEGORICAL:
            n_categorical += 1
        elif record.column_type is ColumnType.BOOLEAN:
            n_boolean += 1

    recommendations.sort(key=lambda r: (r.column, _OPERATION_PRIORITY[r.operation], r.description))
    recommended_operations = [f"{r.column}: {r.description}" for r in recommendations]

    n_rec_columns = len({r.column for r in recommendations})
    notes: list[str] = [
        (
            f"{len(candidate_records)} candidate feature(s) considered; "
            f"{len(recommendations)} transformation recommendation(s) across "
            f"{n_rec_columns} column(s)"
        ),
        "recommendations are structural heuristics only and do not establish predictive benefit",
        "transformations are recommendations only; the DataFrame has not been modified",
    ]
    if n_categorical:
        notes.append(
            f"{n_categorical} categorical candidate(s): categorical encoding is deferred to "
            "later Phase 6 components (not recommended here)"
        )
    if n_boolean:
        notes.append(f"{n_boolean} boolean candidate(s): no transformation is required")
    if any_missing_used:
        notes.append(
            "some recommendations are based on observed non-missing values; missing-value "
            "handling is deferred to Phase 6.5"
        )
    if skipped_missing_columns:
        notes.append(
            f"{len(skipped_missing_columns)} inventory candidate(s) not present in the "
            "DataFrame were skipped: " + ", ".join(sorted(skipped_missing_columns))
        )
    if objective_ctx is not None:
        notes.append(
            "objective matched a fixed refinement vocabulary; it adjusted priority only and "
            "never overrode a mathematical domain"
            if (
                objective_ctx.skew_intent
                or objective_ctx.cyclical_intent
                or objective_ctx.scaling_intent
            )
            else "objective recorded; no refinement vocabulary matched"
        )

    return TransformationRecommendations(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None,
        recommended_operations=recommended_operations,
        recommendations=recommendations,
        objective_used=objective_used,
        notes=notes,
    )
