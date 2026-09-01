"""Phase 6.4 — deterministic, rule-based feature-selection recommendations.

:func:`recommend_feature_selection` looks at the structurally eligible
candidate features from the **Phase-6.2** :class:`FeatureInventory` and
recommends, for each, whether to **retain**, **drop**, or **review** it —
using only transparent, fixed structural / redundancy evidence.

It **recommends only**. It never alters ``df``; never selects, drops, or
transforms a real column; never rebuilds the inventory; never infers or
re-selects the target; never infers the task type (it consumes the
Phase-5.3 :class:`TaskTypeInference`); and never computes model-based
feature importance, target correlation, mutual information, ANOVA /
chi-square feature scores, permutation importance, SHAP, leakage scores,
or any predictive ranking. A recommendation is a **structural** judgement,
never a claim about model performance.

Analysis-only: ``df``, ``inventory``, and ``task_type`` are never mutated;
no file, figure, lineage, version, database, network, or LLM access.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from data_engine.problem_understanding import (
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)
from datapilot.contracts import ColumnType

from .models import (
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
    FeatureSelectionAction,
    FeatureSelectionRecommendation,
    FeatureSelectionRecommendations,
)

# --- tunables (documented in docs/feature-engineering.md) ------------------

# missing fraction at or above this -> the feature is flagged for review
FEATURE_SELECTION_HIGH_MISSING_THRESHOLD = 0.80
# a numeric candidate with at most this many distinct non-null values is
# flagged for review as near-zero variability (1 distinct is already "constant")
FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE = 2
# |Pearson r| at or above this between two numeric candidates -> structural redundancy
FEATURE_SELECTION_HIGH_CORRELATION = 0.95
# minimum finite paired observations before a correlation is computed
FEATURE_SELECTION_MIN_CORR_OBS = 3
# a categorical candidate with at least this many distinct values -> review
FEATURE_SELECTION_HIGH_CARDINALITY = 50

_SUPPORTED_TASKS = frozenset(
    {
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.CLUSTERING,
        TaskType.TIME_SERIES_FORECASTING,
    }
)

_SEPARATORS = re.compile(r"[\s_\-/]+")
_OBJECTIVE_DIM_TOKENS = frozenset({"redundant", "redundancy", "dimensionality", "fewer", "simpler"})
_OBJECTIVE_DIM_PHRASES = (
    "remove redundant",
    "redundant features",
    "avoid duplicate",
    "duplicate features",
    "reduce dimensionality",
    "simplify features",
    "keep fewer",
    "fewer features",
)

# category ranks for the fixed recommendation ordering
_CAT_STRUCTURAL_DROP = 0  # all-missing / constant / identifier-like
_CAT_DUPLICATE = 1  # exact duplicate (drop) / high correlation (review)
_CAT_REVIEW = 2  # high missingness / low variance / high cardinality
_CAT_RETAIN = 3

_NOTE_HEURISTIC = (
    "recommendations are structural / redundancy heuristics only and do not establish "
    "predictive benefit"
)
_NOTE_NOT_MODIFIED = (
    "feature selection is a recommendation only; the DataFrame has not been modified"
)
_NOTE_NO_TARGET_SCORING = (
    "no target-based scoring, model importance, or leakage analysis is performed in Phase 6.4"
)


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


def _objective_dim_intent(objective: str) -> bool:
    normalized = _normalize(objective)
    padded = f" {normalized} "
    tokens = frozenset(t for t in normalized.split() if t)
    return bool(tokens & _OBJECTIVE_DIM_TOKENS) or any(p in padded for p in _OBJECTIVE_DIM_PHRASES)


def _unavailable(reason: str, *, objective_used: bool) -> FeatureSelectionRecommendations:
    return FeatureSelectionRecommendations(
        status=FeatureEngineeringStatus.UNAVAILABLE,
        reason=reason,
        selected_features=[],
        dropped_features=[],
        review_features=[],
        recommendations=[],
        objective_used=objective_used,
        notes=[],
    )


def _column_signature(series: pd.Series, column_type: ColumnType) -> tuple:
    """A deterministic, NaN-aware value signature for exact-duplicate detection."""
    if column_type is ColumnType.NUMERIC or column_type is ColumnType.BOOLEAN:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        return ("num", tuple(None if not np.isfinite(v) else v for v in values))
    if column_type is ColumnType.DATETIME:
        arr = pd.to_datetime(series, errors="coerce").to_numpy(dtype="datetime64[ns]")
        null = pd.isna(arr)
        epoch = arr.astype("int64")
        return ("dt", tuple(None if null[i] else int(epoch[i]) for i in range(len(epoch))))
    return ("obj", tuple(None if pd.isna(v) else str(v) for v in series.to_numpy()))


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < FEATURE_SELECTION_MIN_CORR_OBS:
        return None
    x = a[mask]
    y = b[mask]
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def recommend_feature_selection(
    df: pd.DataFrame,
    inventory: FeatureInventory,
    task_type: TaskTypeInference,
    *,
    objective: str | None = None,
) -> FeatureSelectionRecommendations:
    """Deterministically recommend retain / drop / review for candidate features.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    inventory:
        The **Phase-6.2** :class:`FeatureInventory` — the sole authority
        for which columns are candidate features. A non-model raises
        ``TypeError``; it is **not mutated** and never rebuilt.
    task_type:
        The **Phase-5.3** :class:`TaskTypeInference` — consumed for the
        task type only (never re-inferred). A non-model raises
        ``TypeError``.
    objective:
        The user's objective, **verbatim and optional** — matched against
        a small fixed vocabulary to refine notes only. It never overrides
        a structural rule.

    Returns
    -------
    FeatureSelectionRecommendations
        ``status = completed`` with per-column ``recommendations`` and the
        aligned ``selected_features`` / ``dropped_features`` /
        ``review_features`` (all alphabetical); ``status = unavailable``
        when the inventory or task inference is not usable. A completed
        inventory with no candidate features yields ``status = completed``
        with empty lists and an explicit ``reason``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"recommend_feature_selection expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(inventory, FeatureInventory):
        raise TypeError(
            "recommend_feature_selection expects a FeatureInventory, "
            f"got {type(inventory).__name__}"
        )
    if not isinstance(task_type, TaskTypeInference):
        raise TypeError(
            "recommend_feature_selection expects a TaskTypeInference, "
            f"got {type(task_type).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""
    dim_intent = objective_used and objective is not None and _objective_dim_intent(objective)

    if inventory.status is not FeatureEngineeringStatus.COMPLETED:
        return _unavailable(
            "feature-selection recommendations require a completed feature inventory "
            f"(inventory status = {inventory.status.value})",
            objective_used=objective_used,
        )
    if task_type.status is not ProblemUnderstandingStatus.COMPLETED:
        return _unavailable(
            "feature-selection recommendations require a completed task-type inference "
            f"(task-type status = {task_type.status.value})",
            objective_used=objective_used,
        )
    task = task_type.task_type
    if task is None:
        return _unavailable(
            "task-type inference completed without a task type; cannot produce task-aware "
            "feature-selection recommendations",
            objective_used=objective_used,
        )
    if task not in _SUPPORTED_TASKS:
        return _unavailable(
            f"feature-selection recommendations do not support task type '{task.value}'",
            objective_used=objective_used,
        )

    column_names = [str(c) for c in df.columns]
    candidate_records = [c for c in inventory.candidates if c.candidate and not c.is_target]

    if not candidate_records:
        return FeatureSelectionRecommendations(
            status=FeatureEngineeringStatus.COMPLETED,
            reason="no structurally eligible candidate features are available for "
            "feature-selection recommendations",
            selected_features=[],
            dropped_features=[],
            review_features=[],
            recommendations=[],
            objective_used=objective_used,
            notes=[_NOTE_HEURISTIC, _NOTE_NOT_MODIFIED],
        )

    # sort candidates alphabetically for a deterministic pass
    records = sorted(candidate_records, key=lambda c: c.column)
    present = [r for r in records if r.column in column_names]
    skipped = sorted(r.column for r in records if r.column not in column_names)

    is_forecasting = task is TaskType.TIME_SERIES_FORECASTING

    # (column -> (category, action, reason, evidence))
    decisions: dict[str, tuple[int, FeatureSelectionAction, str, list[str]]] = {}

    # --- exact-duplicate groups (deterministic, NaN-aware) ---------------
    signatures: dict[str, tuple] = {}
    for record in present:
        series = df.iloc[:, column_names.index(record.column)]
        signatures[record.column] = _column_signature(series, record.column_type)
    duplicate_of: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    for column in sorted(signatures):
        sig = signatures[column]
        if sig in seen:
            duplicate_of[column] = seen[sig]
        else:
            seen[sig] = column

    # --- structural rules: first matching rule wins; else left undecided --
    undecided: list[FeatureInventoryCandidate] = []
    for record in present:
        column = record.column
        if record.all_missing:
            decisions[column] = (
                _CAT_STRUCTURAL_DROP,
                FeatureSelectionAction.DROP,
                "the column is entirely missing",
                [f"{record.n_missing} missing / 0 usable observation(s)"],
            )
            continue
        if record.constant:
            decisions[column] = (
                _CAT_STRUCTURAL_DROP,
                FeatureSelectionAction.DROP,
                "the column is constant (1 distinct non-null value)",
                ["a constant column carries no information for any model"],
            )
            continue
        if record.identifier_like:
            decisions[column] = (
                _CAT_STRUCTURAL_DROP,
                FeatureSelectionAction.DROP,
                "the Phase-6.2 inventory marked this column identifier-like",
                list(record.reasons),
            )
            continue
        if column in duplicate_of:
            keep = duplicate_of[column]
            decisions[column] = (
                _CAT_DUPLICATE,
                FeatureSelectionAction.DROP,
                f"exact structural duplicate of '{keep}' (identical observed values)",
                [
                    f"every observed value equals '{keep}' row-for-row (NaN treated as equal)",
                    f"'{keep}' is retained (alphabetically first of the duplicate group)",
                    "this is an exact structural duplicate, not a predictive judgement",
                ],
            )
            continue
        if record.missing_fraction >= FEATURE_SELECTION_HIGH_MISSING_THRESHOLD:
            decisions[column] = (
                _CAT_REVIEW,
                FeatureSelectionAction.REVIEW,
                f"very high missingness ({record.missing_fraction:.1%} of values missing)",
                [
                    (
                        f"missing fraction {record.missing_fraction:.3f} >= "
                        f"{FEATURE_SELECTION_HIGH_MISSING_THRESHOLD:g}"
                    ),
                    (
                        "this is a selection review flag, not a missing-value handling decision "
                        "(imputation is deferred to Phase 6.5)"
                    ),
                ],
            )
            continue
        if (
            record.column_type is ColumnType.NUMERIC
            and record.n_unique <= FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE
        ):
            decisions[column] = (
                _CAT_REVIEW,
                FeatureSelectionAction.REVIEW,
                f"near-zero variability ({record.n_unique} distinct non-null values)",
                [
                    (
                        f"distinct non-null values {record.n_unique} <= "
                        f"{FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE}"
                    ),
                    (
                        "a near-constant numeric feature rarely helps a model; review whether "
                        "it belongs (e.g. as a boolean indicator)"
                    ),
                ],
            )
            continue
        if (
            record.column_type is ColumnType.CATEGORICAL
            and record.n_unique >= FEATURE_SELECTION_HIGH_CARDINALITY
        ):
            decisions[column] = (
                _CAT_REVIEW,
                FeatureSelectionAction.REVIEW,
                f"very high categorical cardinality ({record.n_unique} distinct values)",
                [
                    f"distinct values {record.n_unique} >= {FEATURE_SELECTION_HIGH_CARDINALITY}",
                    (
                        "retained by default — encoding strategy is deferred to Phase 6.5; not "
                        "claimed to be useless"
                    ),
                ],
            )
            continue

        undecided.append(record)

    # --- high-correlation redundancy: only among still-undecided numeric --
    # candidates (which would otherwise all be retained), so a spurious
    # correlation on a tiny overlap with a flagged column can never arise.
    undecided_numeric = sorted(r.column for r in undecided if r.column_type is ColumnType.NUMERIC)
    numeric_values: dict[str, np.ndarray] = {
        column: pd.to_numeric(df.iloc[:, column_names.index(column)], errors="coerce").to_numpy(
            dtype=float
        )
        for column in undecided_numeric
    }
    redundant_with: dict[str, tuple[str, float]] = {}
    anchors: list[str] = []
    for column in undecided_numeric:
        matched: tuple[str, float] | None = None
        for anchor in anchors:
            corr = _pearson(numeric_values[column], numeric_values[anchor])
            if corr is not None and abs(corr) >= FEATURE_SELECTION_HIGH_CORRELATION:
                matched = (anchor, corr)
                break
        if matched is None:
            anchors.append(column)
        else:
            redundant_with[column] = matched

    for record in undecided:
        column = record.column
        if column in redundant_with:
            anchor, corr = redundant_with[column]
            decisions[column] = (
                _CAT_DUPLICATE,
                FeatureSelectionAction.REVIEW,
                f"structural redundancy: |Pearson correlation| {abs(corr):.3f} with '{anchor}'",
                [
                    (
                        f"|r| {abs(corr):.3f} >= {FEATURE_SELECTION_HIGH_CORRELATION:g} on finite "
                        "overlapping observations"
                    ),
                    (
                        f"'{anchor}' is kept as the anchor (alphabetically first of the pair); "
                        "neither is claimed to be more predictive"
                    ),
                ],
            )
            continue
        evidence = ["no deterministic structural reason to exclude this candidate"]
        if is_forecasting and record.column_type is ColumnType.DATETIME:
            evidence.append(
                "datetime feature retained; it may serve as the time index for forecasting"
            )
        decisions[column] = (
            _CAT_RETAIN,
            FeatureSelectionAction.RETAIN,
            "retained structurally (Phase 6.4 does not assess predictive usefulness)",
            evidence,
        )

    ordered = sorted(decisions.items(), key=lambda kv: (kv[1][0], kv[0]))
    recommendations = [
        FeatureSelectionRecommendation(column=col, action=act, reason=reason, evidence=evidence)
        for col, (_, act, reason, evidence) in ordered
    ]

    selected_features = sorted(
        c for c, d in decisions.items() if d[1] is FeatureSelectionAction.RETAIN
    )
    dropped_features = sorted(
        c for c, d in decisions.items() if d[1] is FeatureSelectionAction.DROP
    )
    review_features = sorted(
        c for c, d in decisions.items() if d[1] is FeatureSelectionAction.REVIEW
    )

    notes: list[str] = [
        (
            f"{len(present)} candidate feature(s) assessed for task '{task.value}': "
            f"{len(selected_features)} retained, {len(dropped_features)} dropped, "
            f"{len(review_features)} flagged for review"
        ),
        _NOTE_HEURISTIC,
        _NOTE_NOT_MODIFIED,
        _NOTE_NO_TARGET_SCORING,
    ]
    if skipped:
        notes.append(
            f"{len(skipped)} inventory candidate(s) not present in the DataFrame were skipped: "
            + ", ".join(skipped)
        )
    if dim_intent:
        notes.append(
            "objective favours dimensionality reduction; review flags are surfaced but the "
            "structural safety rules are unchanged"
        )
    elif objective_used:
        notes.append("objective recorded; no dimensionality-reduction vocabulary matched")

    return FeatureSelectionRecommendations(
        status=FeatureEngineeringStatus.COMPLETED,
        reason=None,
        selected_features=selected_features,
        dropped_features=dropped_features,
        review_features=review_features,
        recommendations=recommendations,
        objective_used=objective_used,
        notes=notes,
    )
