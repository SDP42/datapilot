"""Phase 10.2 — permutation feature importance.

:func:`compute_permutation_importance` evaluates an **already-fitted**
scikit-learn-compatible estimator — it never fits, re-fits, or mutates
anything. Uses ``sklearn.inspection.permutation_importance`` (already a
Phase 7.4 dependency — no new one added): shuffles one feature column at
a time and measures how much the model's score degrades, averaged over
``n_repeats`` shuffles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    ExplainabilityStatus,
    ExplanationMethod,
    FeatureImportanceEntry,
    FeatureImportanceResult,
)

MODEL_EXPLAIN_RANDOM_SEED = 42
PERMUTATION_IMPORTANCE_DEFAULT_REPEATS = 10


def _unavailable(reason: str) -> FeatureImportanceResult:
    return FeatureImportanceResult(
        status=ExplainabilityStatus.UNAVAILABLE,
        method=ExplanationMethod.PERMUTATION_IMPORTANCE,
        reason=reason,
    )


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = PERMUTATION_IMPORTANCE_DEFAULT_REPEATS,
    scoring: str | None = None,
    seed: int = MODEL_EXPLAIN_RANDOM_SEED,
) -> FeatureImportanceResult:
    """Rank `feature_names` by how much shuffling each degrades `model`'s score on (X, y).

    `model` must already be fitted and expose scikit-learn's `.predict`
    (or `.score`) interface — this function never fits, re-fits, or
    mutates it. `X` must already be fully numeric (the same Phase 6.5 /
    7.4 preprocessing boundary every other DataPilot modeling entry point
    requires) with `X.shape[1] == len(feature_names)`.

    `scoring` is any scikit-learn scorer string (e.g. `'neg_root_mean_squared_error'`,
    `'f1_macro'`) — when `None`, uses the estimator's own default `.score()`.
    Never fabricates a scoring convention of its own.

    Deterministic for a fixed `seed`: `permutation_importance`'s own
    shuffling uses a seeded `numpy` RNG. Entries are ranked by
    `importance` descending (most important first); ties keep
    `feature_names`' own original order (a stable sort).

    Returns `status = unavailable` for an `X` / `feature_names` length
    mismatch, or an empty `X` — never raises for those, and never
    fabricates an importance value. A scikit-learn-raised error (e.g. the
    model is not actually fitted) propagates as-is — this function makes
    no claim about recovering from a broken estimator.
    """
    if X.ndim != 2:
        return _unavailable(f"X must be a 2D feature matrix; got shape {X.shape}")
    if X.shape[1] != len(feature_names):
        return _unavailable(
            f"X has {X.shape[1]} columns but {len(feature_names)} feature_names were supplied"
        )
    if X.shape[0] == 0:
        return _unavailable("X has no rows; nothing to permute")

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring=scoring
    )

    order = np.argsort(-result.importances_mean, kind="stable")
    entries = [
        FeatureImportanceEntry(
            feature=feature_names[i],
            importance=round(float(result.importances_mean[i]), 6),
            importance_std=round(float(result.importances_std[i]), 6),
            rank=rank,
        )
        for rank, i in enumerate(order, start=1)
    ]

    return FeatureImportanceResult(
        status=ExplainabilityStatus.COMPLETED,
        method=ExplanationMethod.PERMUTATION_IMPORTANCE,
        entries=entries,
    )


__all__ = [
    "MODEL_EXPLAIN_RANDOM_SEED",
    "PERMUTATION_IMPORTANCE_DEFAULT_REPEATS",
    "compute_permutation_importance",
]
