"""Phase 10.4 — partial dependence for one feature.

:func:`compute_partial_dependence` evaluates an **already-fitted**
scikit-learn-compatible estimator — it never fits, re-fits, or mutates
anything, exactly like Phase 10.2 / 10.3. Uses
``sklearn.inspection.partial_dependence`` (already a Phase 7.4
dependency — no new one added).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    ExplainabilityStatus,
    PartialDependencePoint,
    PartialDependenceResult,
)

PARTIAL_DEPENDENCE_DEFAULT_GRID_RESOLUTION = 20


def _unavailable(feature: str | None, reason: str) -> PartialDependenceResult:
    return PartialDependenceResult(
        status=ExplainabilityStatus.UNAVAILABLE, feature=feature, reason=reason
    )


def compute_partial_dependence(
    model: Any,
    X: np.ndarray,
    feature_name: str,
    feature_index: int,
    *,
    grid_resolution: int = PARTIAL_DEPENDENCE_DEFAULT_GRID_RESOLUTION,
) -> PartialDependenceResult:
    """Compute `model`'s average prediction as `feature_name` varies over a deterministic grid.

    `model` must already be fitted and expose scikit-learn's `.predict`
    (or `.predict_proba` for classification, sklearn's own `response_method='auto'`
    default) — this function never fits, re-fits, or mutates it. `X` must
    already be fully numeric, with `feature_index` the column position of
    `feature_name` in `X`.

    The grid is sklearn's own deterministic convention:
    `grid_resolution` values evenly spaced between the 5th and 95th
    percentile of `X[:, feature_index]`'s observed values — never a
    causal claim, purely an association the already-fitted model encodes
    (holding every other feature at its observed distribution, sklearn's
    `method='auto'` / `kind='average'` default).

    Returns `status = unavailable` for an out-of-range `feature_index` or
    an empty `X` — never raises for those.
    """
    if X.ndim != 2:
        return _unavailable(feature_name, f"X must be a 2D feature matrix; got shape {X.shape}")
    if X.shape[0] == 0:
        return _unavailable(feature_name, "X has no rows; nothing to compute dependence over")
    if not (0 <= feature_index < X.shape[1]):
        return _unavailable(
            feature_name,
            f"feature_index {feature_index} is out of range for X with {X.shape[1]} columns",
        )

    from sklearn.inspection import partial_dependence

    result = partial_dependence(
        model, X, [feature_index], grid_resolution=grid_resolution, kind="average"
    )
    grid_values = np.asarray(result["grid_values"][0])
    averages = np.asarray(result["average"][0])

    points = [
        PartialDependencePoint(grid_value=round(float(g), 6), average_prediction=round(float(a), 6))
        for g, a in zip(grid_values, averages, strict=True)
    ]

    return PartialDependenceResult(
        status=ExplainabilityStatus.COMPLETED, feature=feature_name, points=points
    )


__all__ = ["PARTIAL_DEPENDENCE_DEFAULT_GRID_RESOLUTION", "compute_partial_dependence"]
