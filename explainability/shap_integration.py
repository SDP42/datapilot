"""Phase 10.3 — optional SHAP-based feature importance.

:func:`compute_shap_importance` evaluates an **already-fitted**
scikit-learn-compatible estimator — it never fits, re-fits, or mutates
anything, exactly like :func:`explainability.importance.compute_permutation_importance`.
Populates the **same** :class:`~explainability.contracts.FeatureImportanceResult`
contract (`method = ExplanationMethod.SHAP`), not a SHAP-specific shape,
so a caller comparing both methods' rankings does not need two different
result types.

SHAP is an **optional** dependency (the ``explain`` extra) — detected
deterministically by :mod:`explainability.availability`, mirroring
:mod:`dl_engine.availability`'s own ``torch`` boundary exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import Any

import numpy as np

from .contracts import (
    ExplainabilityStatus,
    ExplanationMethod,
    FeatureImportanceEntry,
    FeatureImportanceResult,
)


def _unavailable(reason: str) -> FeatureImportanceResult:
    return FeatureImportanceResult(
        status=ExplainabilityStatus.UNAVAILABLE, method=ExplanationMethod.SHAP, reason=reason
    )


def compute_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    *,
    _import: Callable[[str], ModuleType] = import_module,
) -> FeatureImportanceResult:
    """Rank `feature_names` by mean |SHAP value| over `X`, using `shap.Explainer`.

    `model` must already be fitted and expose `.predict` — this function
    never fits, re-fits, or mutates it. `X` must already be fully numeric
    with `X.shape[1] == len(feature_names)`, the same boundary
    :func:`~explainability.importance.compute_permutation_importance`
    requires.

    Uses the model-agnostic `shap.Explainer(model.predict, X)` path (not a
    model-specific `TreeExplainer` / `DeepExplainer`), so it works for any
    already-fitted estimator this codebase can produce (classical
    scikit-learn or a Phase-8 PyTorch module wrapped behind a `.predict`
    call) without this function needing to branch on model type. Global
    importance per feature is `mean(abs(shap_values))` across rows — a
    standard SHAP summary statistic, not an invented one.

    Returns `status = unavailable` when SHAP is not installed (nothing
    attempted), or for an `X` / `feature_names` mismatch / empty `X` —
    never raises for those. A SHAP- or model-raised error during
    explanation propagates as-is.
    """
    if X.ndim != 2:
        return _unavailable(f"X must be a 2D feature matrix; got shape {X.shape}")
    if X.shape[1] != len(feature_names):
        return _unavailable(
            f"X has {X.shape[1]} columns but {len(feature_names)} feature_names were supplied"
        )
    if X.shape[0] == 0:
        return _unavailable("X has no rows; nothing to explain")

    from .availability import shap_availability

    availability = shap_availability(_import=_import)
    if not availability.available:
        return _unavailable(availability.reason or "")

    shap = _import("shap")

    explainer = shap.Explainer(model.predict, X)
    explanation = explainer(X)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # multiclass: (n_samples, n_features, n_classes) -> average over classes
        values = values.mean(axis=2)

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs, kind="stable")
    entries = [
        FeatureImportanceEntry(
            feature=feature_names[i], importance=round(float(mean_abs[i]), 6), rank=rank
        )
        for rank, i in enumerate(order, start=1)
    ]

    return FeatureImportanceResult(
        status=ExplainabilityStatus.COMPLETED, method=ExplanationMethod.SHAP, entries=entries
    )


__all__ = ["compute_shap_importance"]
