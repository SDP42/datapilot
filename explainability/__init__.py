"""Explainable AI (Phase 10) — feature importance, SHAP, partial dependence.

**A fitted model is never persisted anywhere in this codebase** (see
:mod:`explainability.contracts` for why) — every function here takes an
**already-fitted** scikit-learn-compatible estimator directly from the
caller, never fitting, re-fitting, or mutating it.

**Phase 10.1** establishes the foundation: :class:`ExplanationRequest` /
:class:`ExplanationReport`, returned all-``not_yet_inferred`` by
:func:`understand_explanation` — nothing computed yet.

**Phase 10.2** adds :func:`compute_permutation_importance` — ranks
features by how much shuffling each degrades an already-fitted model's
score (``sklearn.inspection.permutation_importance``, already a Phase-7.4
dependency).

**Phase 10.3** adds optional SHAP-based importance:
:func:`compute_shap_importance` (model-agnostic ``shap.Explainer``,
global importance = mean |SHAP value|). SHAP is an **optional**
dependency (the ``explain`` extra) detected by
:mod:`explainability.availability`, mirroring :mod:`dl_engine.availability`'s
``torch`` boundary exactly; every other Phase 10 capability works without
it. Both 10.2 and 10.3 populate the **same**
:class:`FeatureImportanceResult` contract.

**Phase 10.4** adds :func:`compute_partial_dependence` — one feature's
average-prediction curve over a deterministic grid
(``sklearn.inspection.partial_dependence``).

**Not wired into anything automatically**: each function is standalone;
a caller merges results into :class:`ExplanationReport`'s sections,
exactly like Phase 5/6/7's own standalone-function convention.

Out of scope for Phase 10 (until explicitly implemented): fitting a model
for the caller (explicitly out — see ``explainability.contracts``'s own
docstring); a composed ``run_explainability_pipeline``; counterfactual
explanations; LIME; natural-language explanation text (Phase 11's
concern).
"""

from __future__ import annotations

from .availability import SHAPAvailability, is_shap_available, shap_availability
from .contracts import (
    ExplainabilityStatus,
    ExplanationMethod,
    ExplanationReport,
    ExplanationRequest,
    FeatureImportanceEntry,
    FeatureImportanceResult,
    PartialDependencePoint,
    PartialDependenceResult,
    understand_explanation,
)
from .importance import compute_permutation_importance
from .partial_dependence import compute_partial_dependence
from .shap_integration import compute_shap_importance

__all__ = [
    "ExplainabilityStatus",
    "ExplanationMethod",
    "ExplanationReport",
    "ExplanationRequest",
    "FeatureImportanceEntry",
    "FeatureImportanceResult",
    "PartialDependencePoint",
    "PartialDependenceResult",
    "SHAPAvailability",
    "compute_partial_dependence",
    "compute_permutation_importance",
    "compute_shap_importance",
    "is_shap_available",
    "shap_availability",
    "understand_explanation",
]
