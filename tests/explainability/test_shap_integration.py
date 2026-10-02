"""Phase 10.3 — SHAP-based feature importance
(`explainability.shap_integration.compute_shap_importance`).

Environment-independent failure-path tests run in every environment via
the injectable `_import` seam. Tests that actually compute SHAP values
start with `pytest.importorskip("shap")` and skip cleanly when SHAP is
not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from explainability.contracts import ExplainabilityStatus, ExplanationMethod
from explainability.shap_integration import compute_shap_importance


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def _fitted_model(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 5 + X[:, 1] * 0.01 + rng.normal(size=n) * 0.01
    model = RandomForestRegressor(n_estimators=10, random_state=seed).fit(X, y)
    return model, X


def test_unavailable_when_shap_missing():
    model, X = _fitted_model()
    result = compute_shap_importance(model, X, ["a", "b", "c"], _import=_raise_import_error)
    assert result.status is ExplainabilityStatus.UNAVAILABLE
    assert result.method is ExplanationMethod.SHAP
    assert "shap" in (result.reason or "").lower()


def test_feature_name_mismatch_returns_unavailable_without_shap():
    model, X = _fitted_model()
    result = compute_shap_importance(model, X, ["a", "b"], _import=_raise_import_error)
    assert result.status is ExplainabilityStatus.UNAVAILABLE
    assert "feature_names" in (result.reason or "")


def test_empty_X_returns_unavailable():
    model, X = _fitted_model()
    result = compute_shap_importance(model, np.empty((0, 3)), ["a", "b", "c"])
    assert result.status is ExplainabilityStatus.UNAVAILABLE


def test_dominant_feature_ranked_first_with_real_shap():
    pytest.importorskip("shap")
    model, X = _fitted_model()
    result = compute_shap_importance(model, X, ["a", "b", "c"])
    assert result.status is ExplainabilityStatus.COMPLETED
    assert result.entries[0].feature == "a"
    assert result.entries[0].rank == 1


def test_does_not_mutate_model_or_data():
    pytest.importorskip("shap")
    model, X = _fitted_model()
    X_before = X.copy()
    compute_shap_importance(model, X, ["a", "b", "c"])
    np.testing.assert_array_equal(X, X_before)


def test_result_is_json_roundtrippable():
    pytest.importorskip("shap")
    from explainability.contracts import FeatureImportanceResult

    model, X = _fitted_model()
    result = compute_shap_importance(model, X, ["a", "b", "c"])
    restored = FeatureImportanceResult.model_validate_json(result.model_dump_json())
    assert restored == result
