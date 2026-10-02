"""Phase 10.2 — permutation feature importance
(`explainability.importance.compute_permutation_importance`).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from explainability.contracts import ExplainabilityStatus, ExplanationMethod
from explainability.importance import compute_permutation_importance


def _fitted_linear_model(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 5 + X[:, 1] * 0.01 + rng.normal(size=n) * 0.01  # feature 0 dominates, 2 is noise
    model = LinearRegression().fit(X, y)
    return model, X, y


def test_dominant_feature_ranked_first():
    model, X, y = _fitted_linear_model()
    result = compute_permutation_importance(model, X, y, ["a", "b", "c"])
    assert result.status is ExplainabilityStatus.COMPLETED
    assert result.method is ExplanationMethod.PERMUTATION_IMPORTANCE
    assert result.entries[0].feature == "a"
    assert result.entries[0].rank == 1
    assert len(result.entries) == 3


def test_entries_sorted_descending_by_importance():
    model, X, y = _fitted_linear_model()
    result = compute_permutation_importance(model, X, y, ["a", "b", "c"])
    importances = [e.importance for e in result.entries]
    assert importances == sorted(importances, reverse=True)
    assert [e.rank for e in result.entries] == [1, 2, 3]


def test_feature_name_mismatch_returns_unavailable():
    model, X, y = _fitted_linear_model()
    result = compute_permutation_importance(model, X, y, ["a", "b"])  # only 2 names for 3 columns
    assert result.status is ExplainabilityStatus.UNAVAILABLE
    assert "feature_names" in (result.reason or "")


def test_1d_X_returns_unavailable():
    model, X, y = _fitted_linear_model()
    result = compute_permutation_importance(model, X[:, 0], y, ["a"])
    assert result.status is ExplainabilityStatus.UNAVAILABLE


def test_empty_X_returns_unavailable():
    model, X, y = _fitted_linear_model()
    empty_X = np.empty((0, 3))
    empty_y = np.empty((0,))
    result = compute_permutation_importance(model, empty_X, empty_y, ["a", "b", "c"])
    assert result.status is ExplainabilityStatus.UNAVAILABLE


def test_does_not_mutate_model_or_data():
    model, X, y = _fitted_linear_model()
    coef_before = model.coef_.copy()
    X_before = X.copy()
    compute_permutation_importance(model, X, y, ["a", "b", "c"])
    np.testing.assert_array_equal(model.coef_, coef_before)
    np.testing.assert_array_equal(X, X_before)


def test_deterministic_for_fixed_seed():
    model, X, y = _fitted_linear_model()
    result_a = compute_permutation_importance(model, X, y, ["a", "b", "c"], seed=7)
    result_b = compute_permutation_importance(model, X, y, ["a", "b", "c"], seed=7)
    assert result_a.model_dump_json() == result_b.model_dump_json()


def test_result_is_json_roundtrippable():
    from explainability.contracts import FeatureImportanceResult

    model, X, y = _fitted_linear_model()
    result = compute_permutation_importance(model, X, y, ["a", "b", "c"])
    restored = FeatureImportanceResult.model_validate_json(result.model_dump_json())
    assert restored == result
