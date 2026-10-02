"""Phase 10.4 — partial dependence
(`explainability.partial_dependence.compute_partial_dependence`).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from explainability.contracts import ExplainabilityStatus
from explainability.partial_dependence import compute_partial_dependence


def _fitted_model(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = X[:, 0] * 3 + X[:, 1]
    return LinearRegression().fit(X, y), X


def test_returns_requested_grid_resolution():
    model, X = _fitted_model()
    result = compute_partial_dependence(model, X, "a", 0, grid_resolution=15)
    assert result.status is ExplainabilityStatus.COMPLETED
    assert result.feature == "a"
    assert len(result.points) == 15


def test_increasing_feature_increases_prediction_for_positive_coefficient():
    model, X = _fitted_model()
    result = compute_partial_dependence(model, X, "a", 0)
    values = [p.average_prediction for p in result.points]
    assert values == sorted(values)  # monotonically increasing, coefficient is positive


def test_out_of_range_feature_index_returns_unavailable():
    model, X = _fitted_model()
    result = compute_partial_dependence(model, X, "bad", 5)
    assert result.status is ExplainabilityStatus.UNAVAILABLE
    assert "out of range" in (result.reason or "")


def test_empty_X_returns_unavailable():
    model, X = _fitted_model()
    result = compute_partial_dependence(model, np.empty((0, 2)), "a", 0)
    assert result.status is ExplainabilityStatus.UNAVAILABLE


def test_does_not_mutate_model_or_data():
    model, X = _fitted_model()
    coef_before = model.coef_.copy()
    X_before = X.copy()
    compute_partial_dependence(model, X, "a", 0)
    np.testing.assert_array_equal(model.coef_, coef_before)
    np.testing.assert_array_equal(X, X_before)


def test_deterministic_repeated_calls():
    model, X = _fitted_model()
    result_a = compute_partial_dependence(model, X, "a", 0)
    result_b = compute_partial_dependence(model, X, "a", 0)
    assert result_a.model_dump_json() == result_b.model_dump_json()
