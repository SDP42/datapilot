"""Phase 14.12 — opt-in RandomizedSearchCV deep-tune over one named
catalog estimator (`training.tune_best_candidate` / `pipeline.run_deep_tune`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.modeling import (
    ModelFamily,
    ModelingRequest,
    ModelingStatus,
    run_deep_tune,
    run_expanded_model_search,
)

_N = 300


def _regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1 = rng.normal(50.0, 15.0, _N)
    x2 = rng.uniform(0.0, 100.0, _N)
    y = 2.0 * x1 + 0.5 * x2 + rng.normal(0.0, 5.0, _N)
    return pd.DataFrame({"x1": x1, "x2": x2, "price": y})


def _binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, _N)
    b = rng.normal(0.0, 1.0, _N)
    y = ((1.2 * a - 0.8 * b + rng.normal(0.0, 0.5, _N)) > 0.0).astype(int)
    return pd.DataFrame({"a": a, "b": b, "churn": y})


def test_deep_tune_regression_completes_and_is_reproducible():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-tune-reg", objective="predict price")
    result_a = run_deep_tune(
        df, request, family=ModelFamily.ENSEMBLE, estimator_name="RandomForestRegressor"
    )
    result_b = run_deep_tune(
        df, request, family=ModelFamily.ENSEMBLE, estimator_name="RandomForestRegressor"
    )

    assert result_a.status is ModelingStatus.COMPLETED
    assert result_a.estimator_name == "RandomForestRegressor"
    assert {"mse", "rmse", "mae", "r2"} <= set(result_a.metrics)
    assert result_a.n_iterations == 30
    assert result_a.fit_seconds > 0.0
    # same data + same fixed seed -> identical tuned result
    assert result_a.best_hyperparameters == result_b.best_hyperparameters
    assert result_a.metrics == result_b.metrics


def test_deep_tune_classification_completes():
    df = _binary_df()
    request = ModelingRequest(dataset_id="ds-tune-clf", objective="predict churn")
    result = run_deep_tune(df, request, family=ModelFamily.NEURAL, estimator_name="MLPClassifier")

    assert result.status is ModelingStatus.COMPLETED
    assert {"accuracy", "precision", "recall", "f1"} <= set(result.metrics)
    assert "hidden_layer_sizes" in result.best_hyperparameters


def test_deep_tune_estimator_with_no_hyperparameters_is_unavailable():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-tune-reg", objective="predict price")
    result = run_deep_tune(
        df, request, family=ModelFamily.LINEAR, estimator_name="LinearRegression"
    )

    assert result.status is ModelingStatus.UNAVAILABLE
    assert "no deep-tune hyperparameter space" in (result.reason or "")


def test_deep_tune_estimator_not_in_catalog_is_unavailable():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-tune-reg", objective="predict price")
    result = run_deep_tune(
        df, request, family=ModelFamily.LINEAR, estimator_name="NotARealEstimator"
    )

    assert result.status is ModelingStatus.UNAVAILABLE
    assert "not in the Phase 7.7 catalog" in (result.reason or "")


def test_deep_tune_best_candidate_from_real_search_result():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-tune-reg", objective="predict price")
    search = run_expanded_model_search(df, request)
    best = search.candidates[0]

    result = run_deep_tune(df, request, family=best.family, estimator_name=best.estimator_name)
    # the winner of a 100+ candidate search is always a real catalog member,
    # so deep-tune should either complete or report "no hyperparameters" —
    # never "not in the catalog".
    assert result.status is ModelingStatus.COMPLETED or "not in the Phase 7.7 catalog" not in (
        result.reason or ""
    )
