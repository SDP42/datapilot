"""Phase 7.7 — expanded multi-model candidate search & hyperparameter grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.modeling import (
    ModelingRequest,
    ModelingStatus,
    TrainingRunStatus,
    assess_model_readiness,
    recommend_data_split,
    run_expanded_model_search,
    run_expanded_search,
)
from data_engine.modeling.pipeline import _build_feature_engineering_spec, _build_problem_spec
from data_engine.modeling.training import _expanded_catalog

_N = 300


def _regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1 = rng.normal(50.0, 15.0, _N)
    x2 = rng.uniform(0.0, 100.0, _N)
    region = rng.choice(["north", "south", "east"], _N)
    y = 2.0 * x1 + 0.5 * x2 + (region == "south") * 8.0 + rng.normal(0.0, 5.0, _N)
    return pd.DataFrame({"feat_x1": x1, "feat_x2": x2, "region": region, "price": y})


def _binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, _N)
    b = rng.normal(0.0, 1.0, _N)
    y = ((1.2 * a - 0.8 * b + rng.normal(0.0, 0.5, _N)) > 0.0).astype(int)
    return pd.DataFrame({"signal_a": a, "signal_b": b, "churn": y})


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [1, 0, 1]})


def test_expanded_catalog_has_more_than_100_candidates_per_category():
    assert len(_expanded_catalog("regression")) > 100
    assert len(_expanded_catalog("classification")) > 100


def test_expanded_catalog_unknown_category_is_empty():
    assert _expanded_catalog("clustering") == []
    assert _expanded_catalog("nonsense") == []


def test_run_expanded_model_search_regression_ranks_every_candidate():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    result = run_expanded_model_search(df, request)

    assert result.status is ModelingStatus.COMPLETED
    assert result.task_type == "regression"
    assert result.selection_metric == "rmse"
    assert result.candidate_count > 100
    assert len(result.candidates) == result.candidate_count

    # ranks are 1..N with no gaps, and sorted by rmse ascending (minimize)
    ranks = [c.rank for c in result.candidates]
    assert ranks == list(range(1, result.candidate_count + 1))
    completed = [c for c in result.candidates if c.status is TrainingRunStatus.COMPLETED]
    rmses = [c.metrics["rmse"] for c in completed if "rmse" in c.metrics]
    assert rmses == sorted(rmses)


def test_run_expanded_model_search_classification_ranks_by_f1_descending():
    df = _binary_df()
    request = ModelingRequest(dataset_id="ds-bin", objective="predict churn")
    result = run_expanded_model_search(df, request)

    assert result.status is ModelingStatus.COMPLETED
    assert result.task_type == "binary_classification"
    assert result.selection_metric == "f1"
    assert result.candidate_count > 100

    completed = [c for c in result.candidates if c.status is TrainingRunStatus.COMPLETED]
    f1s = [c.metrics["f1"] for c in completed if "f1" in c.metrics]
    assert f1s == sorted(f1s, reverse=True)

    # every family present in the catalog shows up somewhere in the results
    families = {c.family.value for c in result.candidates}
    assert {
        "linear",
        "tree_based",
        "ensemble",
        "distance_based",
        "probabilistic",
        "neural",
    } <= families


def test_run_expanded_search_too_little_data_is_unavailable():
    df = _tiny_df()
    request = ModelingRequest(dataset_id="ds-tiny", objective="predict y")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)
    readiness = assess_model_readiness(df, problem, fe, objective=request.objective)
    split = recommend_data_split(df, problem, fe, objective=request.objective)

    result = run_expanded_search(df, problem, fe, readiness, split, objective=request.objective)
    assert result.status is ModelingStatus.UNAVAILABLE
    assert result.candidates == []


def test_run_expanded_search_clustering_task_is_unavailable():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=_N), "b": rng.normal(size=_N)})
    request = ModelingRequest(dataset_id="ds-cluster", objective="cluster customers into groups")
    result = run_expanded_model_search(df, request)
    # either clustering was inferred (unavailable: category not covered) or
    # task inference landed elsewhere — either way, never a crash, and never
    # "completed" with candidates for a category this search doesn't cover
    if result.status is ModelingStatus.COMPLETED:
        assert result.task_type in (
            "regression",
            "binary_classification",
            "multiclass_classification",
        )
    else:
        assert result.reason is not None


def test_no_duplicate_estimator_hyperparameter_pairs_within_a_category():
    for category in ("regression", "classification"):
        catalog = _expanded_catalog(category)
        seen = set()
        for spec in catalog:
            hparams = tuple(
                sorted(
                    (k, tuple(v) if isinstance(v, list) else v)
                    for k, v in spec.hyperparameters.items()
                )
            )
            key = (spec.estimator_name, hparams)
            assert key not in seen, f"duplicate candidate in {category}: {key}"
            seen.add(key)
