"""Phase 14.13 — the clustering expanded search
(`training.run_expanded_clustering_search` / `pipeline.run_clustering_search`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_engine.modeling import (
    ModelingRequest,
    ModelingStatus,
    TrainingRunStatus,
    run_clustering_search,
)
from data_engine.modeling.training import _expanded_clustering_catalog

_PER_BLOB = 100


def _three_blobs_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    centers = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    xs, ys = [], []
    for cx, cy in centers:
        xs.append(rng.normal(cx, 1.0, _PER_BLOB))
        ys.append(rng.normal(cy, 1.0, _PER_BLOB))
    return pd.DataFrame({"x1": np.concatenate(xs), "x2": np.concatenate(ys)})


def test_clustering_catalog_has_more_than_50_candidates():
    assert len(_expanded_clustering_catalog()) > 50


def test_run_clustering_search_finds_the_real_cluster_structure():
    df = _three_blobs_df()
    request = ModelingRequest(dataset_id="ds-cluster", objective="cluster customers into segments")
    result = run_clustering_search(df, request)

    assert result.status is ModelingStatus.COMPLETED
    assert result.task_type == "clustering"
    assert result.selection_metric == "silhouette_score"
    assert result.candidate_count > 50
    assert len(result.candidates) == result.candidate_count

    # ranks are 1..N with no gaps, sorted by silhouette descending (maximize)
    ranks = [c.rank for c in result.candidates]
    assert ranks == list(range(1, result.candidate_count + 1))
    completed = [c for c in result.candidates if c.status is TrainingRunStatus.COMPLETED]
    silhouettes = [
        c.metrics["silhouette_score"] for c in completed if "silhouette_score" in c.metrics
    ]
    assert silhouettes == sorted(silhouettes, reverse=True)

    # the winner should have correctly found 3 clusters (the true structure)
    best = result.candidates[0]
    assert best.hyperparameters.get("n_clusters") == 3
    assert best.metrics["silhouette_score"] > 0.5

    # every completed candidate reports all three clustering metrics
    for c in completed:
        assert {"silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"} <= set(
            c.metrics
        )
        assert c.fit_seconds >= 0.0
    assert result.total_fit_seconds > 0.0


def test_run_clustering_search_every_family_present():
    df = _three_blobs_df()
    request = ModelingRequest(dataset_id="ds-cluster", objective="cluster customers into segments")
    result = run_clustering_search(df, request)

    families = {c.family.value for c in result.candidates}
    estimators = {c.estimator_name for c in result.candidates}
    assert {"distance_based", "probabilistic"} <= families
    assert {"KMeans", "AgglomerativeClustering", "DBSCAN", "GaussianMixture"} <= estimators


def test_run_clustering_search_rejects_non_clustering_objective():
    df = _three_blobs_df()
    df["y"] = df["x1"] * 2 + df["x2"]
    request = ModelingRequest(dataset_id="ds-reg", objective="predict y")
    result = run_clustering_search(df, request)

    assert result.status is ModelingStatus.UNAVAILABLE
    assert result.candidates == []


def test_run_clustering_search_too_little_data_is_unavailable():
    df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    request = ModelingRequest(dataset_id="ds-tiny", objective="cluster rows into groups")
    result = run_clustering_search(df, request)

    assert result.status is ModelingStatus.UNAVAILABLE
    assert result.candidates == []
