"""Phase 14.14 — the MLP pipeline orchestration (`dl_engine.pipeline.run_mlp_pipeline`),
the first place an MLP is reachable end to end from a raw DataFrame + objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.modeling import ModelingRequest, TrainingRunStatus
from data_engine.problem_understanding import TaskType
from dl_engine import is_torch_available, run_mlp_pipeline

pytestmark = pytest.mark.skipif(
    not is_torch_available(), reason="PyTorch is not installed in this environment"
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


def test_mlp_pipeline_regression_completes_with_real_metrics():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-mlp-reg", objective="predict price")
    result = run_mlp_pipeline(df, request, epochs=100)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.task_type is TaskType.REGRESSION
    assert result.training is not None and result.training.status is TrainingRunStatus.COMPLETED
    assert result.evaluation is not None
    assert result.evaluation.status is TrainingRunStatus.COMPLETED
    assert {"rmse", "mae", "r2"} <= set(result.evaluation.metrics)
    # a real, reasonably-fit model on this near-linear synthetic data
    assert result.evaluation.metrics["r2"] > 0.5


def test_mlp_pipeline_binary_classification_completes():
    df = _binary_df()
    request = ModelingRequest(dataset_id="ds-mlp-clf", objective="predict churn")
    result = run_mlp_pipeline(df, request, epochs=100)

    assert result.status is TrainingRunStatus.COMPLETED
    assert result.task_type is TaskType.BINARY_CLASSIFICATION
    assert result.evaluation is not None
    assert {"accuracy", "precision", "recall", "f1"} <= set(result.evaluation.metrics)
    assert result.evaluation.metrics["accuracy"] > 0.6


def test_mlp_pipeline_respects_custom_hidden_layer_sizes():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-mlp-reg", objective="predict price")
    result = run_mlp_pipeline(df, request, hidden_layer_sizes=[8, 4], epochs=20)

    assert result.status is TrainingRunStatus.COMPLETED


def test_mlp_pipeline_unavailable_for_a_clustering_objective():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=_N), "b": rng.normal(size=_N)})
    request = ModelingRequest(dataset_id="ds-cluster", objective="cluster rows into segments")
    result = run_mlp_pipeline(df, request)

    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.reason is not None


def test_mlp_pipeline_unavailable_for_too_little_data():
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    request = ModelingRequest(dataset_id="ds-tiny", objective="predict y")
    result = run_mlp_pipeline(df, request)

    assert result.status is TrainingRunStatus.UNAVAILABLE
    assert result.reason is not None


def test_mlp_pipeline_drops_rows_with_missing_target():
    df = _regression_df().copy()
    rng = np.random.default_rng(3)
    df.loc[rng.choice(_N, 20, replace=False), "price"] = np.nan
    request = ModelingRequest(dataset_id="ds-mlp-missing", objective="predict price")
    result = run_mlp_pipeline(df, request, epochs=50)

    assert result.status is TrainingRunStatus.COMPLETED
