"""Post-Phase-7 stabilization — full Phase-5 -> Phase-7 pipeline integration.

These tests exercise :func:`run_modeling_pipeline` end to end. They verify
*composition and contracts* — not the internals of each component, which
are covered by the per-phase unit suites.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_engine import modeling
from data_engine.modeling import (
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    run_modeling_pipeline,
)
from data_engine.problem_understanding import TaskType

COMPLETED = ModelingStatus.COMPLETED
UNAVAILABLE = ModelingStatus.UNAVAILABLE
NOT_YET = ModelingStatus.NOT_YET_INFERRED
_N = 300


# --- deterministic synthetic datasets --------------------------------


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


def _multiclass_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, _N)
    b = rng.normal(0.0, 1.0, _N)
    tier = pd.cut(a + b + rng.normal(0.0, 0.3, _N), bins=3, labels=["low", "mid", "high"])
    return pd.DataFrame({"m1": a, "m2": b, "tier": tier.astype(str)})


def _clustering_df() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    half = _N // 2
    xy = np.vstack(
        [
            rng.normal(-4.0, 1.0, (half, 2)),
            rng.normal(4.0, 1.0, (_N - half, 2)),
        ]
    )
    return pd.DataFrame({"dim_x": xy[:, 0], "dim_y": xy[:, 1], "dim_z": rng.normal(0.0, 1.0, _N)})


def _forecasting_df() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    dates = pd.date_range("2022-01-01", periods=_N, freq="D")
    trend = np.linspace(0.0, 50.0, _N)
    season = 10.0 * np.sin(np.arange(_N) * 2.0 * np.pi / 30.0)
    demand = trend + season + rng.normal(0.0, 3.0, _N)
    return pd.DataFrame({"ds": dates, "exog": rng.normal(0.0, 1.0, _N), "demand": demand})


CASES = {
    "regression": (_regression_df(), "predict the price from the features", TaskType.REGRESSION),
    "binary": (
        _binary_df(),
        "classify whether the customer churns",
        TaskType.BINARY_CLASSIFICATION,
    ),
    "multiclass": (
        _multiclass_df(),
        "classify the tier into several classes",
        TaskType.MULTICLASS_CLASSIFICATION,
    ),
    "clustering": (_clustering_df(), "segment the observations into groups", TaskType.CLUSTERING),
    "forecasting": (
        _forecasting_df(),
        "forecast future demand over time",
        TaskType.TIME_SERIES_FORECASTING,
    ),
}


@pytest.fixture(params=sorted(CASES), ids=sorted(CASES))
def case(request):
    df, objective, expected_task = CASES[request.param]
    return request.param, df.copy(), objective, expected_task


# --- happy path ------------------------------------------------------


def test_public_export():
    assert modeling.run_modeling_pipeline is run_modeling_pipeline
    assert "run_modeling_pipeline" in modeling.__all__


def test_pipeline_reaches_a_recommendation(case):
    name, df, objective, _expected_task = case
    spec = run_modeling_pipeline(df, ModelingRequest(dataset_id=name, objective=objective))

    assert isinstance(spec, ModelingSpec)
    assert spec.status is COMPLETED, spec.reason
    assert spec.selection.status is COMPLETED
    assert spec.selection.selected_family is not None
    assert spec.selection.selected_estimator is not None
    assert spec.selection.selected_score is not None


def test_all_sections_populated_and_coherent(case):
    name, df, objective, _ = case
    spec = run_modeling_pipeline(df, ModelingRequest(dataset_id=name, objective=objective))

    for section in (
        spec.readiness,
        spec.split,
        spec.candidates,
        spec.training,
        spec.evaluation,
        spec.selection,
    ):
        assert section.status is COMPLETED

    assert spec.readiness.ready is True
    assert spec.candidates.candidates
    assert spec.training.runs
    # evaluation is a mirror of training — never a second metric store
    assert spec.evaluation.source == "training_outcome"
    assert spec.evaluation.evaluated_run_count == len(spec.training.runs)
    assert spec.evaluation.successful_run_count == len(spec.training.successful_runs)
    # selection winner is drawn from the actual training runs
    families = {r.family.value for r in spec.training.runs}
    assert spec.selection.selected_family in families


def test_task_type_matches_expectation(case):
    name, df, objective, expected_task = case
    spec = run_modeling_pipeline(df, ModelingRequest(dataset_id=name, objective=objective))
    assert spec.selection.selection_metric is not None
    # the pipeline's Phase-5 stage recognised the intended task
    # (checked indirectly through the split strategy / selection metric)
    if expected_task is TaskType.TIME_SERIES_FORECASTING:
        assert spec.split.strategy is not None
        assert spec.split.strategy.value == "time_ordered_holdout"
    elif expected_task in (
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    ):
        assert spec.selection.selection_metric == "f1"
    elif expected_task is TaskType.REGRESSION:
        assert spec.selection.selection_metric == "rmse"
    elif expected_task is TaskType.CLUSTERING:
        assert spec.selection.selection_metric == "silhouette_score"


def test_request_identity_preserved(case):
    name, df, objective, _ = case
    spec = run_modeling_pipeline(
        df, ModelingRequest(dataset_id=name, dataset_version_id="v7", objective=objective)
    )
    assert spec.dataset_id == name
    assert spec.dataset_version_id == "v7"
    assert spec.objective == objective
    assert spec.objective_provided is True


# --- determinism / safety ------------------------------------------


def test_deterministic_byte_identical(case):
    name, df, objective, _ = case
    req = ModelingRequest(dataset_id=name, objective=objective)
    a = run_modeling_pipeline(df, req).model_dump_json()
    b = run_modeling_pipeline(df, req).model_dump_json()
    assert a == b


def test_input_dataframe_not_mutated(case):
    name, df, objective, _ = case
    before = df.copy(deep=True)
    run_modeling_pipeline(df, ModelingRequest(dataset_id=name, objective=objective))
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(case, tmp_path, monkeypatch):
    name, df, objective, _ = case
    monkeypatch.chdir(tmp_path)
    run_modeling_pipeline(df, ModelingRequest(dataset_id=name, objective=objective))
    assert not list(tmp_path.iterdir())


def test_no_fabricated_identifier_keys(case):
    name, df, objective, _ = case
    payload = json.loads(
        run_modeling_pipeline(
            df, ModelingRequest(dataset_id=name, objective=objective)
        ).model_dump_json()
    )

    def keys(v):
        if isinstance(v, dict):
            for k, val in v.items():
                yield k
                yield from keys(val)
        elif isinstance(v, list):
            for val in v:
                yield from keys(val)

    id_keys = {k for k in keys(payload) if k.endswith("_id")}
    assert id_keys <= {"dataset_id", "dataset_version_id"}
    for banned in ("run_id", "uuid", "guid", "generated_at", "timestamp", "created_at"):
        assert banned not in set(keys(payload))


def test_row_order_invariant_for_non_temporal(case):
    name, df, objective, expected_task = case
    if expected_task is TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("forecasting row order is semantic — covered separately")
    req = ModelingRequest(dataset_id=name, objective=objective)
    base = run_modeling_pipeline(df, req).model_dump_json()
    shuffled = df.sample(frac=1.0, random_state=17).reset_index(drop=True)
    assert run_modeling_pipeline(shuffled, req).model_dump_json() == base


def test_column_order_invariant(case):
    name, df, objective, _ = case
    req = ModelingRequest(dataset_id=name, objective=objective)
    base = run_modeling_pipeline(df, req).model_dump_json()
    reordered = df[list(df.columns)[::-1]]
    assert run_modeling_pipeline(reordered, req).model_dump_json() == base


# --- unavailable cascade (no fabrication after a stopped stage) -----


def test_tiny_dataset_stops_at_readiness_no_downstream_work():
    df = _regression_df().head(8)
    spec = run_modeling_pipeline(df, ModelingRequest(dataset_id="tiny", objective="predict price"))

    assert spec.status is UNAVAILABLE
    assert spec.readiness.status is COMPLETED
    assert spec.readiness.ready is False
    # every downstream stage is explicitly unavailable — never a fabricated result
    assert spec.candidates.status is UNAVAILABLE
    assert spec.candidates.candidates == []
    assert spec.training.status is UNAVAILABLE
    assert spec.training.runs == []
    assert spec.evaluation.status is UNAVAILABLE
    assert spec.selection.status is UNAVAILABLE
    assert spec.selection.selected_family is None
    assert "readiness" in (spec.reason or "")


def test_target_only_frame_is_unavailable():
    df = pd.DataFrame({"price": np.arange(60.0)})
    spec = run_modeling_pipeline(df, ModelingRequest(dataset_id="t", objective="predict price"))
    assert spec.status is UNAVAILABLE
    assert spec.selection.selected_family is None


# --- foundation is untouched --------------------------------------


def test_understand_modeling_still_inference_free():
    spec = modeling.understand_modeling(ModelingRequest(dataset_id="d", objective="x"))
    assert spec.status is NOT_YET
    for section in (
        spec.readiness,
        spec.split,
        spec.candidates,
        spec.training,
        spec.evaluation,
        spec.selection,
    ):
        assert section.status is NOT_YET


# --- forecasting foundation: temporal recs + declared time column ------


def test_forecasting_pipeline_recommends_temporal_features_but_does_not_build_them():
    df, objective, _ = CASES["forecasting"]
    spec = run_modeling_pipeline(
        df.copy(), ModelingRequest(dataset_id="forecasting", objective=objective)
    )
    assert spec.status is COMPLETED
    assert spec.split.strategy is not None and spec.split.strategy.value == "time_ordered_holdout"
    # Phase 7.4 flags the temporal recommendations as unbuilt.
    assert any(
        "Phase 6 recommends" in n and "lag / rolling feature" in n for n in spec.training.notes
    )


def test_declared_time_column_resolves_two_datetime_column_forecasting():
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame(
        {
            "order_date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "ship_date": pd.date_range("2022-03-01", periods=n, freq="D"),
            "exog": rng.normal(0.0, 1.0, n),
            "demand": np.linspace(0.0, 60.0, n) + rng.normal(0.0, 3.0, n),
        }
    )
    obj = "forecast future demand over time"
    # without a declared time column: ambiguous -> unavailable
    ambiguous = run_modeling_pipeline(df, ModelingRequest(dataset_id="a", objective=obj))
    assert ambiguous.status is UNAVAILABLE

    # with a declared time column: resolves and reaches a recommendation
    resolved = run_modeling_pipeline(
        df, ModelingRequest(dataset_id="a", objective=obj, time_column="order_date")
    )
    assert resolved.status is COMPLETED
    assert resolved.selection.selected_family is not None


def test_modeling_request_time_column_is_additive():
    legacy = '{"dataset_id": "d"}'
    req = ModelingRequest.model_validate_json(legacy)
    assert req.time_column is None
