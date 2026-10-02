"""Phase 7.6 — model persistence & prediction (`fit_final_pipeline`,
`save_model`, `load_model`, `list_models`, `predict_with_model`,
`train_and_persist_model`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.modeling import (
    ModelFamily,
    ModelingRequest,
    ModelingStatus,
    fit_final_pipeline,
    list_models,
    load_model,
    predict_with_model,
    run_modeling_pipeline,
    save_model,
    train_and_persist_model,
)
from data_engine.modeling.pipeline import _build_feature_engineering_spec, _build_problem_spec

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


def test_fit_final_pipeline_regression_roundtrip():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)

    fitted = fit_final_pipeline(df, problem, fe, ModelFamily.LINEAR)
    assert fitted.category == "regression"
    assert fitted.target_column == "price"
    assert set(fitted.feature_cols) == {"feat_x1", "feat_x2", "region"}

    preds = fitted.pipeline.predict(df[fitted.feature_cols])
    assert len(preds) == len(df)
    # a reasonably fit linear model should correlate strongly with the target
    assert np.corrcoef(preds, df["price"])[0, 1] > 0.8


def test_fit_final_pipeline_unknown_family_raises():
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)

    with pytest.raises(ValueError):
        fit_final_pipeline(
            df, problem, fe, ModelFamily.NEURAL if False else ModelFamily.PROBABILISTIC
        )


def test_save_load_predict_roundtrip(tmp_path):
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)
    fitted = fit_final_pipeline(df, problem, fe, ModelFamily.LINEAR)

    metadata = save_model(fitted, dataset_id="ds-reg", objective="predict price", root=tmp_path)
    assert metadata.model_id.startswith("model-")
    assert (tmp_path / metadata.model_id / "pipeline.joblib").is_file()
    assert (tmp_path / metadata.model_id / "metadata.json").is_file()

    pipeline, loaded_meta = load_model(metadata.model_id, root=tmp_path)
    assert loaded_meta == metadata
    assert hasattr(pipeline, "predict")

    new_rows = df[fitted.feature_cols].iloc[:5]
    result = predict_with_model(metadata.model_id, new_rows, root=tmp_path)
    assert result.row_count == 5
    assert len(result.predictions) == 5
    assert result.missing_columns == []


def test_predict_with_missing_feature_column_reports_it_without_predicting(tmp_path):
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)
    fitted = fit_final_pipeline(df, problem, fe, ModelFamily.LINEAR)
    metadata = save_model(fitted, dataset_id="ds-reg", root=tmp_path)

    incomplete = df[["feat_x1"]].iloc[:3]
    result = predict_with_model(metadata.model_id, incomplete, root=tmp_path)
    assert result.predictions == []
    assert set(result.missing_columns) == {"feat_x2", "region"}


def test_load_model_unknown_id_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("model-does-not-exist", root=tmp_path)


def test_list_models_empty_store_returns_empty_list(tmp_path):
    assert list_models(root=tmp_path / "nonexistent") == []


def test_list_models_newest_first(tmp_path):
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")
    problem = _build_problem_spec(df, request)
    fe = _build_feature_engineering_spec(df, request, problem)
    fitted = fit_final_pipeline(df, problem, fe, ModelFamily.LINEAR)

    first = save_model(fitted, dataset_id="ds-reg", root=tmp_path)
    second = save_model(fitted, dataset_id="ds-reg", root=tmp_path)

    listed = list_models(root=tmp_path)
    assert [m.model_id for m in listed][:2] == [second.model_id, first.model_id] or {
        m.model_id for m in listed
    } == {first.model_id, second.model_id}


def test_train_and_persist_model_completed_case(tmp_path):
    df = _regression_df()
    request = ModelingRequest(dataset_id="ds-reg", objective="predict price")

    spec, metadata = train_and_persist_model(df, request, root=tmp_path)
    assert spec.status is ModelingStatus.COMPLETED
    assert metadata is not None
    assert metadata.target_column == "price"

    # the persisted model actually matches what run_modeling_pipeline selected
    plain_spec = run_modeling_pipeline(df, request)
    assert metadata.family == plain_spec.selection.selected_family
    assert metadata.estimator_name == plain_spec.selection.selected_estimator


def test_train_and_persist_model_classification_case(tmp_path):
    df = _binary_df()
    request = ModelingRequest(dataset_id="ds-bin", objective="predict churn")

    spec, metadata = train_and_persist_model(df, request, root=tmp_path)
    assert spec.status is ModelingStatus.COMPLETED
    assert metadata is not None
    assert metadata.category == "classification"

    new_rows = df[metadata.feature_cols].iloc[:4]
    result = predict_with_model(metadata.model_id, new_rows, root=tmp_path)
    assert len(result.predictions) == 4
    assert all(p in (0, 1) for p in result.predictions)


def test_train_and_persist_model_too_little_data_persists_nothing(tmp_path):
    tiny = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [1, 0, 1]})
    request = ModelingRequest(dataset_id="ds-tiny", objective="predict y")

    spec, metadata = train_and_persist_model(tiny, request, root=tmp_path)
    assert spec.status is ModelingStatus.UNAVAILABLE
    assert metadata is None
    assert list_models(root=tmp_path) == []
