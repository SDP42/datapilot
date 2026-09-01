"""Phase 7.1 — modeling foundation & ModelingSpec contract."""

from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from data_engine import modeling
from data_engine.modeling import (
    MODEL_ENGINE_VERSION,
    DataSplitPlan,
    EvaluationResults,
    ModelCandidates,
    ModelFamily,
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    TrainingOutcome,
    understand_modeling,
)

NOT_YET = ModelingStatus.NOT_YET_INFERRED


def _spec(**kw) -> ModelingSpec:
    return understand_modeling(ModelingRequest(dataset_id="ds", **kw))


# --- public API ---------------------------------------------------------


def test_public_imports():
    assert modeling.understand_modeling is understand_modeling
    assert modeling.ModelingSpec is ModelingSpec
    assert modeling.ModelingRequest is ModelingRequest


def test_model_engine_version_constant():
    assert MODEL_ENGINE_VERSION == "1"
    assert _spec().model_engine_version == "1"


def test_status_enum_values():
    assert {s.value for s in ModelingStatus} == {"not_yet_inferred", "completed", "unavailable"}


def test_model_family_enum_values():
    assert {f.value for f in ModelFamily} == {
        "linear",
        "tree_based",
        "distance_based",
        "probabilistic",
        "ensemble",
        "neural",
    }


def test_public_exports():
    # Phase 7.1 foundation names remain exported (later increments add more).
    assert {
        "MODEL_ENGINE_VERSION",
        "DataSplitPlan",
        "EvaluationResults",
        "ModelCandidates",
        "ModelFamily",
        "ModelReadiness",
        "ModelSelection",
        "ModelingRequest",
        "ModelingSpec",
        "ModelingStatus",
        "TrainingOutcome",
        "understand_modeling",
    } <= set(modeling.__all__)


# --- request validation -----------------------------------------------


def test_valid_request_returns_spec():
    assert isinstance(_spec(), ModelingSpec)


def test_dict_raises_type_error():
    with pytest.raises(TypeError):
        understand_modeling({"dataset_id": "ds"})


def test_none_raises_type_error():
    with pytest.raises(TypeError):
        understand_modeling(None)


def test_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        understand_modeling(pd.DataFrame({"a": [1, 2]}))


def test_blank_dataset_id_raises_value_error():
    with pytest.raises(ValueError):
        understand_modeling(ModelingRequest(dataset_id=""))


def test_whitespace_dataset_id_raises_value_error():
    with pytest.raises(ValueError):
        understand_modeling(ModelingRequest(dataset_id="   "))


def test_objective_absent():
    spec = _spec()
    assert spec.objective is None
    assert spec.objective_provided is False


def test_objective_present():
    spec = _spec(objective="predict customer churn")
    assert spec.objective == "predict customer churn"
    assert spec.objective_provided is True


def test_blank_objective_preserved_not_used():
    spec = _spec(objective="")
    assert spec.objective == ""
    assert spec.objective_provided is False


def test_whitespace_objective_preserved_not_used():
    spec = _spec(objective="   ")
    assert spec.objective == "   "
    assert spec.objective_provided is False


def test_objective_preserved_verbatim():
    spec = _spec(objective="  Predict CHURN  ")
    assert spec.objective == "  Predict CHURN  "
    assert spec.objective_provided is True


def test_dataset_version_id_echoed():
    spec = understand_modeling(ModelingRequest(dataset_id="sales", dataset_version_id="sales:raw"))
    assert spec.dataset_id == "sales"
    assert spec.dataset_version_id == "sales:raw"


# --- default state ---------------------------------------------------


def test_overall_status_not_yet_inferred():
    spec = _spec()
    assert spec.status is NOT_YET
    assert spec.reason is not None
    assert "trains nothing" in spec.reason
    assert spec.notes == []


def test_all_nested_sections_not_yet_inferred():
    spec = _spec()
    assert spec.readiness.status is NOT_YET
    assert spec.split.status is NOT_YET
    assert spec.candidates.status is NOT_YET
    assert spec.training.status is NOT_YET
    assert spec.evaluation.status is NOT_YET
    assert spec.selection.status is NOT_YET


def test_bare_nested_model_defaults():
    for model in (
        ModelReadiness(),
        DataSplitPlan(),
        ModelCandidates(),
        TrainingOutcome(),
        EvaluationResults(),
        ModelSelection(),
    ):
        assert model.status is NOT_YET
        assert model.reason is None
        assert model.notes == []
    assert ModelCandidates().candidates == []


# --- non-fabrication ------------------------------------------------


def test_nothing_fabricated():
    spec = _spec(objective="predict churn")
    assert spec.candidates.candidates == []
    payload = json.dumps(json.loads(spec.model_dump_json())).lower()
    for token in (
        "logisticregression",
        "randomforest",
        "xgboost",
        "accuracy",
        "rmse",
        "roc_auc",
        "hyperparam",
        "n_estimators",
        "cv_score",
        "shap",
        "prediction",
        "importance",
        "estimator",
        "run_id",
        "uuid",
        "timestamp",
        "generated_at",
    ):
        assert token not in payload


def test_no_generated_identifier_keys():
    payload = json.loads(_spec(objective="x").model_dump_json())

    def keys(v):
        if isinstance(v, dict):
            for k, val in v.items():
                yield k
                yield from keys(val)
        elif isinstance(v, list):
            for val in v:
                yield from keys(val)

    all_keys = set(keys(payload))
    for banned in ("run_id", "uuid", "guid", "generated_at", "timestamp", "created_at"):
        assert banned not in all_keys
    assert {k for k in all_keys if k.endswith("_id")} == {"dataset_id", "dataset_version_id"}


# --- JSON ----------------------------------------------------------


def test_json_serialisable():
    assert isinstance(json.loads(_spec(objective="x").model_dump_json()), dict)


def test_json_round_trip():
    spec = _spec(objective="predict churn")
    assert ModelingSpec.model_validate_json(spec.model_dump_json()) == spec


def test_json_primitive_only():
    payload = json.loads(_spec(objective="x").model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


def test_deterministic_repeated_json():
    a = _spec(objective="predict churn").model_dump_json()
    b = _spec(objective="predict churn").model_dump_json()
    assert a == b


def test_deterministic_no_objective():
    assert _spec().model_dump_json() == _spec().model_dump_json()


# --- safety -------------------------------------------------------


def test_request_not_mutated():
    request = ModelingRequest(dataset_id="ds", dataset_version_id="ds:raw", objective="  keep  ")
    snapshot = request.model_dump_json()
    understand_modeling(request)
    assert request.model_dump_json() == snapshot


def test_no_dataframe_parameter():
    params = inspect.signature(understand_modeling).parameters
    assert list(params) == ["request"]


def test_no_files_or_figures_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    understand_modeling(ModelingRequest(dataset_id="ds", objective="x"))
    assert list(tmp_path.iterdir()) == []


# --- backward compatibility -------------------------------------


def test_phase_5_apis_still_work():
    from data_engine.problem_understanding import (
        ProblemUnderstandingRequest,
        understand_problem,
    )

    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.status.value == "not_yet_inferred"


def test_phase_6_apis_still_work():
    from data_engine.feature_engineering import (
        FeatureEngineeringRequest,
        understand_feature_engineering,
    )

    spec = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="ds"))
    assert spec.status.value == "not_yet_inferred"


def test_understand_feature_engineering_signature_unchanged():
    from data_engine.feature_engineering import understand_feature_engineering

    assert list(inspect.signature(understand_feature_engineering).parameters) == ["request"]


def test_existing_package_imports_valid():
    import importlib

    for name in (
        "data_engine.eda",
        "data_engine.problem_understanding",
        "data_engine.feature_engineering",
        "data_engine.profiling",
        "data_engine.quality",
        "data_engine.validation",
    ):
        assert importlib.import_module(name) is not None


def test_legacy_style_modeling_json_validates():
    minimal = json.dumps({"dataset_id": "ds", "objective_provided": False})
    model = ModelingSpec.model_validate_json(minimal)
    assert model.model_engine_version == "1"
    assert model.status is NOT_YET
    assert model.readiness.status is NOT_YET
    assert model.selection.status is NOT_YET
