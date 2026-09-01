"""Phase 6.1 — feature-engineering foundation & FeatureEngineeringSpec contract."""

from __future__ import annotations

import json

import pytest

import data_engine.feature_engineering as fe
from data_engine.feature_engineering import (
    FEATURE_ENGINEERING_ENGINE_VERSION,
    FeatureEngineeringAssessment,
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    FeatureSelectionRecommendations,
    PreprocessingRequirements,
    TransformationRecommendations,
    understand_feature_engineering,
)

NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED


def _spec(**kw) -> FeatureEngineeringSpec:
    return understand_feature_engineering(FeatureEngineeringRequest(dataset_id="ds", **kw))


# --- public imports / enums ------------------------------------------------


def test_public_imports():
    assert fe.understand_feature_engineering is understand_feature_engineering
    assert fe.FeatureEngineeringSpec is FeatureEngineeringSpec
    assert fe.FEATURE_ENGINEERING_ENGINE_VERSION == "1"


def test_status_enum_values():
    assert {s.value for s in FeatureEngineeringStatus} == {
        "not_yet_inferred",
        "completed",
        "unavailable",
    }


def test_operation_type_enum_values():
    assert {o.value for o in FeatureOperationType} == {
        "transformation",
        "interaction",
        "aggregation",
        "datetime_derivation",
        "categorical_encoding",
        "numerical_scaling",
        "missing_value_handling",
        "feature_selection",
    }


def test_engine_version_constant():
    assert FEATURE_ENGINEERING_ENGINE_VERSION == "1"
    assert _spec().feature_engineering_engine_version == "1"


# --- request validation --------------------------------------------------


def test_non_request_raises_type_error():
    with pytest.raises(TypeError):
        understand_feature_engineering({"dataset_id": "ds"})


def test_non_request_none_raises_type_error():
    with pytest.raises(TypeError):
        understand_feature_engineering(None)


def test_blank_dataset_id_raises_value_error():
    with pytest.raises(ValueError):
        understand_feature_engineering(FeatureEngineeringRequest(dataset_id="   "))


def test_empty_dataset_id_raises_value_error():
    with pytest.raises(ValueError):
        understand_feature_engineering(FeatureEngineeringRequest(dataset_id=""))


def test_valid_request_returns_spec():
    assert isinstance(_spec(), FeatureEngineeringSpec)


# --- model defaults ----------------------------------------------------


def test_overall_status_not_yet_inferred():
    spec = _spec()
    assert spec.status is NOT_YET
    assert spec.reason is not None
    assert "does not yet perform feature engineering" in spec.reason
    assert spec.notes == []


def test_echoes_dataset_identity():
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="sales", dataset_version_id="sales:raw")
    )
    assert spec.dataset_id == "sales"
    assert spec.dataset_version_id == "sales:raw"


def test_nested_sections_all_not_yet_inferred():
    spec = _spec()
    assert spec.inventory.status is NOT_YET
    assert spec.transformations.status is NOT_YET
    assert spec.selection.status is NOT_YET
    assert spec.preprocessing.status is NOT_YET
    assert spec.assessment.status is NOT_YET


def test_nested_sections_not_fabricated():
    spec = _spec()
    assert spec.inventory.candidate_features == []
    assert spec.inventory.excluded_features == []
    assert spec.transformations.recommended_operations == []
    assert spec.selection.selected_features == []
    assert spec.selection.dropped_features == []
    assert spec.preprocessing.required_operations == []
    assert spec.preprocessing.encoding_required is False
    assert spec.preprocessing.scaling_required is False
    assert spec.preprocessing.imputation_required is False
    assert spec.assessment.feasible is None
    assert spec.assessment.blocking_issues == []
    assert spec.assessment.warnings == []


def test_bare_nested_model_defaults():
    for model in (
        FeatureInventory(),
        TransformationRecommendations(),
        FeatureSelectionRecommendations(),
        PreprocessingRequirements(),
        FeatureEngineeringAssessment(),
    ):
        assert model.status is NOT_YET
        assert model.reason is None
        assert model.notes == []
    assert FeatureEngineeringAssessment().feasible is None


# --- objective handling ---------------------------------------------


def test_no_objective():
    spec = _spec()
    assert spec.objective is None
    assert spec.objective_provided is False


def test_real_objective():
    spec = _spec(objective="predict customer churn")
    assert spec.objective == "predict customer churn"
    assert spec.objective_provided is True


def test_blank_objective_preserved_verbatim():
    spec = _spec(objective="   ")
    assert spec.objective == "   "
    assert spec.objective_provided is False


def test_empty_objective_preserved_verbatim():
    spec = _spec(objective="")
    assert spec.objective == ""
    assert spec.objective_provided is False


def test_objective_not_stripped_when_provided():
    spec = _spec(objective="  predict churn  ")
    assert spec.objective == "  predict churn  "
    assert spec.objective_provided is True


# --- JSON --------------------------------------------------------


def test_json_serialisable():
    payload = _spec(objective="x").model_dump_json()
    assert isinstance(json.loads(payload), dict)


def test_json_round_trip():
    spec = _spec(objective="predict churn")
    assert FeatureEngineeringSpec.model_validate_json(spec.model_dump_json()) == spec


def test_json_primitive_only():
    payload = json.loads(_spec(objective="x").model_dump_json())

    def check(value):
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
        else:
            assert value is None or isinstance(value, (str, int, float, bool))

    check(payload)


def test_no_timestamp_or_uuid_fields():
    payload = json.loads(_spec(objective="x").model_dump_json())

    def keys(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from keys(v)
        elif isinstance(value, list):
            for v in value:
                yield from keys(v)

    all_keys = set(keys(payload))
    for banned in ("generated_at", "timestamp", "created_at", "uuid", "guid", "run_id"):
        assert banned not in all_keys
    # the only identity keys are the explicit echoed request fields
    assert {k for k in all_keys if k.endswith("_id")} == {"dataset_id", "dataset_version_id"}


# --- determinism ----------------------------------------------


def test_repeated_calls_byte_identical():
    a = _spec(objective="predict churn").model_dump_json()
    b = _spec(objective="predict churn").model_dump_json()
    assert a == b


def test_repeated_calls_no_objective_byte_identical():
    assert _spec().model_dump_json() == _spec().model_dump_json()


def test_distinct_requests_distinct_specs():
    assert _spec(objective="a") != _spec(objective="b")


# --- safety -------------------------------------------------


def test_request_not_mutated():
    request = FeatureEngineeringRequest(
        dataset_id="ds", dataset_version_id="ds:raw", objective="  keep  "
    )
    snapshot = request.model_dump_json()
    understand_feature_engineering(request)
    assert request.model_dump_json() == snapshot


def test_no_dataframe_parameter():
    import inspect

    params = inspect.signature(understand_feature_engineering).parameters
    assert list(params) == ["request"]


def test_no_files_or_figures_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    understand_feature_engineering(FeatureEngineeringRequest(dataset_id="ds", objective="x"))
    assert list(tmp_path.iterdir()) == []


def test_rejects_dataframe_argument():
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError):
        understand_feature_engineering(pd.DataFrame({"a": [1, 2]}))


# --- backward compatibility ------------------------------


def test_phase_5_problem_understanding_untouched():
    from data_engine.problem_understanding import (
        ProblemUnderstandingRequest,
        understand_problem,
    )

    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.status.value == "not_yet_inferred"


def test_understand_problem_signature_unchanged():
    import inspect

    from data_engine.problem_understanding import understand_problem

    assert list(inspect.signature(understand_problem).parameters) == ["request"]


def test_existing_data_engine_apis_importable():
    import importlib

    for name in (
        "data_engine.eda",
        "data_engine.problem_understanding",
        "data_engine.profiling",
        "data_engine.quality",
        "data_engine.validation",
    ):
        assert importlib.import_module(name) is not None


def test_legacy_style_spec_json_validates():
    minimal = json.dumps(
        {
            "dataset_id": "ds",
            "objective_provided": False,
        }
    )
    model = FeatureEngineeringSpec.model_validate_json(minimal)
    assert model.feature_engineering_engine_version == "1"
    assert model.status is NOT_YET
    assert model.inventory.status is NOT_YET
