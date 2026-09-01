"""Phase 5.1 — the ProblemSpec contract + Problem Understanding foundation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import data_engine.problem_understanding as pu
from data_engine.problem_understanding import (
    PROBLEM_UNDERSTANDING_ENGINE_VERSION,
    CandidateMetrics,
    FeasibilityAssessment,
    ProblemSpec,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
    understand_problem,
)


def _is_json_primitive(obj: object) -> bool:
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_primitive(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_is_json_primitive(v) for v in obj)
    return obj is None or isinstance(obj, (str, int, float, bool))


# --- API / imports ------------------------------------------------


def test_public_symbols_importable_from_package_surface():
    assert pu.understand_problem is understand_problem
    for name in (
        "ProblemSpec",
        "ProblemUnderstandingRequest",
        "ProblemUnderstandingStatus",
        "TargetIdentification",
        "TaskType",
        "TaskTypeInference",
        "CandidateMetrics",
        "FeasibilityAssessment",
        "PROBLEM_UNDERSTANDING_ENGINE_VERSION",
    ):
        assert hasattr(pu, name)


def test_understand_problem_returns_a_problem_spec():
    result = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert isinstance(result, ProblemSpec)


def test_status_enum_has_three_explicit_states():
    assert {s.value for s in ProblemUnderstandingStatus} == {
        "not_yet_inferred",
        "completed",
        "unavailable",
    }


def test_task_type_enum_is_defined_but_not_used_yet():
    # the contract is stable; Phase 5.1 must not populate it
    assert TaskType.REGRESSION.value == "regression"
    assert TaskType.BINARY_CLASSIFICATION.value == "binary_classification"


# --- ProblemSpec model -----------------------------------------


def test_valid_construction_and_defaults():
    spec = ProblemSpec(dataset_id="ds", objective_provided=False)
    assert spec.problem_understanding_engine_version == PROBLEM_UNDERSTANDING_ENGINE_VERSION == "1"
    assert spec.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert isinstance(spec.target, TargetIdentification)
    assert isinstance(spec.task_type, TaskTypeInference)
    assert isinstance(spec.metrics, CandidateMetrics)
    assert isinstance(spec.feasibility, FeasibilityAssessment)
    assert spec.dataset_version_id is None
    assert spec.objective is None
    assert spec.notes == []


def test_every_section_is_not_yet_inferred_and_nothing_is_fabricated():
    spec = understand_problem(
        ProblemUnderstandingRequest(dataset_id="ds", objective="predict revenue")
    )
    assert spec.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.reason is not None  # explains why
    # target
    assert spec.target.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.target.target_column is None
    assert spec.target.candidate_columns == []
    # task type — never a fake "classification"/"regression"
    assert spec.task_type.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.task_type.task_type is None
    # metrics — never fabricated
    assert spec.metrics.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.metrics.primary_metric is None
    assert spec.metrics.metrics == []
    # feasibility — never a fake False
    assert spec.feasibility.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.feasibility.feasible is None
    assert spec.feasibility.blocking_issues == []


def test_dataset_identity_and_objective_are_echoed():
    spec = understand_problem(
        ProblemUnderstandingRequest(
            dataset_id="sales", dataset_version_id="sales:raw", objective="predict churn"
        )
    )
    assert spec.dataset_id == "sales"
    assert spec.dataset_version_id == "sales:raw"
    assert spec.objective == "predict churn"
    assert spec.objective_provided is True


@pytest.mark.parametrize(
    ("objective", "expected"),
    [(None, False), ("", False), ("   ", False), ("predict x", True)],
)
def test_objective_provided_flag(objective, expected):
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds", objective=objective))
    assert spec.objective_provided is expected
    assert spec.objective == objective  # verbatim, even when blank


def test_objective_is_not_inferred_from_anything():
    # no data is passed in; the objective is purely the request field
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.objective is None
    assert spec.objective_provided is False


# --- serialisation -------------------------------------------


def test_problem_spec_json_round_trip():
    spec = understand_problem(
        ProblemUnderstandingRequest(dataset_id="ds", objective="cluster users")
    )
    dumped = spec.model_dump_json()
    assert ProblemSpec.model_validate_json(dumped) == spec


def test_request_json_round_trip():
    request = ProblemUnderstandingRequest(
        dataset_id="ds", dataset_version_id="ds:exec-1", objective="x"
    )
    assert ProblemUnderstandingRequest.model_validate_json(request.model_dump_json()) == request


def test_model_contains_only_json_primitives():
    payload = json.loads(
        understand_problem(ProblemUnderstandingRequest(dataset_id="ds")).model_dump_json()
    )
    assert _is_json_primitive(payload)


def test_hand_built_spec_with_later_increment_shape_round_trips():
    # prove the contract can carry a *completed* result without a redesign
    spec = ProblemSpec(
        dataset_id="ds",
        objective="predict price",
        objective_provided=True,
        status=ProblemUnderstandingStatus.COMPLETED,
        target=TargetIdentification(
            status=ProblemUnderstandingStatus.COMPLETED, target_column="price"
        ),
        task_type=TaskTypeInference(
            status=ProblemUnderstandingStatus.COMPLETED, task_type=TaskType.REGRESSION
        ),
        metrics=CandidateMetrics(
            status=ProblemUnderstandingStatus.COMPLETED,
            primary_metric="rmse",
            metrics=["rmse", "mae"],
        ),
        feasibility=FeasibilityAssessment(
            status=ProblemUnderstandingStatus.COMPLETED, feasible=True
        ),
    )
    assert ProblemSpec.model_validate_json(spec.model_dump_json()) == spec


# --- determinism --------------------------------------------


def test_repeated_calls_produce_identical_serialised_output():
    request = ProblemUnderstandingRequest(dataset_id="ds", objective="predict y")
    dumps = {understand_problem(request).model_dump_json() for _ in range(5)}
    assert len(dumps) == 1


def test_no_timestamp_or_uuid_style_field():
    payload = json.loads(
        understand_problem(ProblemUnderstandingRequest(dataset_id="ds")).model_dump_json()
    )
    assert "generated_at" not in payload
    assert "id" not in payload
    assert not any("uuid" in key.lower() or "timestamp" in key.lower() for key in payload)


# --- validation / errors -----------------------------------


def test_non_request_argument_raises_type_error():
    with pytest.raises(TypeError):
        understand_problem({"dataset_id": "ds"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        understand_problem("ds")  # type: ignore[arg-type]


def test_blank_dataset_id_raises_value_error():
    with pytest.raises(ValueError, match="dataset_id"):
        understand_problem(ProblemUnderstandingRequest(dataset_id=""))
    with pytest.raises(ValueError, match="dataset_id"):
        understand_problem(ProblemUnderstandingRequest(dataset_id="   "))


# --- safety -----------------------------------------------


def test_does_not_mutate_the_request():
    request = ProblemUnderstandingRequest(dataset_id="ds", objective="predict y")
    before = request.model_dump_json()
    understand_problem(request)
    assert request.model_dump_json() == before


def test_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    understand_problem(ProblemUnderstandingRequest(dataset_id="ds", objective="x"))
    assert list(tmp_path.iterdir()) == []


def test_creates_no_figure(monkeypatch):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert plt.get_fignums() == before


def test_takes_no_dataframe_and_reads_no_data():
    # the foundation deliberately accepts only identity + objective — it
    # cannot mutate a DataFrame because it never receives one.
    import inspect

    params = list(inspect.signature(understand_problem).parameters)
    assert params == ["request"]
    df = pd.DataFrame({"a": [1, 2, 3]})
    before = df.copy(deep=True)
    understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    pd.testing.assert_frame_equal(df, before)


# --- backward compatibility ------------------------------


def test_existing_eda_and_engine_imports_unaffected():
    from data_engine.eda import analyze_dataframe
    from data_engine.quality import analyze_dataframe as q_analyze  # noqa: F401
    from data_engine.validation import DatasetVersion  # noqa: F401

    report = analyze_dataframe(pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [2.0, 4, 6, 8]}))
    assert report.univariate.numeric
    assert not hasattr(report, "problem_spec")
