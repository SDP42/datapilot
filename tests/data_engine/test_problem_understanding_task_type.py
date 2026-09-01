"""Phase 5.3 — deterministic ML task-type inference."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data_engine.problem_understanding as pu
from data_engine.problem_understanding import (
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
    identify_target,
    infer_task_type,
    understand_problem,
)

_N = 240


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.normal(100.0, 20.0, _N),
            "churned": ([True, False] * (_N // 2)),
            "region": (["north", "south"] * (_N // 2)),
            "segment": (["A", "B", "C", "D", "E"] * (_N // 5)),
            "grade": ([1, 2, 3] * (_N // 3)),
            "signup_date": pd.date_range("2021-01-01", periods=_N, freq="D"),
        }
    )


def _target(
    column: str | None, *, status: ProblemUnderstandingStatus | None = None, **kw
) -> TargetIdentification:
    return TargetIdentification(
        status=status or ProblemUnderstandingStatus.COMPLETED, target_column=column, **kw
    )


def _task(result: TaskTypeInference) -> TaskType:
    assert result.status is ProblemUnderstandingStatus.COMPLETED
    assert result.task_type is not None
    return result.task_type


# --- API --------------------------------------------------------


def test_public_import():
    assert pu.infer_task_type is infer_task_type


def test_return_type(df):
    assert isinstance(infer_task_type(df, _target("price")), TaskTypeInference)


def test_uses_the_phase_5_1_tasktype_enum(df):
    result = infer_task_type(df, _target("price"))
    assert result.task_type in set(TaskType)


def test_json_round_trip(df):
    result = infer_task_type(df, _target("segment"), objective="classify the segment")
    assert TaskTypeInference.model_validate_json(result.model_dump_json()) == result


def test_json_primitive_only(df):
    payload = json.loads(infer_task_type(df, _target("price")).model_dump_json())

    def prim(o: object) -> bool:
        if isinstance(o, dict):
            return all(prim(v) for v in o.values())
        if isinstance(o, list):
            return all(prim(v) for v in o)
        return o is None or isinstance(o, (str, int, float, bool))

    assert prim(payload)


# --- structural inference --------------------------------------


def test_numeric_continuous_target_is_regression(df):
    assert _task(infer_task_type(df, _target("price"))) is TaskType.REGRESSION


def test_boolean_target_is_binary_classification(df):
    assert _task(infer_task_type(df, _target("churned"))) is TaskType.BINARY_CLASSIFICATION


def test_categorical_two_class_target_is_binary_classification(df):
    assert _task(infer_task_type(df, _target("region"))) is TaskType.BINARY_CLASSIFICATION


def test_categorical_multi_class_target_is_multiclass_classification(df):
    result = infer_task_type(df, _target("segment"))
    assert _task(result) is TaskType.MULTICLASS_CLASSIFICATION
    assert "5 distinct classes" in result.notes[0]


def test_numeric_low_cardinality_target_stays_regression_without_class_evidence(df):
    # `grade` is integer 1/2/3 — without a classification objective it is regression
    assert _task(infer_task_type(df, _target("grade"))) is TaskType.REGRESSION


def test_datetime_target_is_not_automatically_forecasting(df):
    result = infer_task_type(df, _target("signup_date"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert result.task_type is None
    assert "datetime target alone" in result.reason


# --- objective evidence --------------------------------------


def test_explicit_regression_objective(df):
    assert _task(infer_task_type(df, _target("price"), objective="predict house price")) is (
        TaskType.REGRESSION
    )


def test_explicit_classification_objective_on_categorical(df):
    assert (
        _task(infer_task_type(df, _target("region"), objective="classify the region"))
        is TaskType.BINARY_CLASSIFICATION
    )


def test_explicit_multiclass_objective(df):
    assert (
        _task(infer_task_type(df, _target("segment"), objective="assign one of several classes"))
        is TaskType.MULTICLASS_CLASSIFICATION
    )


def test_explicit_clustering_objective_without_target(df):
    result = infer_task_type(
        df, _target(None, candidate_columns=["price"]), objective="segment customers into groups"
    )
    assert _task(result) is TaskType.CLUSTERING
    assert result.objective_used is True


def test_explicit_forecasting_objective_with_datetime_column(df):
    result = infer_task_type(df, _target("price"), objective="forecast next month's price")
    assert _task(result) is TaskType.TIME_SERIES_FORECASTING


def test_objective_normalization(df):
    a = infer_task_type(df, _target("price"), objective="Forecast   NEXT-MONTH's  price")
    b = infer_task_type(df, _target("price"), objective="forecast next month's price")
    assert a.model_dump() == b.model_dump()


def test_objective_absent(df):
    result = infer_task_type(df, _target("price"))
    assert result.objective_used is False
    assert "no objective supplied" in result.notes
    assert _task(result) is TaskType.REGRESSION


def test_blank_objective_is_treated_as_absent(df):
    result = infer_task_type(df, _target("price"), objective="   ")
    assert result.objective_used is False


def test_objective_conflict_with_structural_evidence_keeps_structure(df):
    # continuous numeric target + "classify" -> regression (structurally supported) + a note
    result = infer_task_type(df, _target("price"), objective="classify the customers")
    assert _task(result) is TaskType.REGRESSION
    assert any("suggests classification" in n for n in result.notes)


def test_regression_objective_on_categorical_target_keeps_classification(df):
    result = infer_task_type(df, _target("region"), objective="estimate the revenue")
    assert _task(result) is TaskType.BINARY_CLASSIFICATION
    assert any("suggests regression" in n for n in result.notes)


# --- target handling ---------------------------------------


def test_no_target_and_ordinary_objective_is_unavailable(df):
    result = infer_task_type(df, _target(None), objective="predict the price")
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert result.task_type is None
    assert "no single target column" in result.reason


def test_target_identification_unavailable_is_propagated(df):
    upstream = TargetIdentification(
        status=ProblemUnderstandingStatus.UNAVAILABLE, reason="the DataFrame has no rows"
    )
    result = infer_task_type(df, upstream)
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "target identification was unavailable" in result.reason


def test_missing_target_column_is_unavailable(df):
    result = infer_task_type(df, _target("does_not_exist"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "not in the DataFrame" in result.reason


def test_all_missing_target_is_unavailable():
    frame = pd.DataFrame({"t": [np.nan] * 20, "x": range(20)})
    result = infer_task_type(frame, _target("t"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "entirely missing" in result.reason


def test_constant_target_is_unavailable(df):
    frame = df.assign(k=1.0)
    result = infer_task_type(frame, _target("k"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "constant" in result.reason


def test_target_object_is_not_mutated(df):
    target = identify_target(df, objective="predict house price")
    before = target.model_dump_json()
    infer_task_type(df, target, objective="predict house price")
    assert target.model_dump_json() == before


def test_ambiguous_target_from_identify_target_is_unavailable(df):
    ambiguous = identify_target(df)  # several similarly-plausible columns, no objective
    assert ambiguous.target_column is None
    result = infer_task_type(df, ambiguous)
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE


# --- clustering ------------------------------------------


def test_no_target_plus_clustering_objective_is_clustering(df):
    assert (
        _task(infer_task_type(df, _target(None), objective="cluster the users"))
        is TaskType.CLUSTERING
    )


def test_no_target_plus_no_objective_is_unavailable(df):
    result = infer_task_type(df, _target(None))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE


def test_no_target_is_not_clustering_by_default(df):
    result = infer_task_type(df, _target(None), objective="predict something")
    assert result.task_type is None


# --- forecasting -------------------------------------


def test_forecast_wording_without_any_datetime_column_stays_regression():
    frame = pd.DataFrame({"sales": np.linspace(1.0, 100.0, 60), "region": (["a", "b"] * 30)})
    result = infer_task_type(frame, _target("sales"), objective="forecast future sales")
    assert _task(result) is TaskType.REGRESSION
    assert any("no datetime column" in n for n in result.notes)


def test_datetime_column_without_forecasting_objective_is_not_forecasting(df):
    result = infer_task_type(df, _target("price"), objective="predict customer churn")
    assert _task(result) is not TaskType.TIME_SERIES_FORECASTING


def test_datetime_column_plus_forecasting_objective_plus_numeric_target(df):
    assert (
        _task(infer_task_type(df, _target("price"), objective="forecast next quarter's price"))
        is TaskType.TIME_SERIES_FORECASTING
    )


def test_datetime_target_with_forecasting_objective_is_forecasting(df):
    assert (
        _task(
            infer_task_type(df, _target("signup_date"), objective="forecast the future signup date")
        )
        is TaskType.TIME_SERIES_FORECASTING
    )


# --- determinism ------------------------------------


def test_repeated_calls_identical_json(df):
    dumps = {
        infer_task_type(df, _target("price"), objective="predict price").model_dump_json()
        for _ in range(5)
    }
    assert len(dumps) == 1


def test_row_shuffle_does_not_change_the_result(df):
    target = _target("segment")
    base = infer_task_type(df, target, objective="classify segment")
    shuffled = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    assert infer_task_type(shuffled, target, objective="classify segment").model_dump() == (
        base.model_dump()
    )


def test_column_reorder_does_not_change_the_result(df):
    target = _target("price")
    base = infer_task_type(df, target, objective="forecast next month's price")
    reordered = df[list(df.columns)[::-1]]
    assert infer_task_type(
        reordered, target, objective="forecast next month's price"
    ).model_dump() == (base.model_dump())


# --- safety -------------------------------------


def test_dataframe_is_not_mutated(df):
    before = df.copy(deep=True)
    infer_task_type(df, _target("price"), objective="predict price")
    infer_task_type(df, _target("segment"), objective="classify segment")
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    infer_task_type(df, _target("price"))
    assert list(tmp_path.iterdir()) == []


def test_no_figure_created(df):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    infer_task_type(df, _target("price"))
    assert plt.get_fignums() == before


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        infer_task_type([1, 2, 3], _target("x"))  # type: ignore[arg-type]


def test_non_target_model_raises_type_error(df):
    with pytest.raises(TypeError):
        infer_task_type(df, {"target_column": "price"})  # type: ignore[arg-type]


# --- backward compatibility ------------------------


def test_phase_5_1_and_5_2_still_work(df):
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.task_type.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.task_type.objective_used is False
    tgt = identify_target(df, objective="predict house price")
    assert tgt.target_column == "price"


def test_legacy_task_type_inference_json_still_validates():
    legacy = json.dumps(
        {"status": "not_yet_inferred", "reason": None, "task_type": None, "notes": []}
    )
    restored = TaskTypeInference.model_validate_json(legacy)
    assert restored.objective_used is False
    assert restored.task_type is None


def test_understand_problem_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_problem).parameters) == ["request"]


def test_merge_into_problem_spec_leaves_metrics_and_feasibility_untouched(df):
    request = ProblemUnderstandingRequest(dataset_id="sales", objective="predict house price")
    spec = understand_problem(request)
    target = identify_target(df, objective=request.objective)
    task = infer_task_type(df, target, objective=request.objective)
    merged = spec.model_copy(update={"target": target, "task_type": task})

    assert merged.task_type.status is ProblemUnderstandingStatus.COMPLETED
    assert merged.task_type.task_type is TaskType.REGRESSION
    assert merged.metrics.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert merged.feasibility.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    from data_engine.problem_understanding import ProblemSpec

    assert ProblemSpec.model_validate_json(merged.model_dump_json()) == merged


def test_existing_engine_imports_unaffected(df):
    from data_engine.eda import analyze_dataframe
    from data_engine.quality import analyze_dataframe as q  # noqa: F401
    from data_engine.validation import DatasetVersion  # noqa: F401

    report = analyze_dataframe(pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]}))
    assert report.univariate.numeric
    assert not hasattr(report, "task_type")
