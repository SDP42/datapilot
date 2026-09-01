"""Phase 5.5 — deterministic feasibility assessment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data_engine.problem_understanding as pu
from data_engine.problem_understanding import (
    CandidateMetrics,
    FeasibilityAssessment,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
    assess_feasibility,
    identify_target,
    infer_task_type,
    recommend_metrics,
    understand_problem,
)

_N = 200
COMPLETED = ProblemUnderstandingStatus.COMPLETED
UNAVAILABLE = ProblemUnderstandingStatus.UNAVAILABLE
NOT_YET = ProblemUnderstandingStatus.NOT_YET_INFERRED


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.uniform(10.0, 500.0, _N),
            "size": rng.integers(1, 6, _N),
            "churned": ([True, False] * (_N // 2)),
            "segment": (["A", "B", "C", "D"] * (_N // 4)),
            "signup_date": pd.date_range("2021-01-01", periods=_N, freq="D"),
        }
    )


def _target(col: str | None, *, status=COMPLETED, reason=None) -> TargetIdentification:
    return TargetIdentification(status=status, target_column=col, reason=reason)


def _task(t: TaskType | None, col: str | None = None, *, status=COMPLETED, reason=None):
    return TaskTypeInference(status=status, task_type=t, target_column=col, reason=reason)


def _metrics(*, status=COMPLETED, primary="rmse", names=("rmse", "mae"), reason=None):
    return CandidateMetrics(
        status=status,
        primary_metric=primary if status is COMPLETED else None,
        metrics=list(names) if status is COMPLETED else [],
        reason=reason,
    )


def _assess(df, task, col, *, objective=None, metrics=None):
    return assess_feasibility(
        df,
        _target(col),
        _task(task, col),
        metrics or _metrics(),
        objective=objective,
    )


# --- API -------------------------------------------------------------------


def test_public_import():
    assert pu.assess_feasibility is assess_feasibility


def test_return_type(df):
    assert isinstance(_assess(df, TaskType.REGRESSION, "price"), FeasibilityAssessment)


def test_json_round_trip(df):
    result = _assess(df, TaskType.BINARY_CLASSIFICATION, "churned")
    assert FeasibilityAssessment.model_validate_json(result.model_dump_json()) == result


def test_json_primitive_only(df):
    payload = json.loads(_assess(df, TaskType.REGRESSION, "price").model_dump_json())

    def primitive(v):
        return v is None or isinstance(v, (str, int, float, bool, list))

    assert all(primitive(v) or isinstance(v, list) for v in payload.values())
    for key in ("blocking_issues", "warnings", "notes"):
        assert all(isinstance(x, str) for x in payload[key])


def test_wrong_df_type():
    with pytest.raises(TypeError):
        assess_feasibility([1, 2], _target("a"), _task(TaskType.REGRESSION, "a"), _metrics())


def test_wrong_target_type(df):
    with pytest.raises(TypeError):
        assess_feasibility(df, "price", _task(TaskType.REGRESSION, "price"), _metrics())


def test_wrong_task_type_type(df):
    with pytest.raises(TypeError):
        assess_feasibility(df, _target("price"), "regression", _metrics())


def test_wrong_metrics_type(df):
    with pytest.raises(TypeError):
        assess_feasibility(df, _target("price"), _task(TaskType.REGRESSION, "price"), {})


# --- upstream handling ---------------------------------------------------


def test_unavailable_target(df):
    result = assess_feasibility(
        df,
        _target(None, status=UNAVAILABLE, reason="no plausible target"),
        _task(TaskType.REGRESSION, None),
        _metrics(),
    )
    assert result.status is UNAVAILABLE
    assert result.feasible is None
    assert "target identification is unavailable" in result.reason


def test_unavailable_task_type(df):
    result = assess_feasibility(
        df,
        _target("price"),
        _task(None, "price", status=UNAVAILABLE, reason="contradictory evidence"),
        _metrics(),
    )
    assert result.status is UNAVAILABLE
    assert result.feasible is None
    assert "task-type inference is unavailable" in result.reason


def test_unavailable_metrics(df):
    result = assess_feasibility(
        df,
        _target("price"),
        _task(TaskType.REGRESSION, "price"),
        _metrics(status=UNAVAILABLE, reason="no vocabulary"),
    )
    assert result.status is UNAVAILABLE
    assert result.feasible is None
    assert "candidate-metric recommendation is unavailable" in result.reason


def test_not_yet_inferred_upstream_is_unavailable(df):
    result = assess_feasibility(
        df,
        TargetIdentification(),
        TaskTypeInference(),
        CandidateMetrics(),
    )
    assert result.status is UNAVAILABLE
    assert result.feasible is None


def test_ambiguous_target_supervised_is_unavailable(df):
    result = assess_feasibility(
        df,
        _target(None, reason="ambiguous: two equally likely targets"),
        _task(TaskType.REGRESSION, None),
        _metrics(),
    )
    assert result.status is UNAVAILABLE
    assert result.feasible is None
    assert "no single target column" in result.reason


def test_task_type_none_completed_is_unavailable(df):
    result = assess_feasibility(df, _target("price"), _task(None, "price"), _metrics())
    assert result.status is UNAVAILABLE


def test_clustering_without_target_is_allowed(df):
    result = assess_feasibility(
        df,
        _target(None),
        _task(TaskType.CLUSTERING, None),
        _metrics(primary="silhouette_score", names=("silhouette_score",)),
    )
    assert result.status is COMPLETED
    assert result.feasible is True


# --- dataset size ------------------------------------------------------


def test_zero_rows_blocks(df):
    result = _assess(df.iloc[:0], TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("row(s); at least 2" in b for b in result.blocking_issues)


def test_one_row_blocks(df):
    result = _assess(df.iloc[:1], TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("1 row(s)" in b for b in result.blocking_issues)


def test_small_dataset_warns_but_feasible(df):
    result = _assess(df.iloc[:10], TaskType.REGRESSION, "price")
    assert result.feasible is True
    assert any("fewer than 20" in w for w in result.warnings)


def test_sufficient_rows_no_size_warning(df):
    result = _assess(df, TaskType.REGRESSION, "price")
    assert result.feasible is True
    assert not any("fewer than 20" in w for w in result.warnings)


def test_structural_screen_note_always_present(df):
    result = _assess(df, TaskType.REGRESSION, "price")
    assert any("structural feasibility screen" in n for n in result.notes)


# --- supervised target ----------------------------------------------


def test_valid_target_feasible(df):
    assert _assess(df, TaskType.REGRESSION, "price").feasible is True


def test_missing_target_column_blocks(df):
    result = _assess(df, TaskType.REGRESSION, "not_a_column")
    assert result.feasible is False
    assert any("is not in the DataFrame" in b for b in result.blocking_issues)


def test_all_missing_target_blocks(df):
    d = df.copy()
    d["price"] = np.nan
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("no usable" in b for b in result.blocking_issues)


def test_constant_target_blocks(df):
    d = df.copy()
    d["price"] = 7.0
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("constant" in b for b in result.blocking_issues)


def test_substantial_target_missingness_warns(df):
    d = df.copy()
    d.loc[d.index[: int(_N * 0.5)], "price"] = np.nan
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is True
    assert any("missing" in w for w in result.warnings)


def test_minor_target_missingness_no_warning(df):
    d = df.copy()
    d.loc[d.index[:5], "price"] = np.nan
    result = _assess(d, TaskType.REGRESSION, "price")
    assert not any("missing" in w for w in result.warnings)


# --- classification --------------------------------------------------


def test_binary_classification_feasible(df):
    assert _assess(df, TaskType.BINARY_CLASSIFICATION, "churned").feasible is True


def test_multiclass_classification_feasible(df):
    assert _assess(df, TaskType.MULTICLASS_CLASSIFICATION, "segment").feasible is True


def test_single_class_target_blocks(df):
    d = df.copy()
    d["churned"] = True
    result = _assess(d, TaskType.BINARY_CLASSIFICATION, "churned")
    assert result.feasible is False
    assert any("class(es); at least 2" in b for b in result.blocking_issues)


def test_severe_imbalance_warns_but_feasible(df):
    d = df.copy()
    labels = ["A"] * (_N - 3) + ["B"] * 3
    d["segment"] = labels
    result = _assess(d, TaskType.BINARY_CLASSIFICATION, "segment")
    assert result.feasible is True
    assert any("severe class imbalance" in w for w in result.warnings)


def test_balanced_target_no_imbalance_warning(df):
    result = _assess(df, TaskType.BINARY_CLASSIFICATION, "churned")
    assert not any("imbalance" in w for w in result.warnings)


# --- regression ------------------------------------------------------


def test_regression_valid(df):
    assert _assess(df, TaskType.REGRESSION, "price").feasible is True


def test_regression_too_few_finite_blocks(df):
    d = df.iloc[:5].copy()
    d["price"] = [1.0, np.nan, np.inf, -np.inf, np.nan]
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("finite numeric observation" in b for b in result.blocking_issues)


def test_regression_constant_numeric_blocks(df):
    d = df.copy()
    d["price"] = 3.0
    assert _assess(d, TaskType.REGRESSION, "price").feasible is False


def test_regression_infinite_values_noted(df):
    d = df.copy()
    d.loc[d.index[:3], "price"] = np.inf
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is True
    assert any("non-finite" in n for n in result.notes)


# --- forecasting ----------------------------------------------------


def test_forecasting_valid_datetime_feature(df):
    result = _assess(df, TaskType.TIME_SERIES_FORECASTING, "price")
    assert result.feasible is True


def test_forecasting_no_datetime_blocks(df):
    d = df.drop(columns=["signup_date"])
    result = _assess(d, TaskType.TIME_SERIES_FORECASTING, "price")
    assert result.feasible is False
    assert any("requires at least one datetime column" in b for b in result.blocking_issues)


def test_forecasting_insufficient_timestamps_blocks(df):
    d = df.iloc[:1].copy()
    result = assess_feasibility(
        d,
        _target("price"),
        _task(TaskType.TIME_SERIES_FORECASTING, "price"),
        _metrics(primary="mae", names=("mae", "rmse")),
    )
    assert result.feasible is False
    assert any("timestamp" in b for b in result.blocking_issues)


def test_forecasting_identical_timestamps_blocks(df):
    d = df.iloc[:30].copy()
    d["signup_date"] = pd.Timestamp("2022-01-01")
    result = _assess(d, TaskType.TIME_SERIES_FORECASTING, "price")
    assert result.feasible is False
    assert any("single distinct timestamp" in b for b in result.blocking_issues)


def test_forecasting_datetime_target(df):
    result = _assess(df, TaskType.TIME_SERIES_FORECASTING, "signup_date")
    assert result.feasible is True
    assert any("signup_date" in n for n in result.notes)


# --- clustering ---------------------------------------------------


def test_clustering_valid(df):
    result = assess_feasibility(
        df,
        _target(None),
        _task(TaskType.CLUSTERING, None),
        _metrics(primary="silhouette_score", names=("silhouette_score",)),
    )
    assert result.feasible is True
    assert any("usable variation" in n for n in result.notes)


def test_clustering_no_usable_features_blocks():
    d = pd.DataFrame({"a": [1, 1, 1, 1], "b": [np.nan] * 4})
    result = assess_feasibility(
        d,
        _target(None),
        _task(TaskType.CLUSTERING, None),
        _metrics(primary="silhouette_score", names=("silhouette_score",)),
    )
    assert result.feasible is False
    assert any("usable feature variation" in b for b in result.blocking_issues)


def test_clustering_insufficient_rows_blocks():
    d = pd.DataFrame({"a": [1], "b": [2]})
    result = assess_feasibility(
        d,
        _target(None),
        _task(TaskType.CLUSTERING, None),
        _metrics(primary="silhouette_score", names=("silhouette_score",)),
    )
    assert result.feasible is False


def test_clustering_needs_no_target(df):
    result = assess_feasibility(
        df,
        _target(None),
        _task(TaskType.CLUSTERING, None),
        _metrics(primary="silhouette_score", names=("silhouette_score",)),
    )
    assert result.status is COMPLETED


# --- feature availability -------------------------------------------


def test_target_only_dataframe_blocks():
    d = pd.DataFrame({"price": np.arange(50.0)})
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("only the target column" in b for b in result.blocking_issues)


def test_usable_non_target_feature_ok(df):
    assert _assess(df, TaskType.REGRESSION, "price").feasible is True


def test_all_non_target_columns_missing_blocks():
    d = pd.DataFrame({"price": np.arange(50.0), "a": [np.nan] * 50, "b": [np.nan] * 50})
    result = _assess(d, TaskType.REGRESSION, "price")
    assert result.feasible is False
    assert any("every non-target column is entirely missing" in b for b in result.blocking_issues)


# --- leakage boundary ---------------------------------------------


def test_leakage_note_present_not_assessed(df):
    result = _assess(df, TaskType.REGRESSION, "price")
    assert any("leakage has not been assessed" in n for n in result.notes)
    assert not any("leakage" in b for b in result.blocking_issues)


# --- objective handling -----------------------------------------


def test_objective_recorded_but_not_overriding(df):
    result = _assess(df, TaskType.REGRESSION, "price", objective="predict the price")
    assert any("objective supplied" in n for n in result.notes)
    assert result.feasible is True


def test_blank_objective_treated_as_none(df):
    result = _assess(df, TaskType.REGRESSION, "price", objective="   ")
    assert any("no objective supplied" in n for n in result.notes)


# --- determinism ----------------------------------------------


def test_repeated_calls_identical_json(df):
    a = _assess(df, TaskType.BINARY_CLASSIFICATION, "churned").model_dump_json()
    b = _assess(df, TaskType.BINARY_CLASSIFICATION, "churned").model_dump_json()
    assert a == b


def test_row_shuffle_identical(df):
    base = _assess(df, TaskType.REGRESSION, "price")
    shuffled = df.sample(frac=1.0, random_state=7)
    assert _assess(shuffled, TaskType.REGRESSION, "price") == base


def test_column_reorder_identical(df):
    base = _assess(df, TaskType.REGRESSION, "price")
    reordered = df[list(df.columns)[::-1]]
    assert _assess(reordered, TaskType.REGRESSION, "price") == base


# --- safety --------------------------------------------------


def test_dataframe_not_mutated(df):
    before = df.copy(deep=True)
    _assess(df, TaskType.REGRESSION, "price")
    pd.testing.assert_frame_equal(df, before)


def test_target_not_mutated(df):
    t = _target("price")
    snapshot = t.model_dump_json()
    assess_feasibility(df, t, _task(TaskType.REGRESSION, "price"), _metrics())
    assert t.model_dump_json() == snapshot


def test_task_type_not_mutated(df):
    tt = _task(TaskType.REGRESSION, "price")
    snapshot = tt.model_dump_json()
    assess_feasibility(df, _target("price"), tt, _metrics())
    assert tt.model_dump_json() == snapshot


def test_metrics_not_mutated(df):
    m = _metrics()
    snapshot = m.model_dump_json()
    assess_feasibility(df, _target("price"), _task(TaskType.REGRESSION, "price"), m)
    assert m.model_dump_json() == snapshot


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _assess(df, TaskType.REGRESSION, "price")
    assert list(tmp_path.iterdir()) == []


# --- integration --------------------------------------------


def test_merge_into_problem_spec(df):
    spec = understand_problem(
        ProblemUnderstandingRequest(dataset_id="ds", objective="predict the price")
    )
    target = identify_target(df, objective="predict the price")
    task = infer_task_type(df, target, objective="predict the price")
    metrics = recommend_metrics(df, task, objective="predict the price")
    feasibility = assess_feasibility(df, target, task, metrics, objective="predict the price")

    merged = spec.model_copy(update={"feasibility": feasibility})

    assert merged.target == spec.target
    assert merged.task_type == spec.task_type
    assert merged.metrics == spec.metrics
    assert merged.feasibility.status is COMPLETED
    assert merged.feasibility.feasible is True
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


def test_full_pipeline_classification(df):
    target = identify_target(df, objective="classify churned customers")
    task = infer_task_type(df, target, objective="classify churned customers")
    metrics = recommend_metrics(df, task, objective="classify churned customers")
    result = assess_feasibility(df, target, task, metrics)
    assert result.status is COMPLETED
    assert result.feasible is True


# --- backward compatibility ---------------------------------


def test_legacy_feasibility_json_still_validates():
    legacy = '{"status": "not_yet_inferred", "reason": null, "feasible": null}'
    model = FeasibilityAssessment.model_validate_json(legacy)
    assert model.blocking_issues == []
    assert model.warnings == []
    assert model.notes == []


def test_phase_5_1_to_5_4_unchanged(df):
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.status is NOT_YET
    assert spec.feasibility.status is NOT_YET

    target = identify_target(df, objective="predict the price")
    assert target.target_column == "price"
    task = infer_task_type(df, target, objective="predict the price")
    assert task.task_type is TaskType.REGRESSION
    metrics = recommend_metrics(df, task, objective="predict the price")
    assert metrics.primary_metric == "rmse"
