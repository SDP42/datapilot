"""Phase 5.4 — deterministic candidate-metrics recommendation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data_engine.problem_understanding as pu
from data_engine.problem_understanding import (
    CandidateMetrics,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
    identify_target,
    infer_task_type,
    recommend_metrics,
    understand_problem,
)

_N = 200


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.uniform(10.0, 500.0, _N),  # positive, non-zero
            "delta": rng.normal(0.0, 5.0, _N),  # contains negatives
            "with_zero": np.concatenate([[0.0], rng.uniform(1.0, 9.0, _N - 1)]),
            "churned": ([True, False] * (_N // 2)),
            "region": (["north", "south"] * (_N // 2)),
            "segment": (["A", "B", "C", "D"] * (_N // 4)),
            "signup_date": pd.date_range("2021-01-01", periods=_N, freq="D"),
        }
    )


def _tt(task: TaskType | None, target: str | None = None, *, status=None) -> TaskTypeInference:
    return TaskTypeInference(
        status=status or ProblemUnderstandingStatus.COMPLETED,
        task_type=task,
        target_column=target,
    )


def _ok(result: CandidateMetrics) -> CandidateMetrics:
    assert result.status is ProblemUnderstandingStatus.COMPLETED
    assert result.primary_metric is not None
    assert result.primary_metric in result.metrics
    return result


# --- API -------------------------------------------------------


def test_public_import():
    assert pu.recommend_metrics is recommend_metrics


def test_return_type(df):
    assert isinstance(recommend_metrics(df, _tt(TaskType.REGRESSION, "price")), CandidateMetrics)


def test_json_round_trip(df):
    result = recommend_metrics(df, _tt(TaskType.BINARY_CLASSIFICATION, "churned"))
    assert CandidateMetrics.model_validate_json(result.model_dump_json()) == result


def test_json_primitive_only(df):
    payload = json.loads(recommend_metrics(df, _tt(TaskType.REGRESSION, "price")).model_dump_json())

    def prim(o: object) -> bool:
        if isinstance(o, dict):
            return all(prim(v) for v in o.values())
        if isinstance(o, list):
            return all(prim(v) for v in o)
        return o is None or isinstance(o, (str, int, float, bool))

    assert prim(payload)


# --- regression -------------------------------------------


def test_default_regression_metrics(df):
    result = _ok(recommend_metrics(df, _tt(TaskType.REGRESSION, "price")))
    assert result.metrics[:3] == ["rmse", "mae", "r2"]


def test_rmse_is_regression_primary_by_default(df):
    assert recommend_metrics(df, _tt(TaskType.REGRESSION, "price")).primary_metric == "rmse"


def test_mae_objective_prioritises_mae(df):
    result = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="minimize absolute error"
    )
    assert result.primary_metric == "mae"
    assert result.metrics[0] == "mae"


def test_r2_objective_prioritises_r2(df):
    result = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="maximise explained variance"
    )
    assert result.primary_metric == "r2"


def test_mape_included_for_positive_nonzero_target(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    assert "mape" in result.metrics
    assert result.metrics[-1] == "mape"  # appended after the defaults


def test_mape_excluded_when_target_contains_zero(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, "with_zero"))
    assert "mape" not in result.metrics
    assert any("zero" in n for n in result.notes)


def test_mape_excluded_when_target_contains_negative(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, "delta"))
    assert "mape" not in result.metrics
    assert any("negative" in n for n in result.notes)


def test_mape_excluded_when_target_column_unknown(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, None))
    assert "mape" not in result.metrics
    assert any("unknown" in n for n in result.notes)


def test_percentage_error_objective_but_incompatible_target(df):
    result = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "delta"), objective="minimise the percentage error"
    )
    assert "mape" not in result.metrics
    assert result.primary_metric != "mape"
    assert any("mape" in n and "ignored" in n for n in result.notes)


# --- binary classification ----------------------------


def test_default_binary_metrics(df):
    result = _ok(recommend_metrics(df, _tt(TaskType.BINARY_CLASSIFICATION, "churned")))
    assert result.metrics == ["f1", "roc_auc", "precision", "recall", "accuracy"]


def test_f1_is_binary_primary_by_default(df):
    assert (
        recommend_metrics(df, _tt(TaskType.BINARY_CLASSIFICATION, "churned")).primary_metric == "f1"
    )


def test_precision_objective(df):
    result = recommend_metrics(
        df, _tt(TaskType.BINARY_CLASSIFICATION, "churned"), objective="avoid false positives"
    )
    assert result.primary_metric == "precision"


def test_recall_objective(df):
    result = recommend_metrics(
        df,
        _tt(TaskType.BINARY_CLASSIFICATION, "churned"),
        objective="we must avoid false negatives",
    )
    assert result.primary_metric == "recall"


def test_balance_objective_keeps_f1(df):
    result = recommend_metrics(
        df,
        _tt(TaskType.BINARY_CLASSIFICATION, "churned"),
        objective="balance precision and recall",
    )
    assert result.primary_metric == "f1"


def test_imbalanced_objective_prioritises_f1_over_accuracy(df):
    result = recommend_metrics(
        df, _tt(TaskType.BINARY_CLASSIFICATION, "churned"), objective="the positive class is rare"
    )
    assert result.primary_metric == "f1"
    assert any("imbalance" in n for n in result.notes)


# --- multiclass -----------------------------------


def test_default_multiclass_metrics(df):
    result = _ok(recommend_metrics(df, _tt(TaskType.MULTICLASS_CLASSIFICATION, "segment")))
    assert result.metrics == ["f1_macro", "accuracy", "precision_macro", "recall_macro"]


def test_macro_f1_is_multiclass_primary(df):
    assert (
        recommend_metrics(df, _tt(TaskType.MULTICLASS_CLASSIFICATION, "segment")).primary_metric
        == "f1_macro"
    )


def test_multiclass_has_no_binary_only_metric(df):
    result = recommend_metrics(df, _tt(TaskType.MULTICLASS_CLASSIFICATION, "segment"))
    assert "roc_auc" not in result.metrics
    assert "f1" not in result.metrics  # only f1_macro


def test_multiclass_imbalance_prioritises_macro_f1(df):
    result = recommend_metrics(
        df,
        _tt(TaskType.MULTICLASS_CLASSIFICATION, "segment"),
        objective="the classes are imbalanced",
    )
    assert result.primary_metric == "f1_macro"


# --- clustering ---------------------------------


def test_clustering_metrics_returned(df):
    result = _ok(recommend_metrics(df, _tt(TaskType.CLUSTERING, None)))
    assert result.metrics == [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ]


def test_silhouette_is_clustering_primary(df):
    assert recommend_metrics(df, _tt(TaskType.CLUSTERING, None)).primary_metric == (
        "silhouette_score"
    )


def test_clustering_needs_no_target(df):
    # a targetless clustering task is still fully handled
    result = recommend_metrics(df, _tt(TaskType.CLUSTERING, None), objective="segment customers")
    assert result.status is ProblemUnderstandingStatus.COMPLETED


# --- forecasting -------------------------------


def test_forecasting_metrics(df):
    result = _ok(recommend_metrics(df, _tt(TaskType.TIME_SERIES_FORECASTING, "price")))
    assert result.metrics[:2] == ["mae", "rmse"]
    assert result.primary_metric == "mae"


def test_forecasting_mape_compatibility(df):
    positive = recommend_metrics(df, _tt(TaskType.TIME_SERIES_FORECASTING, "price"))
    assert "mape" in positive.metrics
    negative = recommend_metrics(df, _tt(TaskType.TIME_SERIES_FORECASTING, "delta"))
    assert "mape" not in negative.metrics


def test_forecasting_task_from_the_real_pipeline(df):
    target = identify_target(df, objective="forecast next month's price")
    task = infer_task_type(df, target, objective="forecast next month's price")
    assert task.task_type is TaskType.TIME_SERIES_FORECASTING
    result = recommend_metrics(df, task, objective="forecast next month's price")
    assert result.status is ProblemUnderstandingStatus.COMPLETED
    assert result.metrics[:2] == ["mae", "rmse"]


# --- unavailable / unsupported ---------------


def test_task_inference_unavailable_is_unavailable(df):
    upstream = TaskTypeInference(
        status=ProblemUnderstandingStatus.UNAVAILABLE, reason="no single target column"
    )
    result = recommend_metrics(df, upstream)
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert result.metrics == []
    assert result.primary_metric is None
    assert "not completed" in result.reason


def test_missing_task_type_is_unavailable(df):
    result = recommend_metrics(df, _tt(None))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE


def test_multilabel_classification_is_unsupported(df):
    result = recommend_metrics(df, _tt(TaskType.MULTILABEL_CLASSIFICATION, "segment"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "multilabel_classification" in result.reason


def test_other_task_is_unsupported(df):
    result = recommend_metrics(df, _tt(TaskType.OTHER, "price"))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "other" in result.reason


# --- objective behaviour ---------------------


def test_objective_normalisation(df):
    a = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="Minimize   ABSOLUTE-error"
    )
    b = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="minimize absolute error"
    )
    assert a.model_dump() == b.model_dump()


def test_objective_absent(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    assert result.objective_used is False
    assert any("no objective supplied" in n for n in result.notes)


def test_blank_objective_is_treated_as_absent(df):
    result = recommend_metrics(df, _tt(TaskType.REGRESSION, "price"), objective="   ")
    assert result.objective_used is False


def test_conflicting_objective_preferences_resolved_deterministically(df):
    result = recommend_metrics(
        df,
        _tt(TaskType.REGRESSION, "price"),
        objective="minimize both the absolute error and the squared error",
    )
    assert result.primary_metric == "mae"  # alphabetically first of {mae, rmse}
    assert any("alphabetically first" in n for n in result.notes)


def test_ranking_wording_does_not_fabricate_a_ranking_metric(df):
    result = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="rank the customers by predicted price"
    )
    assert result.metrics[:3] == ["rmse", "mae", "r2"]
    assert not any("rank" in m for m in result.metrics)
    assert any("ranking-specific evaluation is not yet supported" in n for n in result.notes)


def test_incompatible_metric_word_for_task_is_ignored(df):
    result = recommend_metrics(
        df, _tt(TaskType.REGRESSION, "price"), objective="maximise the roc auc"
    )
    assert result.primary_metric == "rmse"  # roc_auc is not a regression metric
    assert any("roc_auc" in n and "ignored" in n for n in result.notes)


# --- determinism ---------------------------


def test_repeated_calls_identical_json(df):
    dumps = {
        recommend_metrics(
            df, _tt(TaskType.REGRESSION, "price"), objective="minimize absolute error"
        ).model_dump_json()
        for _ in range(5)
    }
    assert len(dumps) == 1


def test_row_shuffle_does_not_change_the_result(df):
    base = recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    shuffled = df.sample(frac=1.0, random_state=11).reset_index(drop=True)
    assert recommend_metrics(shuffled, _tt(TaskType.REGRESSION, "price")).model_dump() == (
        base.model_dump()
    )


def test_column_reorder_does_not_change_the_result(df):
    base = recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    reordered = df[list(df.columns)[::-1]]
    assert recommend_metrics(reordered, _tt(TaskType.REGRESSION, "price")).model_dump() == (
        base.model_dump()
    )


# --- safety -------------------------------


def test_dataframe_is_not_mutated(df):
    before = df.copy(deep=True)
    recommend_metrics(df, _tt(TaskType.REGRESSION, "price"), objective="minimize absolute error")
    recommend_metrics(df, _tt(TaskType.BINARY_CLASSIFICATION, "churned"))
    pd.testing.assert_frame_equal(df, before)


def test_task_result_is_not_mutated(df):
    task = infer_task_type(df, identify_target(df, objective="predict house price"))
    before = task.model_dump_json()
    recommend_metrics(df, task, objective="predict house price")
    assert task.model_dump_json() == before


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    assert list(tmp_path.iterdir()) == []


def test_no_figure_created(df):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    recommend_metrics(df, _tt(TaskType.REGRESSION, "price"))
    assert plt.get_fignums() == before


def test_invalid_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        recommend_metrics([1, 2, 3], _tt(TaskType.REGRESSION))  # type: ignore[arg-type]


def test_invalid_task_result_raises_type_error(df):
    with pytest.raises(TypeError):
        recommend_metrics(df, {"task_type": "regression"})  # type: ignore[arg-type]


# --- integration --------------------------


def test_explicit_problem_spec_merge(df):
    request = ProblemUnderstandingRequest(dataset_id="sales", objective="predict house price")
    spec = understand_problem(request)
    target = identify_target(df, objective=request.objective)
    task = infer_task_type(df, target, objective=request.objective)
    metrics = recommend_metrics(df, task, objective=request.objective)
    merged = spec.model_copy(update={"target": target, "task_type": task, "metrics": metrics})

    assert merged.target == target  # unchanged
    assert merged.task_type == task  # unchanged
    assert merged.metrics.status is ProblemUnderstandingStatus.COMPLETED
    assert merged.metrics.primary_metric == "rmse"
    assert merged.feasibility.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert merged.status is ProblemUnderstandingStatus.NOT_YET_INFERRED  # overall unchanged

    from data_engine.problem_understanding import ProblemSpec

    assert ProblemSpec.model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility --------------


def test_legacy_candidate_metrics_json_still_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "primary_metric": None,
            "metrics": [],
            "notes": [],
        }
    )
    restored = CandidateMetrics.model_validate_json(legacy)
    assert restored.objective_used is False
    assert restored.metrics == []


def test_legacy_task_type_inference_json_without_target_column_validates():
    legacy = json.dumps(
        {"status": "completed", "reason": None, "task_type": "regression", "notes": []}
    )
    restored = TaskTypeInference.model_validate_json(legacy)
    assert restored.target_column is None


def test_phase_5_1_to_5_3_still_work(df):
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.metrics.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.metrics.objective_used is False
    target = identify_target(df, objective="predict house price")
    assert target.target_column == "price"
    task = infer_task_type(df, target, objective="predict house price")
    assert task.task_type is TaskType.REGRESSION
    assert task.target_column == "price"  # 5.3 now echoes the target column


def test_understand_problem_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_problem).parameters) == ["request"]


def test_existing_engine_imports_unaffected(df):
    from data_engine.eda import analyze_dataframe
    from data_engine.quality import analyze_dataframe as q  # noqa: F401
    from data_engine.validation import DatasetVersion  # noqa: F401

    report = analyze_dataframe(pd.DataFrame({"a": [1.0, 2, 3], "b": [3.0, 2, 1]}))
    assert report.univariate.numeric
    assert not hasattr(report, "metrics")
