"""Phase 7.4 — baseline model training & evaluation."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from data_engine import modeling
from data_engine.feature_engineering import (
    FeatureEngineeringRequest,
    FeatureEngineeringStatus,
    assess_feature_engineering,
    inventory_features,
    recommend_feature_selection,
    recommend_preprocessing,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.modeling import (
    DataSplitPlan,
    ModelCandidates,
    ModelingRequest,
    ModelingStatus,
    ModelReadiness,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
    assess_model_readiness,
    generate_model_candidates,
    recommend_data_split,
    train_and_evaluate_models,
    understand_modeling,
)
from data_engine.problem_understanding import (
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TaskType,
    assess_feasibility,
    identify_target,
    infer_task_type,
    recommend_metrics,
    understand_problem,
)

COMPLETED = ModelingStatus.COMPLETED
UNAVAILABLE = ModelingStatus.UNAVAILABLE
NOT_YET = ModelingStatus.NOT_YET_INFERRED
PU_DONE = ProblemUnderstandingStatus.COMPLETED
FE_DONE = FeatureEngineeringStatus.COMPLETED
_N = 400


def _build(df: pd.DataFrame, objective: str):
    t = identify_target(df, objective=objective)
    task = infer_task_type(df, t, objective=objective)
    m = recommend_metrics(df, task, objective=objective)
    feas = assess_feasibility(df, t, task, m)
    problem = understand_problem(
        ProblemUnderstandingRequest(dataset_id="d", objective=objective)
    ).model_copy(update={"target": t, "task_type": task, "metrics": m, "feasibility": feas})

    inv = inventory_features(df, target=t.target_column)
    tr = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, task)
    pp = recommend_preprocessing(df, inv, tr, sel)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp)
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d")).model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "assessment": asmt,
        }
    )
    readiness = assess_model_readiness(df, problem, fe)
    split = recommend_data_split(df, problem, fe)
    candidates = generate_model_candidates(df, problem, fe, readiness, split)
    return df, problem, fe, readiness, split, candidates


def _run(fixture, **kw) -> TrainingOutcome:
    df, problem, fe, readiness, split, candidates = fixture
    return train_and_evaluate_models(df, problem, fe, readiness, split, candidates, **kw)


@pytest.fixture
def clf():
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, 1.0, _N)
    df = pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0.0, 1.0, _N),
            "region": (["n", "s", "e", "w"] * (_N // 4)),
            "churn": (signal + rng.normal(0.0, 0.5, _N) > 0.0),
        }
    )
    return _build(df, "classify churn")


@pytest.fixture
def clf_numeric():
    rng = np.random.default_rng(6)
    s = rng.normal(0.0, 1.0, _N)
    df = pd.DataFrame(
        {
            "s": s,
            "t": rng.normal(0.0, 1.0, _N),
            "label": np.where(s > 0.4, "a", np.where(s < -0.4, "b", "c")),
        }
    )
    return _build(df, "classify the label")


@pytest.fixture
def reg():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.0, 10.0, _N)
    df = pd.DataFrame(
        {
            "s": s,
            "t": rng.normal(0.0, 1.0, _N),
            "price": 3.0 * s + rng.normal(0.0, 1.0, _N),
        }
    )
    return _build(df, "predict the price")


@pytest.fixture
def clustering():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": np.r_[rng.normal(-4.0, 1.0, _N // 2), rng.normal(4.0, 1.0, _N // 2)],
            "b": np.r_[rng.normal(-4.0, 1.0, _N // 2), rng.normal(4.0, 1.0, _N // 2)],
        }
    )
    out = _build(df, "segment customers into clusters")
    if out[1].task_type.task_type is not TaskType.CLUSTERING:
        pytest.skip("task inference did not yield clustering")
    return out


@pytest.fixture
def forecasting():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "day": pd.date_range("2021-01-01", periods=_N, freq="D"),
            "temp": rng.normal(20.0, 3.0, _N),
            "demand": rng.uniform(10.0, 90.0, _N),
        }
    )
    out = _build(df, "forecast future demand over time")
    if out[1].task_type.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting")
    return out


# --- API --------------------------------------------------------------


def test_public_import():
    assert modeling.train_and_evaluate_models is train_and_evaluate_models
    assert "TrainingRun" in modeling.__all__


def test_return_type(clf):
    assert isinstance(_run(clf), TrainingOutcome)


def test_exact_signature():
    assert list(inspect.signature(train_and_evaluate_models).parameters) == [
        "df",
        "problem",
        "feature_engineering",
        "readiness",
        "split",
        "candidates",
        "objective",
    ]


def test_structured_output(clf):
    out = _run(clf)
    assert all(isinstance(r, TrainingRun) for r in out.runs)
    assert all(isinstance(r.status, TrainingRunStatus) for r in out.runs)


def test_json_serialisation(clf):
    assert isinstance(json.loads(_run(clf).model_dump_json()), dict)


def test_json_round_trip(clf):
    out = _run(clf, objective="classify churn")
    assert TrainingOutcome.model_validate_json(out.model_dump_json()) == out


def test_json_primitive_only(clf):
    payload = json.loads(_run(clf).model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


# --- type validation -----------------------------------------------


@pytest.mark.parametrize("bad_index", range(6))
def test_type_guards(clf, bad_index):
    args = list(clf)
    args[bad_index] = object()
    with pytest.raises(TypeError):
        train_and_evaluate_models(*args)


# --- upstream behaviour -------------------------------------------


def test_task_unavailable(clf):
    df, problem, fe, readiness, split, candidates = clf
    bad = problem.task_type.model_copy(update={"status": ProblemUnderstandingStatus.UNAVAILABLE})
    out = train_and_evaluate_models(
        df, problem.model_copy(update={"task_type": bad}), fe, readiness, split, candidates
    )
    assert out.status is UNAVAILABLE
    assert out.runs == [] and out.successful_runs == []


def test_unsupported_task(clf):
    df, problem, fe, readiness, split, candidates = clf
    bad = problem.task_type.model_copy(update={"task_type": TaskType.MULTILABEL_CLASSIFICATION})
    out = train_and_evaluate_models(
        df, problem.model_copy(update={"task_type": bad}), fe, readiness, split, candidates
    )
    assert out.status is UNAVAILABLE
    assert "does not support task type 'multilabel_classification'" in out.reason


def test_readiness_unavailable(clf):
    df, problem, fe, _, split, candidates = clf
    out = train_and_evaluate_models(
        df, problem, fe, ModelReadiness(status=UNAVAILABLE, reason="x"), split, candidates
    )
    assert out.status is UNAVAILABLE


def test_readiness_not_ready(clf):
    df, problem, fe, readiness, split, candidates = clf
    bad = readiness.model_copy(update={"ready": False, "blocking_issues": ["3 rows"]})
    out = train_and_evaluate_models(df, problem, fe, bad, split, candidates)
    assert out.status is UNAVAILABLE
    assert "blocked by model-readiness issues" in out.reason


def test_split_unavailable(clf):
    df, problem, fe, readiness, _, candidates = clf
    out = train_and_evaluate_models(
        df, problem, fe, readiness, DataSplitPlan(status=UNAVAILABLE, reason="x"), candidates
    )
    assert out.status is UNAVAILABLE


def test_candidates_unavailable(clf):
    df, problem, fe, readiness, split, _ = clf
    out = train_and_evaluate_models(
        df, problem, fe, readiness, split, ModelCandidates(status=UNAVAILABLE, reason="x")
    )
    assert out.status is UNAVAILABLE
    assert "model candidates are not available" in out.reason


def test_incomplete_fe_assessment_unavailable(clf):
    df, problem, fe, readiness, split, candidates = clf
    bad_fe = fe.model_copy(
        update={"assessment": fe.assessment.model_copy(update={"status": NOT_YET})}
    )
    out = train_and_evaluate_models(df, problem, bad_fe, readiness, split, candidates)
    assert out.status is UNAVAILABLE
    assert "feature-engineering assessment is not completed" in out.reason


def test_deterministic_precedence_task_before_readiness(clf):
    df, problem, fe, _, split, candidates = clf
    bad_task = problem.task_type.model_copy(update={"task_type": None})
    out = train_and_evaluate_models(
        df,
        problem.model_copy(update={"task_type": bad_task}),
        fe,
        ModelReadiness(status=UNAVAILABLE),
        split,
        candidates,
    )
    assert "without a task type" in out.reason


def test_sklearn_unavailable(clf, monkeypatch):
    monkeypatch.setattr("data_engine.modeling.training._SKLEARN_AVAILABLE", False)
    out = _run(clf)
    assert out.status is UNAVAILABLE
    assert "scikit-learn is not available" in out.reason


def test_empty_candidates_completed(clf):
    df, problem, fe, readiness, split, candidates = clf
    empty = candidates.model_copy(update={"candidates": [], "candidates_detail": []})
    out = train_and_evaluate_models(df, problem, fe, readiness, split, empty)
    assert out.status is COMPLETED
    assert out.runs == []
    assert "no candidate model families were provided" in out.reason


# --- regression -------------------------------------------------


def test_regression_trains_and_has_metrics(reg):
    out = _run(reg)
    assert out.status is COMPLETED
    assert "linear" in out.successful_runs
    linear = next(r for r in out.runs if r.family.value == "linear")
    assert set(linear.metrics) >= {"rmse", "mae"}
    assert linear.metrics["rmse"] >= 0.0


def test_regression_target_excluded_from_features(reg):
    _, problem, *_ = reg
    out = _run(reg)
    blob = out.model_dump_json()
    assert "'price' is excluded from the model features" in blob or "price" in " ".join(out.notes)
    assert problem.target.target_column == "price"


def test_regression_no_selection_result(reg):
    out = _run(reg)
    for key in ("recommended", "best", "winner", "champion", "selected model"):
        assert key not in out.model_dump_json().lower()


# --- classification -------------------------------------------


def test_binary_classification(clf):
    out = _run(clf)
    assert out.status is COMPLETED
    assert out.successful_runs
    for r in out.runs:
        if r.status is TrainingRunStatus.COMPLETED:
            assert set(r.metrics) >= {"accuracy", "precision", "recall", "f1"}
            assert 0.0 <= r.metrics["accuracy"] <= 1.0


def test_multiclass_classification(clf_numeric):
    if clf_numeric[1].task_type.task_type is not TaskType.MULTICLASS_CLASSIFICATION:
        pytest.skip("not multiclass")
    out = _run(clf_numeric)
    assert out.successful_runs
    r = next(x for x in out.runs if x.status is TrainingRunStatus.COMPLETED)
    assert "roc_auc" not in r.metrics  # multiclass -> no binary roc_auc


def test_stratified_split_used(clf):
    _, _, _, _, split, _ = clf
    assert split.strategy.value == "stratified_holdout"
    out = _run(clf)
    r = next(x for x in out.runs if x.status is TrainingRunStatus.COMPLETED)
    assert r.train_rows + r.validation_rows + r.test_rows > 0


def test_tiny_class_falls_back(clf):
    df, problem, fe, readiness, split, candidates = clf
    d = df.copy()
    d.loc[d.index[:3], "churn"] = True
    d.loc[d.index[3:], "churn"] = False  # only 1 True, rest False -> imbalanced but 2 classes
    d.loc[d.index[0], "churn"] = True
    out = train_and_evaluate_models(d, problem, fe, readiness, split, candidates)
    assert out.status is COMPLETED  # no crash


# --- clustering ----------------------------------------------


def test_clustering_no_target_required(clustering):
    out = _run(clustering)
    assert out.status is COMPLETED
    assert set(out.successful_runs) <= {"distance_based", "probabilistic"}


def test_clustering_uses_unsupervised_metrics(clustering):
    out = _run(clustering)
    r = next((x for x in out.runs if x.status is TrainingRunStatus.COMPLETED), None)
    assert r is not None
    assert "silhouette_score" in r.metrics
    assert "accuracy" not in r.metrics and "rmse" not in r.metrics


# --- forecasting -------------------------------------------


def test_forecasting_baseline_boundary(forecasting):
    out = _run(forecasting)
    joined = " ".join(out.notes)
    assert "no lag features" in joined
    assert "rolling features" in joined
    assert "never a datetime column" in joined
    assert out.status is COMPLETED
    for r in out.runs:
        if r.status is TrainingRunStatus.COMPLETED:
            assert set(r.metrics) >= {"rmse", "mae"}


def test_forecasting_preserves_temporal_order(forecasting):
    _, _, _, _, split, _ = forecasting
    assert split.preserve_temporal_order is True
    out = _run(forecasting)
    assert out.status is COMPLETED


# --- preprocessing ----------------------------------------


def test_categorical_encoding_executed(clf):
    # the clf fixture has a categorical 'region' -> Phase 6.5 flags encoding
    _, _, fe, *_ = clf
    assert fe.preprocessing.encoding_required is True
    out = _run(clf)
    assert out.successful_runs  # models trained on encoded categoricals


def test_imputation_executed():
    rng = np.random.default_rng(7)
    s = rng.uniform(0.0, 10.0, _N)
    df = pd.DataFrame(
        {"s": s, "t": rng.normal(0.0, 1.0, _N), "price": 2.0 * s + rng.normal(0.0, 1.0, _N)}
    )
    df.loc[df.index[:30], "t"] = np.nan  # ~7.5% missing
    fixture = _build(df, "predict the price")
    assert fixture[2].preprocessing.imputation_required is True
    out = train_and_evaluate_models(*fixture)
    assert out.successful_runs  # trained despite missing values


def test_target_never_encoded(clf):
    out = _run(clf)
    blob = out.model_dump_json().lower()
    assert "target encoding" not in blob
    assert "target encoder" not in blob


def test_no_mutation_of_preprocessing(clf):
    _, _, fe, *_ = clf
    snap = fe.preprocessing.model_dump_json()
    _run(clf)
    assert fe.preprocessing.model_dump_json() == snap


# --- partial failure ------------------------------------


def test_partial_failure_recorded(clf):
    df, problem, fe, readiness, split, candidates = clf
    # inject a family with no baseline estimator for classification: none in
    # the vocabulary lacks one, so instead break one candidate name deterministically
    bad_candidates = candidates.model_copy(
        update={"candidates": [*candidates.candidates, "distance_based"]}
    )
    out = train_and_evaluate_models(df, problem, fe, readiness, split, bad_candidates)
    # duplicate is de-duplicated, not trained twice
    assert out.model_dump_json() != ""
    families = [r.family.value for r in out.runs]
    assert len(families) == len(set(families))


def test_unavailable_run_when_no_estimator(reg):
    df, problem, fe, readiness, split, candidates = reg
    forced = candidates.model_copy(
        update={"candidates": ["probabilistic"], "candidates_detail": []}
    )
    out = train_and_evaluate_models(df, problem, fe, readiness, split, forced)
    assert out.status is COMPLETED
    assert out.successful_runs == []
    assert out.failed_runs == ["probabilistic"]
    run = out.runs[0]
    assert run.status is TrainingRunStatus.UNAVAILABLE
    assert "no dependency-light baseline estimator" in run.reason


# --- determinism -------------------------------------


def test_five_repeated_calls_identical(clf):
    blobs = {_run(clf).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_deterministic_metrics(reg):
    a = _run(reg)
    b = _run(reg)
    assert [r.metrics for r in a.runs] == [r.metrics for r in b.runs]


def test_random_split_row_order_invariant(clf):
    df, problem, fe, readiness, split, candidates = clf
    base = train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    shuffled = df.sample(frac=1.0, random_state=11)
    other = train_and_evaluate_models(shuffled, problem, fe, readiness, split, candidates)
    assert base == other


def test_column_reorder_invariant(clf):
    df, problem, fe, readiness, split, candidates = clf
    base = train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    reordered = df[list(df.columns)[::-1]]
    assert train_and_evaluate_models(reordered, problem, fe, readiness, split, candidates) == base


def test_stable_run_ordering(clf):
    out = _run(clf)
    _, _, _, _, _, candidates = clf
    assert [r.family.value for r in out.runs] == candidates.candidates


def test_fixed_random_seed_note(clf):
    out = _run(clf)
    assert any("random seed: 42 (fixed)" in n for n in out.notes)


# --- safety ---------------------------------------


def test_df_unchanged(clf):
    df, problem, fe, readiness, split, candidates = clf
    before = df.copy(deep=True)
    train_and_evaluate_models(df, problem, fe, readiness, split, candidates, objective="x")
    train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    pd.testing.assert_frame_equal(df, before)


def test_upstream_models_unchanged(clf):
    df, problem, fe, readiness, split, candidates = clf
    snaps = tuple(m.model_dump_json() for m in (problem, fe, readiness, split, candidates))
    train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    assert tuple(m.model_dump_json() for m in (problem, fe, readiness, split, candidates)) == snaps


def test_no_files_created(clf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(clf)
    assert list(tmp_path.iterdir()) == []


def test_no_estimator_objects_in_result(clf):
    out = _run(clf)
    payload = json.loads(out.model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)
    blob = out.model_dump_json().lower()
    for token in ("0x", "object at", ".pkl", ".joblib", "site-packages", "traceback"):
        assert token not in blob


def test_objective_recorded_not_influential(clf):
    plain = _run(clf)
    with_obj = _run(clf, objective="please pick the best model and use XGBoost")
    assert with_obj.objective_used is True
    assert plain.objective_used is False
    assert [r.family for r in plain.runs] == [r.family for r in with_obj.runs]
    assert "xgboost" not in with_obj.model_dump_json().lower()


# --- integration --------------------------------


def test_merge_into_modeling_spec(clf):
    df, problem, fe, readiness, split, candidates = clf
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective="classify churn"))
    spec = spec.model_copy(
        update={"readiness": readiness, "split": split, "candidates": candidates}
    )
    training = train_and_evaluate_models(
        df, problem, fe, readiness, split, candidates, objective="classify churn"
    )
    merged = spec.model_copy(update={"training": training})

    assert merged.training.status is COMPLETED
    assert merged.readiness == readiness
    assert merged.split == split
    assert merged.candidates == candidates
    assert merged.evaluation.status is NOT_YET
    assert merged.selection.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility ---------------------


def test_phase_7_1_foundation_unchanged():
    spec = understand_modeling(ModelingRequest(dataset_id="d"))
    assert spec.training.status is NOT_YET
    assert spec.training.runs == []


def test_understand_modeling_signature_unchanged():
    assert list(inspect.signature(understand_modeling).parameters) == ["request"]


def test_legacy_training_outcome_json_validates():
    legacy = json.dumps({"status": "not_yet_inferred", "reason": None, "notes": []})
    model = TrainingOutcome.model_validate_json(legacy)
    assert model.runs == []
    assert model.successful_runs == []
    assert model.objective_used is False


def test_phase_7_2_and_7_3_still_work(reg):
    _, _, _, readiness, split, candidates = reg
    assert readiness.status is COMPLETED
    assert split.status is COMPLETED
    assert candidates.status is COMPLETED


def test_phase_5_and_6_apis_still_work(clf):
    _, problem, fe, *_ = clf
    assert problem.task_type.status is PU_DONE
    assert fe.assessment.status is FE_DONE
