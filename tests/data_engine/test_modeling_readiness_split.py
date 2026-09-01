"""Phase 7.2 — model readiness & data-split planning."""

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
    FeatureInventory,
    assess_feature_engineering,
    inventory_features,
    recommend_feature_selection,
    recommend_preprocessing,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.modeling import (
    DataSplitPlan,
    DataSplitStrategy,
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    ModelReadiness,
    assess_model_readiness,
    recommend_data_split,
    understand_modeling,
)
from data_engine.problem_understanding import (
    ProblemSpec,
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

_N = 300


def _problem(df: pd.DataFrame, target_hint: str, objective: str) -> ProblemSpec:
    t = identify_target(df, objective=objective)
    task = infer_task_type(df, t, objective=objective)
    m = recommend_metrics(df, task, objective=objective)
    feas = assess_feasibility(df, t, task, m)
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="d", objective=objective))
    return spec.model_copy(
        update={"target": t, "task_type": task, "metrics": m, "feasibility": feas}
    )


def _fe(df: pd.DataFrame, problem: ProblemSpec):
    target = problem.target.target_column
    inv = inventory_features(df, target=target)
    tr = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, problem.task_type)
    pp = recommend_preprocessing(df, inv, tr, sel)
    asmt = assess_feature_engineering(df, inv, tr, sel, pp)
    spec = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d"))
    return spec.model_copy(
        update={
            "inventory": inv,
            "transformations": tr,
            "selection": sel,
            "preprocessing": pp,
            "assessment": asmt,
        }
    )


@pytest.fixture
def clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "amount": rng.uniform(1.0, 100.0, _N),
            "score": rng.normal(50.0, 10.0, _N),
            "region": (["north", "south", "east", "west"] * (_N // 4)),
            "churn": ([True, False] * (_N // 2)),
        }
    )


@pytest.fixture
def reg_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "sqft": rng.uniform(500.0, 4000.0, _N),
            "rooms": rng.integers(1, 8, _N),
            "price": rng.uniform(1e5, 9e5, _N),
        }
    )


@pytest.fixture
def ts_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "day": pd.date_range("2021-01-01", periods=_N, freq="D"),
            "demand": rng.uniform(10.0, 90.0, _N),
        }
    )


@pytest.fixture
def clf(clf_df):
    problem = _problem(clf_df, "churn", "classify churn")
    return clf_df, problem, _fe(clf_df, problem)


@pytest.fixture
def reg(reg_df):
    problem = _problem(reg_df, "price", "predict the price")
    return reg_df, problem, _fe(reg_df, problem)


@pytest.fixture
def ts(ts_df):
    problem = _problem(ts_df, "demand", "forecast future demand over time")
    return ts_df, problem, _fe(ts_df, problem)


# --- API --------------------------------------------------------------


def test_public_imports():
    assert modeling.assess_model_readiness is assess_model_readiness
    assert modeling.recommend_data_split is recommend_data_split


def test_return_types(clf):
    df, problem, fe = clf
    assert isinstance(assess_model_readiness(df, problem, fe), ModelReadiness)
    assert isinstance(recommend_data_split(df, problem, fe), DataSplitPlan)


def test_exact_signatures():
    assert list(inspect.signature(assess_model_readiness).parameters) == [
        "df",
        "problem",
        "feature_engineering",
        "objective",
    ]
    assert list(inspect.signature(recommend_data_split).parameters) == [
        "df",
        "problem",
        "feature_engineering",
        "objective",
    ]


def test_json_round_trip(clf):
    df, problem, fe = clf
    r = assess_model_readiness(df, problem, fe, objective="classify churn")
    s = recommend_data_split(df, problem, fe, objective="classify churn")
    assert ModelReadiness.model_validate_json(r.model_dump_json()) == r
    assert DataSplitPlan.model_validate_json(s.model_dump_json()) == s


def test_json_primitive_only(clf):
    df, problem, fe = clf
    for model in (assess_model_readiness(df, problem, fe), recommend_data_split(df, problem, fe)):
        payload = json.loads(model.model_dump_json())

        def check(v):
            if isinstance(v, dict):
                [check(x) for x in v.values()]
            elif isinstance(v, list):
                [check(x) for x in v]
            else:
                assert v is None or isinstance(v, (str, int, float, bool))

        check(payload)


def test_structured_output(clf):
    df, problem, fe = clf
    r = assess_model_readiness(df, problem, fe)
    assert isinstance(r.ready, bool)
    assert isinstance(r.blocking_issues, list)
    s = recommend_data_split(df, problem, fe)
    assert isinstance(s.strategy, DataSplitStrategy)


# --- type validation -----------------------------------------------


@pytest.mark.parametrize("fn", [assess_model_readiness, recommend_data_split])
def test_non_dataframe_type_error(fn, clf):
    _, problem, fe = clf
    with pytest.raises(TypeError):
        fn({"a": [1]}, problem, fe)


@pytest.mark.parametrize("fn", [assess_model_readiness, recommend_data_split])
def test_non_problem_type_error(fn, clf):
    df, _, fe = clf
    with pytest.raises(TypeError):
        fn(df, {"status": "completed"}, fe)


@pytest.mark.parametrize("fn", [assess_model_readiness, recommend_data_split])
def test_non_fe_type_error(fn, clf):
    df, problem, _ = clf
    with pytest.raises(TypeError):
        fn(df, problem, {"status": "completed"})


# --- readiness ----------------------------------------------------


def test_valid_supervised_pipeline_ready(clf):
    df, problem, fe = clf
    r = assess_model_readiness(df, problem, fe)
    assert r.status is COMPLETED
    assert r.ready is True
    assert r.reason is None
    assert r.blocking_issues == []
    assert r.target_available is True
    assert r.target_usable is True
    assert r.eligible_feature_count >= 1
    assert r.feature_engineering_assessment_usable is True
    assert r.sufficient_observations is True


def test_regression_pipeline_ready(reg):
    df, problem, fe = reg
    r = assess_model_readiness(df, problem, fe)
    assert r.ready is True


def test_missing_target_column_blocks(clf):
    df, problem, fe = clf
    bad_target = problem.target.model_copy(update={"target_column": "not_a_column"})
    bad_problem = problem.model_copy(update={"target": bad_target})
    r = assess_model_readiness(df, bad_problem, fe)
    assert r.status is COMPLETED
    assert r.ready is False
    assert any("not in the DataFrame" in b for b in r.blocking_issues)


def test_no_target_for_supervised_blocks(clf):
    df, problem, fe = clf
    bad_target = problem.target.model_copy(update={"target_column": None})
    r = assess_model_readiness(df, problem.model_copy(update={"target": bad_target}), fe)
    assert r.ready is False
    assert any("no target column" in b for b in r.blocking_issues)


def test_constant_target_blocks(clf_df):
    d = clf_df.copy()
    d["churn"] = True
    problem = _problem(clf_df, "churn", "classify churn")
    fe = _fe(clf_df, problem)
    r = assess_model_readiness(d, problem, fe)
    assert r.ready is False
    assert any("constant" in b for b in r.blocking_issues)


def test_no_eligible_features_blocks(clf):
    df, problem, fe = clf
    empty_inv = FeatureInventory(status=FE_DONE, candidate_features=[], candidates=[])
    empty_sel = fe.selection.model_copy(
        update={"selected_features": [], "review_features": [], "dropped_features": []}
    )
    bad_fe = fe.model_copy(update={"inventory": empty_inv, "selection": empty_sel})
    r = assess_model_readiness(df, problem, bad_fe)
    assert r.ready is False
    assert any("no structurally eligible feature" in b for b in r.blocking_issues)


def test_incomplete_task_inference_unavailable(clf):
    df, problem, fe = clf
    bad_task = problem.task_type.model_copy(
        update={"status": ProblemUnderstandingStatus.NOT_YET_INFERRED}
    )
    r = assess_model_readiness(df, problem.model_copy(update={"task_type": bad_task}), fe)
    assert r.status is UNAVAILABLE
    assert r.ready is None
    assert "task-type inference is not completed" in r.reason


def test_incomplete_inventory_unavailable(clf):
    df, problem, fe = clf
    bad_fe = fe.model_copy(update={"inventory": FeatureInventory()})
    r = assess_model_readiness(df, problem, bad_fe)
    assert r.status is UNAVAILABLE
    assert "feature inventory is not completed" in r.reason


def test_incomplete_fe_assessment_unavailable(clf):
    df, problem, fe = clf
    bad_fe = fe.model_copy(
        update={"assessment": fe.assessment.model_copy(update={"status": NOT_YET})}
    )
    r = assess_model_readiness(df, problem, bad_fe)
    assert r.status is UNAVAILABLE
    assert "feature-engineering assessment is not completed" in r.reason


def test_failed_fe_assessment_blocks(clf):
    df, problem, fe = clf
    bad = fe.assessment.model_copy(
        update={"feasible": False, "blocking_issues": ["[x] a fabricated inconsistency"]}
    )
    r = assess_model_readiness(df, problem, fe.model_copy(update={"assessment": bad}))
    assert r.ready is False
    assert any("structurally infeasible" in b for b in r.blocking_issues)


def test_phase5_infeasible_blocks(clf):
    df, problem, fe = clf
    bad_feas = problem.feasibility.model_copy(
        update={"status": PU_DONE, "feasible": False, "reason": "too few rows"}
    )
    r = assess_model_readiness(df, problem.model_copy(update={"feasibility": bad_feas}), fe)
    assert r.ready is False
    assert any("Phase 5 feasibility" in b for b in r.blocking_issues)


def test_warnings_without_blocking(clf_df):
    d = clf_df.iloc[:60].copy()  # small -> row warning, still >= MODEL_READINESS_MIN_ROWS
    problem = _problem(d, "churn", "classify churn")
    fe = _fe(d, problem)
    r = assess_model_readiness(d, problem, fe)
    assert r.ready is True
    assert r.warnings
    assert any("fewer than" in w for w in r.warnings)


def test_too_few_rows_blocks(clf_df):
    d = clf_df.iloc[:5].copy()
    problem = _problem(clf_df, "churn", "classify churn")
    fe = _fe(clf_df, problem)
    r = assess_model_readiness(d, problem, fe)
    assert r.ready is False
    assert r.sufficient_observations is False
    assert any("at least" in b for b in r.blocking_issues)


def test_no_predictive_claims(clf):
    df, problem, fe = clf
    r = assess_model_readiness(df, problem, fe)
    blob = r.model_dump_json().lower()
    assert "accuracy" not in blob
    assert "will perform" not in blob
    assert "good model" in blob  # the disclaimer text
    assert any("structural check only" in n for n in r.notes)


def test_clustering_needs_no_target():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"a": rng.normal(0, 1, _N), "b": rng.normal(0, 1, _N)})
    problem = _problem(d, "", "segment customers into clusters")
    if problem.task_type.task_type is not TaskType.CLUSTERING:
        pytest.skip("task inference did not yield clustering")
    fe = _fe(d, problem)
    r = assess_model_readiness(d, problem, fe)
    assert r.status is COMPLETED
    assert r.target_available is False


# --- split planning --------------------------------------------


def test_split_classification_stratified(clf):
    df, problem, fe = clf
    s = recommend_data_split(df, problem, fe)
    assert s.status is COMPLETED
    assert s.strategy is DataSplitStrategy.STRATIFIED_HOLDOUT
    assert s.stratify is True
    assert s.preserve_temporal_order is False
    assert s.shuffle is True


def test_split_regression_not_stratified(reg):
    df, problem, fe = reg
    s = recommend_data_split(df, problem, fe)
    assert s.strategy is DataSplitStrategy.RANDOM_HOLDOUT
    assert s.stratify is False
    assert s.shuffle is True
    assert any("continuous target" in n for n in s.notes)


def test_split_forecasting_time_ordered(ts):
    df, problem, fe = ts
    if problem.task_type.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting")
    s = recommend_data_split(df, problem, fe)
    assert s.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    assert s.preserve_temporal_order is True
    assert s.shuffle is False
    assert s.stratify is False
    assert any("chronological" in n for n in s.notes)
    assert any("does not infer a forecasting task" in n for n in s.notes)


def test_split_unsupported_task_unavailable(clf):
    df, problem, fe = clf
    bad_task = problem.task_type.model_copy(
        update={"task_type": TaskType.MULTILABEL_CLASSIFICATION}
    )
    s = recommend_data_split(df, problem.model_copy(update={"task_type": bad_task}), fe)
    assert s.status is UNAVAILABLE
    assert "does not support task type" in s.reason


def test_split_missing_upstream_unavailable(clf):
    df, problem, fe = clf
    bad_task = problem.task_type.model_copy(update={"status": NOT_YET})
    s = recommend_data_split(df, problem.model_copy(update={"task_type": bad_task}), fe)
    assert s.status is UNAVAILABLE


def test_split_deterministic_fractions_large(clf):
    df, problem, fe = clf
    s = recommend_data_split(df, problem, fe)
    assert (s.train_fraction, s.validation_fraction, s.test_fraction) == (0.7, 0.15, 0.15)


def test_split_small_data_no_validation(clf_df):
    d = clf_df.iloc[:120].copy()
    problem = _problem(clf_df, "churn", "classify churn")
    fe = _fe(clf_df, problem)
    s = recommend_data_split(d, problem, fe)
    assert (s.train_fraction, s.validation_fraction, s.test_fraction) == (0.8, None, 0.2)
    assert any("train/test split only" in n for n in s.notes)


def test_split_stratify_off_for_tiny_class(clf_df):
    d = clf_df.copy()
    labels = ["A"] * (_N - 1) + ["B"]
    d["churn"] = labels
    problem = _problem(d, "churn", "classify churn")
    fe = _fe(d, problem)
    s = recommend_data_split(d, problem, fe)
    if problem.task_type.task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
        assert s.stratify is False
        assert s.strategy is DataSplitStrategy.RANDOM_HOLDOUT


def test_split_deterministic_reasons(clf):
    df, problem, fe = clf
    a = recommend_data_split(df, problem, fe)
    b = recommend_data_split(df.sample(frac=1.0, random_state=9), problem, fe)
    assert a.notes == b.notes


# --- determinism -------------------------------------------


def test_repeated_calls_identical(clf):
    df, problem, fe = clf
    r = {assess_model_readiness(df, problem, fe).model_dump_json() for _ in range(5)}
    s = {recommend_data_split(df, problem, fe).model_dump_json() for _ in range(5)}
    assert len(r) == 1 and len(s) == 1


def test_row_shuffle_identical(clf):
    df, problem, fe = clf
    shuffled = df.sample(frac=1.0, random_state=7)
    assert assess_model_readiness(shuffled, problem, fe) == assess_model_readiness(df, problem, fe)
    assert recommend_data_split(shuffled, problem, fe) == recommend_data_split(df, problem, fe)


def test_column_reorder_identical(clf):
    df, problem, fe = clf
    reordered = df[list(df.columns)[::-1]]
    assert assess_model_readiness(reordered, problem, fe) == assess_model_readiness(df, problem, fe)
    assert recommend_data_split(reordered, problem, fe) == recommend_data_split(df, problem, fe)


def test_stable_blocking_ordering(clf):
    df, problem, fe = clf
    bad_target = problem.target.model_copy(update={"target_column": "not_a_column"})
    p = problem.model_copy(update={"target": bad_target})
    a = assess_model_readiness(df.iloc[:5], p, fe)
    b = assess_model_readiness(df.iloc[:5], p, fe)
    assert a.blocking_issues == b.blocking_issues == sorted(a.blocking_issues)


# --- safety ----------------------------------------------


def test_df_unchanged(clf):
    df, problem, fe = clf
    before = df.copy(deep=True)
    assess_model_readiness(df, problem, fe, objective="classify churn")
    recommend_data_split(df, problem, fe, objective="classify churn")
    pd.testing.assert_frame_equal(df, before)


def test_upstream_models_unchanged(clf):
    df, problem, fe = clf
    snaps = (problem.model_dump_json(), fe.model_dump_json())
    assess_model_readiness(df, problem, fe)
    recommend_data_split(df, problem, fe)
    assert (problem.model_dump_json(), fe.model_dump_json()) == snaps


def test_no_files_created(clf, tmp_path, monkeypatch):
    df, problem, fe = clf
    monkeypatch.chdir(tmp_path)
    assess_model_readiness(df, problem, fe)
    recommend_data_split(df, problem, fe)
    assert list(tmp_path.iterdir()) == []


def test_no_actual_split_performed(clf):
    df, problem, fe = clf
    s = recommend_data_split(df, problem, fe)
    blob = s.model_dump_json()
    # only fractions/flags — never row indices or datasets
    assert "index" not in blob.lower()
    assert isinstance(s.train_fraction, float)


# --- integration --------------------------------------


def test_merge_into_modeling_spec(clf):
    df, problem, fe = clf
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective="classify churn"))
    readiness = assess_model_readiness(df, problem, fe, objective="classify churn")
    split = recommend_data_split(df, problem, fe, objective="classify churn")
    merged = spec.model_copy(update={"readiness": readiness, "split": split})

    assert merged.readiness.status is COMPLETED
    assert merged.split.status is COMPLETED
    assert merged.candidates.status is NOT_YET
    assert merged.training.status is NOT_YET
    assert merged.evaluation.status is NOT_YET
    assert merged.selection.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility -------------------------


def test_phase_7_1_foundation_unchanged():
    spec = understand_modeling(ModelingRequest(dataset_id="d"))
    assert spec.status is NOT_YET
    assert spec.readiness.status is NOT_YET
    assert spec.split.status is NOT_YET


def test_understand_modeling_signature_unchanged():
    assert list(inspect.signature(understand_modeling).parameters) == ["request"]


def test_legacy_readiness_json_validates():
    legacy = json.dumps({"status": "not_yet_inferred", "reason": None, "notes": []})
    model = ModelReadiness.model_validate_json(legacy)
    assert model.ready is None
    assert model.blocking_issues == []
    assert model.eligible_feature_count == 0


def test_legacy_split_json_validates():
    legacy = json.dumps({"status": "not_yet_inferred", "reason": None, "notes": []})
    model = DataSplitPlan.model_validate_json(legacy)
    assert model.strategy is None
    assert model.train_fraction is None
    assert model.stratify is False


def test_phase_5_and_6_apis_still_work(clf_df):
    problem = _problem(clf_df, "churn", "classify churn")
    assert problem.task_type.status is PU_DONE
    fe = _fe(clf_df, problem)
    assert fe.assessment.status is FE_DONE


def test_modeling_spec_still_constructs():
    assert ModelingSpec(dataset_id="d", objective_provided=False).status is NOT_YET
