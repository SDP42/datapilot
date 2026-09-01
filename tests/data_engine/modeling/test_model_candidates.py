"""Phase 7.3 — model candidate generation."""

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
    ModelCandidate,
    ModelCandidates,
    ModelFamily,
    ModelingRequest,
    ModelingStatus,
    ModelReadiness,
    assess_model_readiness,
    generate_model_candidates,
    recommend_data_split,
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
    return df, problem, fe, readiness, split


@pytest.fixture
def clf_mixed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, _N),
            "b": rng.normal(5.0, 2.0, _N),
            "cat": (["x", "y", "z", "w"] * (_N // 4)),
            "churn": ([True, False] * (_N // 2)),
        }
    )
    return _build(df, "classify churn")


@pytest.fixture
def clf_numeric():
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, _N),
            "b": rng.normal(5.0, 2.0, _N),
            "c": rng.uniform(0.0, 1.0, _N),
            "label": (["p", "q", "r"] * (_N // 3) + ["p"]),
        }
    )
    return _build(df, "classify the label")


@pytest.fixture
def reg_numeric():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, _N),
            "b": rng.normal(5.0, 2.0, _N),
            "price": rng.uniform(1.0, 9.0, _N),
        }
    )
    return _build(df, "predict the price")


@pytest.fixture
def reg_mixed():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "a": rng.normal(0.0, 1.0, _N),
            "region": (["n", "s", "e", "w"] * (_N // 4)),
            "price": rng.uniform(1.0, 9.0, _N),
        }
    )
    return _build(df, "predict the price")


@pytest.fixture
def clustering():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(0.0, 1.0, _N), "b": rng.normal(0.0, 1.0, _N)})
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
            "demand": rng.uniform(10.0, 90.0, _N),
        }
    )
    out = _build(df, "forecast future demand over time")
    if out[1].task_type.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting")
    return out


def _gen(fixture, **kw) -> ModelCandidates:
    df, problem, fe, readiness, split = fixture
    return generate_model_candidates(df, problem, fe, readiness, split, **kw)


# --- API -----------------------------------------------------------


def test_public_import():
    assert modeling.generate_model_candidates is generate_model_candidates
    assert "ModelCandidate" in modeling.__all__


def test_return_type(clf_mixed):
    assert isinstance(_gen(clf_mixed), ModelCandidates)


def test_exact_signature():
    assert list(inspect.signature(generate_model_candidates).parameters) == [
        "df",
        "problem",
        "feature_engineering",
        "readiness",
        "split",
        "objective",
    ]


def test_structured_model(clf_mixed):
    c = _gen(clf_mixed)
    assert all(isinstance(x, ModelCandidate) for x in c.candidates_detail)
    assert all(isinstance(x.family, ModelFamily) for x in c.candidates_detail)


def test_json_serialisation(clf_mixed):
    assert isinstance(json.loads(_gen(clf_mixed).model_dump_json()), dict)


def test_json_round_trip(clf_mixed):
    c = _gen(clf_mixed, objective="classify churn")
    assert ModelCandidates.model_validate_json(c.model_dump_json()) == c


def test_json_primitive_only(clf_mixed):
    payload = json.loads(_gen(clf_mixed).model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


# --- type validation ------------------------------------------


def test_non_dataframe_type_error(clf_mixed):
    _, problem, fe, readiness, split = clf_mixed
    with pytest.raises(TypeError):
        generate_model_candidates({"a": [1]}, problem, fe, readiness, split)


def test_non_problem_type_error(clf_mixed):
    df, _, fe, readiness, split = clf_mixed
    with pytest.raises(TypeError):
        generate_model_candidates(df, {"x": 1}, fe, readiness, split)


def test_non_fe_type_error(clf_mixed):
    df, problem, _, readiness, split = clf_mixed
    with pytest.raises(TypeError):
        generate_model_candidates(df, problem, {"x": 1}, readiness, split)


def test_non_readiness_type_error(clf_mixed):
    df, problem, fe, _, split = clf_mixed
    with pytest.raises(TypeError):
        generate_model_candidates(df, problem, fe, {"x": 1}, split)


def test_non_split_type_error(clf_mixed):
    df, problem, fe, readiness, _ = clf_mixed
    with pytest.raises(TypeError):
        generate_model_candidates(df, problem, fe, readiness, {"x": 1})


# --- upstream behaviour --------------------------------------


def test_readiness_unavailable(clf_mixed):
    df, problem, fe, _, split = clf_mixed
    bad = ModelReadiness(status=UNAVAILABLE, reason="x")
    c = generate_model_candidates(df, problem, fe, bad, split)
    assert c.status is UNAVAILABLE
    assert c.candidates == [] and c.candidates_detail == []
    assert "model readiness is not completed" in c.reason


def test_readiness_not_yet_inferred(clf_mixed):
    df, problem, fe, _, split = clf_mixed
    c = generate_model_candidates(df, problem, fe, ModelReadiness(), split)
    assert c.status is UNAVAILABLE


def test_readiness_not_ready(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    bad = readiness.model_copy(
        update={"ready": False, "blocking_issues": ["the dataset has 3 rows"]}
    )
    c = generate_model_candidates(df, problem, fe, bad, split)
    assert c.status is UNAVAILABLE
    assert "blocked by model-readiness issues" in c.reason


def test_split_unavailable(clf_mixed):
    df, problem, fe, readiness, _ = clf_mixed
    c = generate_model_candidates(
        df, problem, fe, readiness, DataSplitPlan(status=UNAVAILABLE, reason="x")
    )
    assert c.status is UNAVAILABLE
    assert "data-split plan is not completed" in c.reason


def test_split_not_yet_inferred(clf_mixed):
    df, problem, fe, readiness, _ = clf_mixed
    c = generate_model_candidates(df, problem, fe, readiness, DataSplitPlan())
    assert c.status is UNAVAILABLE


def test_task_unavailable(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    bad_task = problem.task_type.model_copy(
        update={"status": ProblemUnderstandingStatus.UNAVAILABLE}
    )
    c = generate_model_candidates(
        df, problem.model_copy(update={"task_type": bad_task}), fe, readiness, split
    )
    assert c.status is UNAVAILABLE
    assert "task-type inference is not completed" in c.reason


def test_task_none(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    bad_task = problem.task_type.model_copy(update={"task_type": None})
    c = generate_model_candidates(
        df, problem.model_copy(update={"task_type": bad_task}), fe, readiness, split
    )
    assert c.status is UNAVAILABLE
    assert "without a task type" in c.reason


def test_unsupported_task(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    bad_task = problem.task_type.model_copy(
        update={"task_type": TaskType.MULTILABEL_CLASSIFICATION}
    )
    c = generate_model_candidates(
        df, problem.model_copy(update={"task_type": bad_task}), fe, readiness, split
    )
    assert c.status is UNAVAILABLE
    assert "does not support task type 'multilabel_classification'" in c.reason


def test_deterministic_precedence_task_before_readiness(clf_mixed):
    df, problem, fe, _, split = clf_mixed
    bad_task = problem.task_type.model_copy(update={"task_type": None})
    c = generate_model_candidates(
        df,
        problem.model_copy(update={"task_type": bad_task}),
        fe,
        ModelReadiness(status=UNAVAILABLE, reason="also broken"),
        split,
    )
    assert "without a task type" in c.reason  # task checked first


def test_fe_assessment_incomplete_unavailable(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    bad_fe = fe.model_copy(
        update={"assessment": fe.assessment.model_copy(update={"status": NOT_YET})}
    )
    c = generate_model_candidates(df, problem, bad_fe, readiness, split)
    assert c.status is UNAVAILABLE
    assert "feature-engineering assessment is not completed" in c.reason


# --- candidate generation ----------------------------------


def test_regression_families(reg_mixed):
    c = _gen(reg_mixed)
    assert c.status is COMPLETED
    assert set(c.candidates) >= {"linear", "tree_based", "ensemble"}
    assert "distance_based" not in c.candidates  # categorical present


def test_regression_numeric_adds_distance(reg_numeric):
    c = _gen(reg_numeric)
    assert c.candidates == ["linear", "tree_based", "ensemble", "distance_based"]


def test_binary_classification_families(clf_mixed):
    c = _gen(clf_mixed)
    assert set(c.candidates) >= {"linear", "tree_based", "ensemble", "probabilistic"}
    assert "distance_based" not in c.candidates  # categorical present
    assert "neural" not in c.candidates  # below scale threshold


def test_multiclass_families(clf_numeric):
    c = _gen(clf_numeric)
    if clf_numeric[1].task_type.task_type is TaskType.MULTICLASS_CLASSIFICATION:
        assert set(c.candidates) >= {"linear", "tree_based", "ensemble", "probabilistic"}
        assert "distance_based" in c.candidates  # all-numeric features


def test_clustering_families(clustering):
    c = _gen(clustering)
    assert c.status is COMPLETED
    assert set(c.candidates) == {"distance_based", "probabilistic"}


def test_forecasting_families_and_boundary(forecasting):
    c = _gen(forecasting)
    assert c.status is COMPLETED
    assert set(c.candidates) == {"linear", "tree_based", "ensemble"}
    assert any("does not create lag features" in n for n in c.notes)
    assert all(any("lag features" in e for e in cand.evidence) for cand in c.candidates_detail)
    assert any("does not infer a forecasting task from a datetime column" in n for n in c.notes)


def test_candidate_ordering_deterministic(clf_mixed):
    c = _gen(clf_mixed)
    order = {
        "linear": 0,
        "tree_based": 1,
        "ensemble": 2,
        "probabilistic": 3,
        "distance_based": 4,
        "neural": 5,
    }
    ranks = [order[x] for x in c.candidates]
    assert ranks == sorted(ranks)


def test_no_duplicate_candidates(clf_mixed):
    c = _gen(clf_mixed)
    assert len(c.candidates) == len(set(c.candidates))


def test_structured_matches_string_candidates(clf_mixed):
    c = _gen(clf_mixed)
    assert [x.family.value for x in c.candidates_detail] == c.candidates


def test_neural_added_at_scale():
    rng = np.random.default_rng(9)
    n = 1200
    df = pd.DataFrame(
        {**{f"f{i}": rng.normal(0.0, 1.0, n) for i in range(25)}, "y": ([True, False] * (n // 2))}
    )
    fixture = _build(df, "classify y")
    c = generate_model_candidates(*fixture)
    if fixture[1].task_type.task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
        assert "neural" in c.candidates
        assert c.candidates[-1] == "neural"


# --- structural awareness --------------------------------


def test_preprocessing_note_present(clf_mixed):
    c = _gen(clf_mixed)
    assert any("preprocessing requirements exist upstream" in n for n in c.notes)


def test_fe_recommendations_note_present(clf_mixed):
    c = _gen(clf_mixed)
    assert any("feature-engineering recommendations are available upstream" in n for n in c.notes)


def test_evidence_has_structural_items_no_performance_claims(clf_mixed):
    c = _gen(clf_mixed)
    for cand in c.candidates_detail:
        joined = " ".join(cand.evidence + [cand.reason]).lower()
        assert "task type is" in " ".join(cand.evidence).lower()
        for banned in (
            "best model",
            "highest accuracy",
            "likely to outperform",
            "most accurate",
            "best predictive power",
            "lowest rmse",
        ):
            assert banned not in joined


# --- objective ------------------------------------------


def test_objective_none(clf_mixed):
    assert _gen(clf_mixed).objective_used is False


def test_objective_empty(clf_mixed):
    assert _gen(clf_mixed, objective="").objective_used is False


def test_objective_whitespace(clf_mixed):
    assert _gen(clf_mixed, objective="   ").objective_used is False


def test_objective_real_recorded(clf_mixed):
    c = _gen(clf_mixed, objective="please use XGBoost with target encoding")
    assert c.objective_used is True
    assert any("did not change any candidate family" in n for n in c.notes)


def test_objective_cannot_override_structural_rules(clf_mixed):
    plain = set(_gen(clf_mixed).candidates)
    biased = set(_gen(clf_mixed, objective="only a neural network, nothing else").candidates)
    assert plain == biased
    blob = _gen(clf_mixed, objective="use target encoding").model_dump_json().lower()
    assert "target encoding" not in blob
    assert "xgboost" not in blob


# --- determinism --------------------------------------


def test_repeated_calls_identical(clf_mixed):
    blobs = {_gen(clf_mixed).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_row_shuffle_identical(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    shuffled = df.sample(frac=1.0, random_state=7)
    assert generate_model_candidates(
        shuffled, problem, fe, readiness, split
    ) == generate_model_candidates(df, problem, fe, readiness, split)


def test_column_reorder_identical(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    reordered = df[list(df.columns)[::-1]]
    assert generate_model_candidates(
        reordered, problem, fe, readiness, split
    ) == generate_model_candidates(df, problem, fe, readiness, split)


def test_stable_reasons_and_evidence(clf_mixed):
    a = _gen(clf_mixed)
    b = _gen(clf_mixed)
    assert [(x.family, x.reason, x.evidence) for x in a.candidates_detail] == [
        (x.family, x.reason, x.evidence) for x in b.candidates_detail
    ]


# --- safety ------------------------------------------


def test_df_unchanged(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    before = df.copy(deep=True)
    generate_model_candidates(df, problem, fe, readiness, split, objective="classify churn")
    generate_model_candidates(df, problem, fe, readiness, split)
    pd.testing.assert_frame_equal(df, before)


def test_upstream_models_unchanged(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    snaps = tuple(m.model_dump_json() for m in (problem, fe, readiness, split))
    generate_model_candidates(df, problem, fe, readiness, split)
    assert tuple(m.model_dump_json() for m in (problem, fe, readiness, split)) == snaps


def test_no_files_created(clf_mixed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _gen(clf_mixed)
    assert list(tmp_path.iterdir()) == []


def test_no_model_objects_or_metrics(clf_mixed):
    blob = _gen(clf_mixed).model_dump_json().lower()
    for token in (
        "fitted estimator",
        "fitted model",
        "prediction",
        "accuracy",
        "rmse",
        "roc_auc",
        "n_estimators",
        "hyperparam",
        "shap",
        "run_id",
        "uuid",
        "timestamp",
        "generated_at",
    ):
        assert token not in blob


# --- integration -----------------------------------


def test_merge_into_modeling_spec(clf_mixed):
    df, problem, fe, readiness, split = clf_mixed
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective="classify churn"))
    spec = spec.model_copy(update={"readiness": readiness, "split": split})
    candidates = generate_model_candidates(
        df, problem, fe, readiness, split, objective="classify churn"
    )
    merged = spec.model_copy(update={"candidates": candidates})

    assert merged.candidates.status is COMPLETED
    assert merged.training.status is NOT_YET
    assert merged.evaluation.status is NOT_YET
    assert merged.selection.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility -----------------------


def test_phase_7_1_foundation_unchanged():
    spec = understand_modeling(ModelingRequest(dataset_id="d"))
    assert spec.candidates.status is NOT_YET
    assert spec.candidates.candidates == []


def test_understand_modeling_signature_unchanged():
    assert list(inspect.signature(understand_modeling).parameters) == ["request"]


def test_legacy_model_candidates_json_validates():
    legacy = json.dumps(
        {"status": "not_yet_inferred", "reason": None, "candidates": [], "notes": []}
    )
    model = ModelCandidates.model_validate_json(legacy)
    assert model.candidates_detail == []
    assert model.objective_used is False


def test_phase_7_2_still_works(reg_numeric):
    *_, readiness, split = reg_numeric
    assert readiness.status is COMPLETED
    assert split.status is COMPLETED


def test_phase_5_and_6_apis_still_work(clf_mixed):
    _, problem, fe, _, _ = clf_mixed
    assert problem.task_type.status is PU_DONE
    assert fe.assessment.status is FE_DONE
