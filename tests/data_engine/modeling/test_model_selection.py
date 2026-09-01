"""Phase 7.5 — model selection & recommendation."""

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
    DataSplitStrategy,
    ModelCandidates,
    ModelFamily,
    ModelingRequest,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    ModelSelectionRank,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
    assess_model_readiness,
    generate_model_candidates,
    recommend_data_split,
    select_model,
    train_and_evaluate_models,
    understand_modeling,
)
from data_engine.problem_understanding import (
    ProblemSpec,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
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


# --- real end-to-end fixtures (module-scoped: the 7.4 training is slow) ---


def _pipeline(df: pd.DataFrame, objective: str):
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
    training = train_and_evaluate_models(df, problem, fe, readiness, split, candidates)
    return problem, fe, readiness, split, candidates, training


@pytest.fixture(scope="module")
def clf_pipeline():
    rng = np.random.default_rng(0)
    s = rng.normal(0.0, 1.0, _N)
    df = pd.DataFrame(
        {
            "signal": s,
            "noise": rng.normal(0.0, 1.0, _N),
            "region": (["n", "s", "e", "w"] * (_N // 4)),
            "churn": (s + rng.normal(0.0, 0.5, _N) > 0.0),
        }
    )
    return _pipeline(df, "classify churn")


@pytest.fixture(scope="module")
def reg_pipeline():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.0, 10.0, _N)
    df = pd.DataFrame(
        {"s": s, "t": rng.normal(0.0, 1.0, _N), "price": 3.0 * s + rng.normal(0.0, 1.0, _N)}
    )
    return _pipeline(df, "predict the price")


@pytest.fixture(scope="module")
def clustering_pipeline():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "a": np.r_[rng.normal(-4.0, 1.0, _N // 2), rng.normal(4.0, 1.0, _N // 2)],
            "b": np.r_[rng.normal(-4.0, 1.0, _N // 2), rng.normal(4.0, 1.0, _N // 2)],
        }
    )
    out = _pipeline(df, "segment customers into clusters")
    if out[0].task_type.task_type is not TaskType.CLUSTERING:
        pytest.skip("task inference did not yield clustering")
    return out


@pytest.fixture(scope="module")
def forecasting_pipeline():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "day": pd.date_range("2021-01-01", periods=_N, freq="D"),
            "temp": rng.normal(20.0, 3.0, _N),
            "demand": rng.uniform(10.0, 90.0, _N),
        }
    )
    out = _pipeline(df, "forecast future demand over time")
    if out[0].task_type.task_type is not TaskType.TIME_SERIES_FORECASTING:
        pytest.skip("task inference did not yield forecasting")
    return out


def _select(pipeline, **kw) -> ModelSelection:
    problem, fe, readiness, split, candidates, training = pipeline
    return select_model(problem, fe, readiness, split, candidates, training, **kw)


# --- synthetic-contract builders (fast, controlled) ---------------------


def _completed(model, **kw):
    return model.model_copy(update={"status": COMPLETED, **kw})


def _synthetic_problem(task: TaskType) -> ProblemSpec:
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="d"))
    return spec.model_copy(update={"task_type": TaskTypeInference(status=PU_DONE, task_type=task)})


def _synthetic_fe():
    fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d"))
    return fe.model_copy(
        update={
            "assessment": fe.assessment.model_copy(update={"status": FE_DONE, "feasible": True})
        }
    )


def _synthetic_readiness(ready: bool = True) -> ModelReadiness:
    return ModelReadiness(status=COMPLETED, ready=ready, blocking_issues=[] if ready else ["x"])


def _synthetic_split() -> DataSplitPlan:
    return DataSplitPlan(status=COMPLETED, strategy=DataSplitStrategy.RANDOM_HOLDOUT)


def _synthetic_candidates(*families: str) -> ModelCandidates:
    return ModelCandidates(status=COMPLETED, candidates=list(families))


def _run(family: str, estimator: str, status: TrainingRunStatus, metrics=None, reason=None):
    return TrainingRun(
        family=ModelFamily(family),
        estimator_name=estimator,
        status=status,
        metrics=metrics or {},
        reason=reason,
    )


def _training(*runs: TrainingRun) -> TrainingOutcome:
    return TrainingOutcome(
        status=COMPLETED,
        runs=list(runs),
        successful_runs=[r.family.value for r in runs if r.status is TrainingRunStatus.COMPLETED],
        failed_runs=[r.family.value for r in runs if r.status is not TrainingRunStatus.COMPLETED],
    )


def _sel(
    task: TaskType,
    training: TrainingOutcome,
    *,
    families: tuple[str, ...] = (
        "linear",
        "tree_based",
        "ensemble",
        "probabilistic",
        "distance_based",
    ),
    ready: bool = True,
    **kw,
) -> ModelSelection:
    return select_model(
        _synthetic_problem(task),
        _synthetic_fe(),
        _synthetic_readiness(ready),
        _synthetic_split(),
        _synthetic_candidates(*families),
        training,
        **kw,
    )


# --- API ------------------------------------------------------------


def test_public_import():
    assert modeling.select_model is select_model
    assert "select_model" in modeling.__all__
    assert "ModelSelectionRank" in modeling.__all__


def test_return_type():
    out = _sel(
        TaskType.REGRESSION,
        _training(_run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0})),
    )
    assert isinstance(out, ModelSelection)


def test_exact_signature():
    assert list(inspect.signature(select_model).parameters) == [
        "problem",
        "feature_engineering",
        "readiness",
        "split",
        "candidates",
        "training",
        "objective",
    ]


def test_structured_ranking():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 2.0}),
            _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
    )
    assert all(isinstance(r, ModelSelectionRank) for r in out.ranking)


def test_json_round_trip(reg_pipeline):
    out = _select(reg_pipeline, objective="predict the price")
    assert ModelSelection.model_validate_json(out.model_dump_json()) == out


def test_json_primitive_only(clf_pipeline):
    payload = json.loads(_select(clf_pipeline).model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


# --- type guards -------------------------------------------------


@pytest.mark.parametrize("bad_index", range(6))
def test_type_guards(bad_index):
    args = [
        _synthetic_problem(TaskType.REGRESSION),
        _synthetic_fe(),
        _synthetic_readiness(),
        _synthetic_split(),
        _synthetic_candidates("linear"),
        _training(_run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0})),
    ]
    args[bad_index] = object()
    with pytest.raises(TypeError):
        select_model(*args)


# --- task selection metric --------------------------------------


def test_regression_uses_rmse_minimize():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run(
                "linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 2.0, "mae": 1.0}
            ),
            _run("tree_based", "DecisionTreeRegressor", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
    )
    assert out.selection_metric == "rmse"
    assert out.selection_direction == "minimize"
    assert out.selected_family == "tree_based"
    assert out.selected_score == 1.0


def test_forecasting_uses_rmse_minimize():
    out = _sel(
        TaskType.TIME_SERIES_FORECASTING,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 5.0}),
            _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 4.0}),
        ),
        families=("linear", "tree_based", "ensemble"),
    )
    assert out.selection_metric == "rmse"
    assert out.selected_family == "ensemble"
    assert any("does not infer forecasting from datetime columns" in n for n in out.notes)


def test_binary_classification_uses_f1_maximize():
    out = _sel(
        TaskType.BINARY_CLASSIFICATION,
        _training(
            _run(
                "linear",
                "LogisticRegression",
                TrainingRunStatus.COMPLETED,
                {"accuracy": 0.99, "f1": 0.6},
            ),
            _run(
                "ensemble",
                "RandomForestClassifier",
                TrainingRunStatus.COMPLETED,
                {"accuracy": 0.8, "f1": 0.9},
            ),
        ),
    )
    assert out.selection_metric == "f1"
    assert out.selection_direction == "maximize"
    assert out.selected_family == "ensemble"  # higher f1, not higher accuracy


def test_multiclass_uses_f1_maximize():
    out = _sel(
        TaskType.MULTICLASS_CLASSIFICATION,
        _training(_run("linear", "LogisticRegression", TrainingRunStatus.COMPLETED, {"f1": 0.7})),
    )
    assert out.selection_metric == "f1" and out.selected_family == "linear"


def test_clustering_uses_silhouette_maximize():
    out = _sel(
        TaskType.CLUSTERING,
        _training(
            _run(
                "distance_based",
                "KMeans",
                TrainingRunStatus.COMPLETED,
                {"silhouette_score": 0.5, "davies_bouldin_score": 0.9},
            ),
            _run(
                "probabilistic",
                "GaussianMixture",
                TrainingRunStatus.COMPLETED,
                {"silhouette_score": 0.7},
            ),
        ),
        families=("probabilistic", "distance_based"),
    )
    assert out.selection_metric == "silhouette_score"
    assert out.selection_direction == "maximize"
    assert out.selected_family == "probabilistic"


# --- ranking -------------------------------------------------


def test_ranking_ascending_rmse():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 3.0}),
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("tree_based", "DecisionTreeRegressor", TrainingRunStatus.COMPLETED, {"rmse": 2.0}),
        ),
    )
    scores = [r.score for r in out.ranking if r.rank is not None]
    assert scores == sorted(scores)
    assert [r.rank for r in out.ranking if r.rank is not None] == [1, 2, 3]


def test_ranking_descending_f1():
    out = _sel(
        TaskType.BINARY_CLASSIFICATION,
        _training(
            _run("linear", "LogisticRegression", TrainingRunStatus.COMPLETED, {"f1": 0.4}),
            _run("ensemble", "RandomForestClassifier", TrainingRunStatus.COMPLETED, {"f1": 0.9}),
        ),
    )
    scores = [r.score for r in out.ranking if r.rank is not None]
    assert scores == sorted(scores, reverse=True)


def test_family_tie_break():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
    )
    assert out.selected_family == "linear"  # linear precedes ensemble in the fixed order
    assert any("tied on rmse" in n for n in out.notes)


def test_estimator_name_tie_break():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "ZLinear", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("linear", "ALinear", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
        families=("linear",),
    )
    assert out.selected_estimator == "ALinear"


def test_no_duplicate_ranking_entries():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("tree_based", "DecisionTreeRegressor", TrainingRunStatus.FAILED, reason="boom"),
        ),
    )
    keys = [(r.family, r.estimator_name) for r in out.ranking]
    assert len(keys) == len(set(keys))


# --- missing metrics --------------------------------------


def test_required_metric_absent_not_substituted():
    out = _sel(
        TaskType.BINARY_CLASSIFICATION,
        _training(
            _run(
                "linear",
                "LogisticRegression",
                TrainingRunStatus.COMPLETED,
                {"accuracy": 0.9, "roc_auc": 0.95},
            )
        ),
    )
    assert out.status is COMPLETED
    assert out.selected_family is None
    assert out.selected_score is None
    assert "no completed training run had a usable 'f1'" in out.reason
    entry = out.ranking[0]
    assert entry.score is None
    assert "selection metric 'f1' is unavailable" in entry.reason


def test_run_with_metric_ranked_over_run_without():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("tree_based", "DecisionTreeRegressor", TrainingRunStatus.COMPLETED, {"mae": 1.0}),
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 2.0}),
        ),
    )
    assert out.selected_family == "linear"
    ranks = {r.family: r.rank for r in out.ranking}
    assert ranks["linear"] == 1
    assert ranks["tree_based"] is None


# --- failed / unavailable runs --------------------------


def test_failed_run_ignored_but_visible():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run(
                "ensemble",
                "RandomForestRegressor",
                TrainingRunStatus.FAILED,
                reason="ValueError: bad",
            ),
        ),
    )
    assert out.selected_family == "linear"
    failed = next(r for r in out.ranking if r.family == "ensemble")
    assert failed.rank is None
    assert failed.status == "failed"
    assert "training run is failed: ValueError: bad" in failed.reason


def test_unavailable_run_ignored():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("neural", "MLPRegressor", TrainingRunStatus.UNAVAILABLE, reason="no estimator"),
        ),
        families=("linear", "neural"),
    )
    assert out.selected_family == "linear"
    entry = next(r for r in out.ranking if r.family == "neural")
    assert entry.rank is None


def test_all_runs_failed():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.FAILED, reason="boom"),
            _run("tree_based", "DecisionTreeRegressor", TrainingRunStatus.FAILED, reason="boom"),
        ),
    )
    assert out.status is COMPLETED
    assert out.selected_family is None
    assert out.selection_metric == "rmse"


def test_mixed_success_failed_unavailable():
    out = _sel(
        TaskType.BINARY_CLASSIFICATION,
        _training(
            _run("linear", "LogisticRegression", TrainingRunStatus.COMPLETED, {"f1": 0.8}),
            _run("tree_based", "DecisionTreeClassifier", TrainingRunStatus.FAILED, reason="x"),
            _run("neural", "MLPClassifier", TrainingRunStatus.UNAVAILABLE, reason="y"),
        ),
        families=("linear", "tree_based", "neural"),
    )
    assert out.selected_family == "linear"
    assert len(out.ranking) == 3


# --- empty training / no runs -----------------------


def test_empty_training_outcome():
    out = _sel(TaskType.REGRESSION, TrainingOutcome(status=COMPLETED))
    assert out.status is COMPLETED
    assert out.selected_family is None
    assert out.ranking == []
    assert "no model training runs are available for selection" in out.reason


# --- upstream precedence ---------------------------


@pytest.mark.parametrize(
    ("broken", "fragment"),
    [
        ("task", "task-type inference is not completed"),
        ("readiness", "model readiness is not completed"),
        ("ready", "blocked by model-readiness issues"),
        ("split", "data-split plan is not completed"),
        ("candidates", "model candidates are not available"),
        ("training", "model training is not completed"),
        ("assessment", "feature-engineering assessment is not completed"),
    ],
)
def test_upstream_precedence(broken, fragment):
    problem = _synthetic_problem(TaskType.REGRESSION)
    fe = _synthetic_fe()
    readiness = _synthetic_readiness(True)
    split = _synthetic_split()
    candidates = _synthetic_candidates("linear")
    training = _training(
        _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0})
    )

    if broken == "task":
        problem = problem.model_copy(
            update={
                "task_type": TaskTypeInference(status=ProblemUnderstandingStatus.NOT_YET_INFERRED)
            }
        )
    elif broken == "readiness":
        readiness = ModelReadiness()
    elif broken == "ready":
        readiness = ModelReadiness(status=COMPLETED, ready=False, blocking_issues=["3 rows"])
    elif broken == "split":
        split = DataSplitPlan()
    elif broken == "candidates":
        candidates = ModelCandidates()
    elif broken == "training":
        training = TrainingOutcome(status=UNAVAILABLE, reason="x")
    elif broken == "assessment":
        fe = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="d"))

    out = select_model(problem, fe, readiness, split, candidates, training)
    assert out.status is UNAVAILABLE
    assert fragment in out.reason
    assert out.ranking == []


def test_unsupported_task():
    out = _sel(
        TaskType.MULTILABEL_CLASSIFICATION,
        _training(_run("linear", "LogisticRegression", TrainingRunStatus.COMPLETED, {"f1": 0.9})),
    )
    assert out.status is UNAVAILABLE
    assert "does not support task type 'multilabel_classification'" in out.reason


def test_precedence_task_before_readiness():
    problem = _synthetic_problem(TaskType.REGRESSION).model_copy(
        update={"task_type": TaskTypeInference(status=PU_DONE, task_type=None)}
    )
    out = select_model(
        problem,
        _synthetic_fe(),
        ModelReadiness(status=UNAVAILABLE),
        _synthetic_split(),
        _synthetic_candidates("linear"),
        _training(),
    )
    assert "without a task type" in out.reason


# --- candidate consistency -----------------------


def test_unknown_family_run_not_selected():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("neural", "MLPRegressor", TrainingRunStatus.COMPLETED, {"rmse": 0.1}),
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
        families=("linear", "tree_based"),  # 'neural' is not a candidate
    )
    assert out.selected_family == "linear"
    entry = next(r for r in out.ranking if r.family == "neural")
    assert entry.rank is None
    assert "not a Phase 7.3 candidate" in entry.reason


# --- objective -----------------------------------


def test_objective_none():
    out = _sel(
        TaskType.REGRESSION,
        _training(_run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0})),
    )
    assert out.objective_used is False


def test_objective_blank_and_whitespace():
    t = _training(_run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}))
    assert _sel(TaskType.REGRESSION, t, objective="").objective_used is False
    assert _sel(TaskType.REGRESSION, t, objective="   ").objective_used is False


def test_objective_cannot_override_metric_rules():
    training = _training(
        _run(
            "linear",
            "LogisticRegression",
            TrainingRunStatus.COMPLETED,
            {"accuracy": 0.99, "f1": 0.5},
        ),
        _run(
            "ensemble",
            "RandomForestClassifier",
            TrainingRunStatus.COMPLETED,
            {"accuracy": 0.7, "f1": 0.8},
        ),
    )
    plain = _sel(TaskType.BINARY_CLASSIFICATION, training)
    biased = _sel(
        TaskType.BINARY_CLASSIFICATION,
        training,
        objective="pick the most accurate model, forget f1, and use ROC-AUC",
    )
    assert plain.selected_family == biased.selected_family == "ensemble"
    assert biased.selection_metric == "f1"
    assert biased.objective_used is True


# --- determinism ------------------------------


def test_five_repeated_calls_identical(clf_pipeline):
    blobs = {_select(clf_pipeline).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_ranking_and_reasons_stable():
    training = _training(
        _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 2.0}),
        _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
    )
    a = _sel(TaskType.REGRESSION, training)
    b = _sel(TaskType.REGRESSION, training)
    assert [(r.family, r.rank, r.score, r.reason) for r in a.ranking] == [
        (r.family, r.rank, r.score, r.reason) for r in b.ranking
    ]
    assert a.notes == b.notes


def test_no_statistical_superiority_claim():
    out = _sel(
        TaskType.REGRESSION,
        _training(
            _run("linear", "LinearRegression", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
            _run("ensemble", "RandomForestRegressor", TrainingRunStatus.COMPLETED, {"rmse": 1.0}),
        ),
    )
    blob = out.model_dump_json().lower()
    for term in ("significan", "p-value", "p value", "confidence interval", "hypothesis test"):
        assert term not in blob
    assert any("tie is an ordering choice" in n for n in out.notes)


# --- safety ------------------------------------


def test_upstream_models_unchanged(clf_pipeline):
    problem, fe, readiness, split, candidates, training = clf_pipeline
    snaps = tuple(
        m.model_dump_json() for m in (problem, fe, readiness, split, candidates, training)
    )
    select_model(problem, fe, readiness, split, candidates, training, objective="x")
    select_model(problem, fe, readiness, split, candidates, training)
    assert (
        tuple(m.model_dump_json() for m in (problem, fe, readiness, split, candidates, training))
        == snaps
    )


def test_no_files_created(clf_pipeline, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _select(clf_pipeline)
    assert list(tmp_path.iterdir()) == []


def test_no_estimator_or_recompute_in_output(clf_pipeline):
    out = _select(clf_pipeline)
    blob = out.model_dump_json().lower()
    for token in ("0x", "object at", ".pkl", ".joblib", "predict(", "fit(", "traceback"):
        assert token not in blob


def test_notes_transparency(reg_pipeline):
    out = _select(reg_pipeline)
    joined = " ".join(out.notes)
    assert "no model was" in joined and "retrained" in joined
    assert "no metric recomputed" in joined
    assert "no final estimator artifact was persisted" in joined


# --- integration ----------------------------


def test_merge_into_modeling_spec(clf_pipeline):
    problem, fe, readiness, split, candidates, training = clf_pipeline
    spec = understand_modeling(ModelingRequest(dataset_id="d", objective="classify churn"))
    spec = spec.model_copy(
        update={
            "readiness": readiness,
            "split": split,
            "candidates": candidates,
            "training": training,
        }
    )
    selection = select_model(
        problem, fe, readiness, split, candidates, training, objective="classify churn"
    )
    merged = spec.model_copy(update={"selection": selection})

    assert merged.selection.status is COMPLETED
    assert merged.readiness == readiness
    assert merged.split == split
    assert merged.candidates == candidates
    assert merged.training == training
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility ---------------


def test_phase_7_1_foundation_unchanged():
    spec = understand_modeling(ModelingRequest(dataset_id="d"))
    assert spec.selection.status is NOT_YET
    assert spec.selection.ranking == []
    assert spec.selection.selected_family is None


def test_understand_modeling_signature_unchanged():
    assert list(inspect.signature(understand_modeling).parameters) == ["request"]


def test_legacy_model_selection_json_validates():
    legacy = json.dumps({"status": "not_yet_inferred", "reason": None, "notes": []})
    model = ModelSelection.model_validate_json(legacy)
    assert model.ranking == []
    assert model.selected_family is None
    assert model.objective_used is False


def test_full_phase_7_pipeline_still_works(reg_pipeline):
    problem, fe, readiness, split, candidates, training = reg_pipeline
    assert readiness.status is COMPLETED
    assert split.status is COMPLETED
    assert candidates.status is COMPLETED
    assert training.status is COMPLETED
    assert select_model(problem, fe, readiness, split, candidates, training).status is COMPLETED
