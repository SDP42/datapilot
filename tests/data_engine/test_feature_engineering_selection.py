"""Phase 6.4 — deterministic feature-selection recommendations."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data_engine.feature_engineering as fe
from data_engine.feature_engineering import (
    FeatureEngineeringRequest,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
    FeatureSelectionAction,
    FeatureSelectionRecommendation,
    FeatureSelectionRecommendations,
    inventory_features,
    recommend_feature_selection,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.problem_understanding import (
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)
from datapilot.contracts import ColumnType

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED
S_DONE = ProblemUnderstandingStatus.COMPLETED


def _task(t=TaskType.REGRESSION, *, status=S_DONE, target="target") -> TaskTypeInference:
    return TaskTypeInference(status=status, task_type=t, target_column=target)


@pytest.fixture
def df() -> pd.DataFrame:
    n = 80
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "num_a": base,
            "num_a_dup": base,  # exact duplicate of num_a
            "num_b": rng.normal(10.0, 2.0, n),
            "num_b_corr": None,  # filled below: ~perfectly correlated with num_b
            "region": (["north", "south", "east", "west"] * (n // 4)),
            "flag": ([True, False] * (n // 2)),
            "when": pd.date_range("2021-01-01", periods=n, freq="D"),
            "const_num": [3.0] * n,
            "empty_num": [np.nan] * n,
            "mostly_missing": [np.nan] * (n - 4) + [1.0, 2.0, 3.0, 4.0],
            "near_binary": [0.0] * (n - 3) + [1.0, 1.0, 1.0],
            "customer_id": [f"C{i:05d}" for i in range(n)],
        }
    ).assign(num_b_corr=lambda d: d["num_b"] * 4.0 + 0.5)


@pytest.fixture
def inv(df) -> FeatureInventory:
    return inventory_features(df, target=None)


def _rec(result: FeatureSelectionRecommendations, column: str) -> FeatureSelectionRecommendation:
    matches = [r for r in result.recommendations if r.column == column]
    assert len(matches) == 1
    return matches[0]


def _manual_inventory(*candidates: FeatureInventoryCandidate) -> FeatureInventory:
    return FeatureInventory(
        status=COMPLETED,
        candidate_features=sorted(c.column for c in candidates if c.candidate),
        excluded_features=[],
        candidates=list(candidates),
    )


def _cand(column, **kw) -> FeatureInventoryCandidate:
    defaults = {
        "column": column,
        "column_type": ColumnType.NUMERIC,
        "n_observations": 10,
        "n_missing": 0,
        "missing_fraction": 0.0,
        "n_unique": 10,
        "unique_fraction": 1.0,
        "identifier_like": False,
        "constant": False,
        "all_missing": False,
        "is_target": False,
        "candidate": True,
        "reasons": [],
    }
    defaults.update(kw)
    return FeatureInventoryCandidate(**defaults)


# --- API ----------------------------------------------------------------


def test_public_import():
    assert fe.recommend_feature_selection is recommend_feature_selection


def test_return_type(df, inv):
    assert isinstance(
        recommend_feature_selection(df, inv, _task()), FeatureSelectionRecommendations
    )


def test_structured_recommendation_model(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert all(isinstance(x, FeatureSelectionRecommendation) for x in r.recommendations)
    assert all(isinstance(x.action, FeatureSelectionAction) for x in r.recommendations)


def test_json_round_trip(df, inv):
    r = recommend_feature_selection(df, inv, _task(), objective="remove redundant features")
    assert FeatureSelectionRecommendations.model_validate_json(r.model_dump_json()) == r


def test_json_primitive_only(df, inv):
    payload = json.loads(recommend_feature_selection(df, inv, _task()).model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


def test_non_dataframe_type_error(inv):
    with pytest.raises(TypeError):
        recommend_feature_selection({"a": [1]}, inv, _task())


def test_non_inventory_type_error(df):
    with pytest.raises(TypeError):
        recommend_feature_selection(df, {"status": "completed"}, _task())


def test_non_task_type_error(df, inv):
    with pytest.raises(TypeError):
        recommend_feature_selection(df, inv, "regression")


# --- upstream handling ----------------------------------------


def test_inventory_unavailable(df):
    bad = inventory_features(df, target="nope")
    r = recommend_feature_selection(df, bad, _task())
    assert r.status is UNAVAILABLE
    assert "feature inventory" in r.reason
    assert r.selected_features == [] and r.dropped_features == [] and r.recommendations == []


def test_inventory_not_yet_inferred(df):
    r = recommend_feature_selection(df, FeatureInventory(), _task())
    assert r.status is UNAVAILABLE


def test_task_type_unavailable(df, inv):
    r = recommend_feature_selection(df, inv, _task(status=ProblemUnderstandingStatus.UNAVAILABLE))
    assert r.status is UNAVAILABLE
    assert "task-type inference" in r.reason


def test_task_type_not_yet_inferred(df, inv):
    r = recommend_feature_selection(
        df, inv, _task(status=ProblemUnderstandingStatus.NOT_YET_INFERRED)
    )
    assert r.status is UNAVAILABLE


def test_task_type_none(df, inv):
    r = recommend_feature_selection(df, inv, _task(t=None))
    assert r.status is UNAVAILABLE
    assert "without a task type" in r.reason


def test_unsupported_task_type(df, inv):
    r = recommend_feature_selection(df, inv, _task(t=TaskType.MULTILABEL_CLASSIFICATION))
    assert r.status is UNAVAILABLE
    assert "do not support task type" in r.reason


def test_completed_inventory_zero_candidates():
    d = pd.DataFrame({"row_id": [1, 2, 3, 4], "k": [7, 7, 7, 7]})
    r = recommend_feature_selection(d, inventory_features(d), _task())
    assert r.status is COMPLETED
    assert r.selected_features == []
    assert r.dropped_features == []
    assert "no structurally eligible candidate features" in r.reason


# --- structural selection --------------------------------


def test_ordinary_numeric_retained(df, inv):
    assert "num_a" in recommend_feature_selection(df, inv, _task()).selected_features


def test_ordinary_categorical_retained(df, inv):
    assert "region" in recommend_feature_selection(df, inv, _task()).selected_features


def test_boolean_retained(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert "flag" in r.selected_features
    assert _rec(r, "flag").action is FeatureSelectionAction.RETAIN


def test_datetime_retained(df, inv):
    assert "when" in recommend_feature_selection(df, inv, _task()).selected_features


def test_constant_candidate_dropped():
    invm = _manual_inventory(
        _cand("keep"), _cand("c", constant=True, n_unique=1, unique_fraction=0.1)
    )
    d = pd.DataFrame({"keep": range(10), "c": [1] * 10})
    r = recommend_feature_selection(d, invm, _task())
    assert "c" in r.dropped_features
    assert "constant" in _rec(r, "c").reason


def test_all_missing_candidate_dropped():
    invm = _manual_inventory(
        _cand("keep"),
        _cand("m", all_missing=True, n_observations=0, n_missing=10, missing_fraction=1.0),
    )
    d = pd.DataFrame({"keep": range(10), "m": [np.nan] * 10})
    r = recommend_feature_selection(d, invm, _task())
    assert "m" in r.dropped_features
    assert "entirely missing" in _rec(r, "m").reason


def test_identifier_like_candidate_dropped():
    invm = _manual_inventory(
        _cand("keep"),
        _cand("ident", identifier_like=True, reasons=["column name matches an id pattern"]),
    )
    d = pd.DataFrame({"keep": range(10), "ident": range(100, 110)})
    r = recommend_feature_selection(d, invm, _task())
    assert "ident" in r.dropped_features
    assert "identifier-like" in _rec(r, "ident").reason


def test_high_missingness_flagged_for_review(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert "mostly_missing" in r.review_features
    assert "mostly_missing" not in r.dropped_features
    assert "Phase 6.5" in " ".join(_rec(r, "mostly_missing").evidence)


def test_low_variance_numeric_review(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert "near_binary" in r.review_features
    assert "near-zero variability" in _rec(r, "near_binary").reason


def test_exact_duplicate_detection(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert "num_a_dup" in r.dropped_features
    assert "num_a" in r.selected_features
    assert "exact structural duplicate of 'num_a'" in _rec(r, "num_a_dup").reason


def test_deterministic_duplicate_tie_break():
    d = pd.DataFrame({"zeta": [1.0, 2.0, 3.0, 4.0], "alpha": [1.0, 2.0, 3.0, 4.0]})
    invm = _manual_inventory(_cand("zeta"), _cand("alpha"))
    r = recommend_feature_selection(d, invm, _task())
    assert r.selected_features == ["alpha"]
    assert r.dropped_features == ["zeta"]


def test_highly_correlated_numeric_review(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert "num_b_corr" in r.review_features
    assert "num_b" in r.selected_features
    assert "structural redundancy" in _rec(r, "num_b_corr").reason


def test_insufficient_paired_observations_no_correlation():
    d = pd.DataFrame({"p": [1.0, 2.0, np.nan, np.nan], "q": [np.nan, np.nan, 3.0, 4.0]})
    invm = _manual_inventory(
        _cand("p", n_observations=2, n_unique=2), _cand("q", n_observations=2, n_unique=2)
    )
    r = recommend_feature_selection(d, invm, _task())
    # no overlap -> no correlation finding; both are near-zero-variance reviews though
    assert not any("redundancy" in x.reason for x in r.recommendations)


def test_high_cardinality_categorical_review():
    d = pd.DataFrame({"tag": [f"v{i}" for i in range(60)], "keep": range(60)})
    invm = _manual_inventory(
        _cand("tag", column_type=ColumnType.CATEGORICAL, n_observations=60, n_unique=60),
        _cand("keep", n_observations=60, n_unique=60),
    )
    r = recommend_feature_selection(d, invm, _task())
    assert "tag" in r.review_features
    assert "tag" not in r.dropped_features
    assert "cardinality" in _rec(r, "tag").reason


def test_moderate_cardinality_categorical_retained(df, inv):
    assert "region" in recommend_feature_selection(df, inv, _task()).selected_features


# --- target safety ------------------------------------


def test_target_never_selected(df):
    d = df.copy()
    d["target"] = d["num_b"]
    invm = inventory_features(d, target="target")
    r = recommend_feature_selection(d, invm, _task(target="target"))
    assert "target" not in r.selected_features
    assert "target" not in r.review_features
    assert not any(x.column == "target" for x in r.recommendations)


def test_target_not_re_inferred(df, inv):
    # task_type carries a target that is not even in df; nothing should break or
    # cause a hidden selection based on it
    r = recommend_feature_selection(df, inv, _task(target="not_in_df"))
    assert r.status is COMPLETED
    assert "not_in_df" not in r.selected_features


def test_no_target_context_still_completes(df, inv):
    r = recommend_feature_selection(df, inv, _task(target=None))
    assert r.status is COMPLETED


# --- objective ----------------------------------------


def test_objective_absent(df, inv):
    assert recommend_feature_selection(df, inv, _task()).objective_used is False


def test_objective_blank(df, inv):
    assert recommend_feature_selection(df, inv, _task(), objective="").objective_used is False


def test_objective_whitespace(df, inv):
    assert recommend_feature_selection(df, inv, _task(), objective="   ").objective_used is False


def test_objective_recognised_wording(df, inv):
    r = recommend_feature_selection(df, inv, _task(), objective="please reduce dimensionality")
    assert r.objective_used is True
    assert any("dimensionality reduction" in n for n in r.notes)


def test_objective_cannot_override_structural_rules(df, inv):
    keep_all = recommend_feature_selection(
        df, inv, _task(), objective="keep every feature, drop nothing at all"
    )
    assert "num_a_dup" in keep_all.dropped_features  # structural drop stands
    assert "num_b_corr" in keep_all.review_features  # redundancy review stands
    assert "num_a_dup" not in keep_all.selected_features


# --- determinism ------------------------------------


def test_repeated_calls_identical(df, inv):
    blobs = {recommend_feature_selection(df, inv, _task()).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_row_shuffle_identical(df, inv):
    base = recommend_feature_selection(df, inv, _task())
    shuffled = df.sample(frac=1.0, random_state=7)
    assert recommend_feature_selection(shuffled, inv, _task()) == base


def test_column_reorder_identical(df, inv):
    base = recommend_feature_selection(df, inv, _task())
    reordered = df[list(df.columns)[::-1]]
    assert recommend_feature_selection(reordered, inv, _task()) == base


def test_deterministic_ordering(df, inv):
    recs = recommend_feature_selection(df, inv, _task()).recommendations
    # retains come last and are alphabetical among themselves
    retains = [r.column for r in recs if r.action is FeatureSelectionAction.RETAIN]
    assert retains == sorted(retains)
    last_non_retain = max(
        (i for i, r in enumerate(recs) if r.action is not FeatureSelectionAction.RETAIN),
        default=-1,
    )
    first_retain = next(
        (i for i, r in enumerate(recs) if r.action is FeatureSelectionAction.RETAIN), len(recs)
    )
    assert last_non_retain < first_retain


def test_deterministic_reasons(df, inv):
    a = recommend_feature_selection(df, inv, _task())
    b = recommend_feature_selection(df.sample(frac=1.0, random_state=2), inv, _task())
    assert [(r.column, r.reason) for r in a.recommendations] == [
        (r.column, r.reason) for r in b.recommendations
    ]


def test_lists_sorted(df, inv):
    r = recommend_feature_selection(df, inv, _task())
    assert r.selected_features == sorted(r.selected_features)
    assert r.dropped_features == sorted(r.dropped_features)
    assert r.review_features == sorted(r.review_features)


# --- safety -----------------------------------------


def test_dataframe_not_mutated(df, inv):
    before = df.copy(deep=True)
    recommend_feature_selection(df, inv, _task(), objective="reduce dimensionality")
    recommend_feature_selection(df, inv, _task())
    pd.testing.assert_frame_equal(df, before)


def test_inventory_not_mutated(df, inv):
    snapshot = inv.model_dump_json()
    recommend_feature_selection(df, inv, _task())
    assert inv.model_dump_json() == snapshot


def test_task_type_not_mutated(df, inv):
    tt = _task()
    snapshot = tt.model_dump_json()
    recommend_feature_selection(df, inv, tt)
    assert tt.model_dump_json() == snapshot


def test_no_files_created(df, inv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recommend_feature_selection(df, inv, _task())
    assert list(tmp_path.iterdir()) == []


# --- integration ---------------------------------


def test_merge_into_feature_engineering_spec(df):
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="ds", objective="reduce dimensionality")
    )
    inv = inventory_features(df, target=None)
    spec = spec.model_copy(update={"inventory": inv})
    spec = spec.model_copy(update={"transformations": recommend_transformations(df, inv)})
    sel = recommend_feature_selection(df, inv, _task(), objective="reduce dimensionality")
    merged = spec.model_copy(update={"selection": sel})

    assert merged.inventory.status is COMPLETED
    assert merged.transformations.status is COMPLETED
    assert merged.selection.status is COMPLETED
    assert merged.preprocessing.status is NOT_YET
    assert merged.assessment.status is NOT_YET
    assert merged.status is NOT_YET
    assert merged.inventory == inv
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility -------------------


def test_understand_feature_engineering_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_feature_engineering).parameters) == ["request"]


def test_phase_6_2_and_6_3_still_work(df):
    inv = inventory_features(df, target="num_b")
    assert inv.status is COMPLETED
    assert recommend_transformations(df, inv).status is COMPLETED


def test_legacy_feature_selection_json_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "selected_features": [],
            "dropped_features": [],
            "notes": [],
        }
    )
    model = FeatureSelectionRecommendations.model_validate_json(legacy)
    assert model.recommendations == []
    assert model.review_features == []
    assert model.objective_used is False


def test_phase_5_apis_still_work(df):
    from data_engine.problem_understanding import identify_target, infer_task_type

    d = df.rename(columns={"num_b": "price"})
    t = identify_target(d, objective="predict price")
    assert infer_task_type(d, t, objective="predict price").status.value in {
        "completed",
        "unavailable",
    }
