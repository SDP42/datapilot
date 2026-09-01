"""Phase 6.5 — deterministic preprocessing requirements."""

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
    FeatureOperationType,
    FeatureSelectionRecommendations,
    PreprocessingRequirement,
    PreprocessingRequirements,
    TransformationRecommendations,
    inventory_features,
    recommend_feature_selection,
    recommend_preprocessing,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.problem_understanding import (
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED

_OP_IMPUTATION = "missing-value imputation"
_OP_ENCODING = "categorical encoding"
_OP_SCALING = "numerical scaling"


def _task() -> TaskTypeInference:
    return TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED, task_type=TaskType.REGRESSION
    )


@pytest.fixture
def df() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "big_scale": rng.uniform(
                40000.0, 60000.0, n
            ),  # large magnitude, ~symmetric -> 6.3 scaling
            "age": [30.0, 31.0, 29.0, 40.0, 35.0, 33.0] * 10,
            "age_missing": ([25.0, 26.0, 27.0] * ((n - 6) // 3)) + [np.nan] * 6,  # 10% missing
            "city": (["ny", "sf", "la", "chi"] * (n // 4)),
            "flag": ([True, False] * (n // 2)),
            "when": pd.date_range("2021-01-01", periods=n, freq="D"),
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "const_col": [1.0] * n,
        }
    )


@pytest.fixture
def inv(df) -> FeatureInventory:
    return inventory_features(df, target=None)


@pytest.fixture
def trans(df, inv) -> TransformationRecommendations:
    return recommend_transformations(df, inv)


@pytest.fixture
def sel(df, inv) -> FeatureSelectionRecommendations:
    return recommend_feature_selection(df, inv, _task())


@pytest.fixture
def pp(df, inv, trans, sel) -> PreprocessingRequirements:
    return recommend_preprocessing(df, inv, trans, sel)


def _req(result: PreprocessingRequirements, column: str) -> list[PreprocessingRequirement]:
    return [r for r in result.requirements if r.column == column]


# --- API ----------------------------------------------------------------


def test_public_import():
    assert fe.recommend_preprocessing is recommend_preprocessing


def test_return_type(pp):
    assert isinstance(pp, PreprocessingRequirements)


def test_structured_output(pp):
    assert all(isinstance(r, PreprocessingRequirement) for r in pp.requirements)
    assert all(isinstance(r.operation, FeatureOperationType) for r in pp.requirements)


def test_json_serialisation(pp):
    assert isinstance(json.loads(pp.model_dump_json()), dict)


def test_json_round_trip(df, inv, trans, sel):
    r = recommend_preprocessing(df, inv, trans, sel, objective="prepare features")
    assert PreprocessingRequirements.model_validate_json(r.model_dump_json()) == r


def test_json_primitive_only(pp):
    payload = json.loads(pp.model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


def test_type_guard_df(inv, trans, sel):
    with pytest.raises(TypeError):
        recommend_preprocessing({"a": [1]}, inv, trans, sel)


def test_type_guard_inventory(df, trans, sel):
    with pytest.raises(TypeError):
        recommend_preprocessing(df, {"status": "completed"}, trans, sel)


def test_type_guard_transformations(df, inv, sel):
    with pytest.raises(TypeError):
        recommend_preprocessing(df, inv, {"status": "completed"}, sel)


def test_type_guard_selection(df, inv, trans):
    with pytest.raises(TypeError):
        recommend_preprocessing(df, inv, trans, {"status": "completed"})


# --- upstream handling ------------------------------------


def test_inventory_unavailable(df, trans, sel):
    bad = inventory_features(df, target="nope")
    r = recommend_preprocessing(df, bad, trans, sel)
    assert r.status is UNAVAILABLE
    assert "feature inventory" in r.reason
    assert r.required_operations == []
    assert not (r.encoding_required or r.scaling_required or r.imputation_required)


def test_inventory_not_yet_inferred(df, trans, sel):
    r = recommend_preprocessing(df, FeatureInventory(), trans, sel)
    assert r.status is UNAVAILABLE


def test_transformations_unavailable(df, inv, sel):
    bad = TransformationRecommendations(status=UNAVAILABLE, reason="x")
    r = recommend_preprocessing(df, inv, bad, sel)
    assert r.status is UNAVAILABLE
    assert "transformation recommendations" in r.reason


def test_transformations_not_yet_inferred(df, inv, sel):
    r = recommend_preprocessing(df, inv, TransformationRecommendations(), sel)
    assert r.status is UNAVAILABLE


def test_selection_unavailable(df, inv, trans):
    bad = FeatureSelectionRecommendations(status=UNAVAILABLE, reason="x")
    r = recommend_preprocessing(df, inv, trans, bad)
    assert r.status is UNAVAILABLE
    assert "feature-selection recommendations" in r.reason


def test_selection_not_yet_inferred(df, inv, trans):
    r = recommend_preprocessing(df, inv, trans, FeatureSelectionRecommendations())
    assert r.status is UNAVAILABLE


def test_completed_upstream_zero_recommendations():
    d = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]}
    )
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    r = recommend_preprocessing(d, inv, trans, sel)
    assert r.status is COMPLETED
    assert r.required_operations == []
    assert not (r.encoding_required or r.scaling_required or r.imputation_required)


def test_no_eligible_features_completed():
    d = pd.DataFrame({"row_id": [1, 2, 3, 4], "k": [9, 9, 9, 9]})
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    r = recommend_preprocessing(d, inv, trans, sel)
    assert r.status is COMPLETED
    assert "no retained or review feature columns" in r.reason


# --- encoding ------------------------------------------


def test_categorical_candidate_requires_encoding(pp):
    assert _OP_ENCODING in pp.required_operations
    assert pp.encoding_required is True
    assert [r.description for r in _req(pp, "city")] == [_OP_ENCODING]


def test_boolean_candidate_no_encoding(pp):
    assert _req(pp, "flag") == []


def test_numeric_candidate_no_encoding(pp):
    assert not any(r.description == _OP_ENCODING for r in _req(pp, "age"))


def test_datetime_candidate_no_generic_encoding(pp):
    assert not any(r.description == _OP_ENCODING for r in _req(pp, "when"))


def test_dropped_categorical_no_requirement():
    d = pd.DataFrame({"dup_a": ["x", "y"] * 20, "dup_b": ["x", "y"] * 20, "keep": range(40)})
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    # dup_b is an exact duplicate of dup_a -> dropped
    assert "dup_b" in sel.dropped_features
    r = recommend_preprocessing(d, inv, trans, sel)
    assert _req(r, "dup_b") == []


def test_high_cardinality_categorical_requirement_only():
    d = pd.DataFrame({"tag": [f"seg{i % 55}" for i in range(120)], "v": range(120)})
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    r = recommend_preprocessing(d, inv, trans, sel)
    tag_reqs = [x.description for x in _req(r, "tag")]
    assert tag_reqs == [_OP_ENCODING]  # no invented specialised encoder
    blob = r.model_dump_json().lower()
    assert "target encoding" not in blob


# --- scaling -------------------------------------------


def test_phase63_scaling_recommendation_requires_scaling(pp):
    assert _OP_SCALING in pp.required_operations
    assert pp.scaling_required is True
    assert [r.description for r in _req(pp, "big_scale")] == [_OP_SCALING]
    assert any("Phase 6.3" in e for e in _req(pp, "big_scale")[0].evidence)


def test_transformation_without_scaling_no_fabricated_scaling(df, inv, sel):
    # a strictly-positive large-range column gets a log rec, not a scaling rec
    d = df.copy()
    d["huge_pos"] = np.geomspace(1.0, 1e7, len(d))
    inv2 = inventory_features(d)
    trans2 = recommend_transformations(d, inv2)
    sel2 = recommend_feature_selection(d, inv2, _task())
    r = recommend_preprocessing(d, inv2, trans2, sel2)
    assert not any(x.description == _OP_SCALING for x in _req(r, "huge_pos"))


def test_categorical_boolean_datetime_no_scaling(pp):
    for col in ("city", "flag", "when"):
        assert not any(r.description == _OP_SCALING for r in _req(pp, col))


def test_dropped_numeric_no_scaling():
    d = pd.DataFrame(
        {
            "a": np.r_[np.full(40, 50000.0) + np.arange(40)],
            "a_dup": np.r_[np.full(40, 50000.0) + np.arange(40)],
            "keep": range(40),
        }
    )
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    assert "a_dup" in sel.dropped_features
    r = recommend_preprocessing(d, inv, trans, sel)
    assert _req(r, "a_dup") == []


# --- missing values -----------------------------------


def test_no_missing_no_imputation(pp):
    assert not any(r.description == _OP_IMPUTATION for r in _req(pp, "age"))


def test_moderate_missingness_requires_imputation(pp):
    assert _OP_IMPUTATION in pp.required_operations
    assert pp.imputation_required is True
    reqs = _req(pp, "age_missing")
    assert reqs[0].description == _OP_IMPUTATION
    assert any("missing value" in e for e in reqs[0].evidence)


def test_high_missingness_preserves_review_semantics(df):
    d = df.copy()
    d["scarce"] = [1.0, 2.0, 3.0, 4.0, 5.0] + [np.nan] * (len(d) - 5)  # ~92% missing
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    assert "scarce" in sel.review_features
    r = recommend_preprocessing(d, inv, trans, sel)
    reqs = _req(r, "scarce")
    assert reqs and reqs[0].description == _OP_IMPUTATION
    assert "review" in reqs[0].reason.lower()


def test_all_missing_excluded_no_imputation(df):
    d = df.copy()
    d["empty"] = np.nan
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    r = recommend_preprocessing(d, inv, trans, sel)
    assert _req(r, "empty") == []


def test_target_column_not_imputed(df):
    d = df.copy()
    d.loc[d.index[:5], "age"] = np.nan
    d = d.rename(columns={"age": "target"})
    d.loc[d.index[:5], "target"] = np.nan
    inv = inventory_features(d, target="target")
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(
        d,
        inv,
        TaskTypeInference(
            status=ProblemUnderstandingStatus.COMPLETED,
            task_type=TaskType.REGRESSION,
            target_column="target",
        ),
    )
    r = recommend_preprocessing(d, inv, trans, sel)
    assert _req(r, "target") == []


# --- interactions ------------------------------------


def test_phase63_recommendations_consumed_not_executed(pp, df):
    before = df.copy(deep=True)
    recommend_preprocessing(
        df,
        inventory_features(df),
        recommend_transformations(df, inventory_features(df)),
        recommend_feature_selection(df, inventory_features(df), _task()),
    )
    pd.testing.assert_frame_equal(df, before)
    assert any("upstream dependency" in n for n in pp.notes)


def test_phase64_review_respected(pp):
    # age_missing is retained; scarce-style review handled elsewhere; ensure review tag surfaces
    d_notes = " ".join(pp.notes)
    assert "requirements stage" in d_notes


def test_no_duplicated_operations(pp):
    seen = set()
    for r in pp.requirements:
        key = (r.column, r.description)
        assert key not in seen
        seen.add(key)


def test_upstream_results_unchanged(df, inv, trans, sel):
    snaps = (inv.model_dump_json(), trans.model_dump_json(), sel.model_dump_json())
    recommend_preprocessing(df, inv, trans, sel)
    assert (inv.model_dump_json(), trans.model_dump_json(), sel.model_dump_json()) == snaps


# --- objective ---------------------------------------


def test_objective_none(pp):
    assert pp.objective_used is False


def test_objective_blank(df, inv, trans, sel):
    assert recommend_preprocessing(df, inv, trans, sel, objective="").objective_used is False


def test_objective_whitespace(df, inv, trans, sel):
    assert recommend_preprocessing(df, inv, trans, sel, objective="   ").objective_used is False


def test_objective_recognised_wording(df, inv, trans, sel):
    r = recommend_preprocessing(
        df, inv, trans, sel, objective="handle missing values and scale features"
    )
    assert r.objective_used is True
    assert any("data preparation" in n for n in r.notes)


def test_objective_cannot_override_structural_rules(df, inv, trans, sel):
    r = recommend_preprocessing(df, inv, trans, sel, objective="use target encoding for everything")
    blob = r.model_dump_json().lower()
    assert "target encoding" not in blob
    assert _OP_ENCODING in r.required_operations  # plain encoding still required for city


# --- determinism -----------------------------------


def test_repeated_calls_identical(df, inv, trans, sel):
    blobs = {recommend_preprocessing(df, inv, trans, sel).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_row_shuffle_identical(df, inv, trans, sel):
    base = recommend_preprocessing(df, inv, trans, sel)
    shuffled = df.sample(frac=1.0, random_state=4)
    assert recommend_preprocessing(shuffled, inv, trans, sel) == base


def test_column_reorder_identical(df, inv, trans, sel):
    base = recommend_preprocessing(df, inv, trans, sel)
    reordered = df[list(df.columns)[::-1]]
    assert recommend_preprocessing(reordered, inv, trans, sel) == base


def test_required_operations_fixed_order(pp):
    order = [_OP_IMPUTATION, _OP_ENCODING, _OP_SCALING]
    positions = [order.index(op) for op in pp.required_operations]
    assert positions == sorted(positions)


def test_requirements_sorted_within_operation(pp):
    by_op: dict[str, list[str]] = {}
    for r in pp.requirements:
        by_op.setdefault(r.description, []).append(r.column)
    for cols in by_op.values():
        assert cols == sorted(cols)


def test_flags_agree_with_required_operations(pp):
    assert pp.encoding_required is (_OP_ENCODING in pp.required_operations)
    assert pp.scaling_required is (_OP_SCALING in pp.required_operations)
    assert pp.imputation_required is (_OP_IMPUTATION in pp.required_operations)


def test_stable_reasons(df, inv, trans, sel):
    a = recommend_preprocessing(df, inv, trans, sel)
    b = recommend_preprocessing(df.sample(frac=1.0, random_state=2), inv, trans, sel)
    assert [(r.column, r.description, r.reason) for r in a.requirements] == [
        (r.column, r.description, r.reason) for r in b.requirements
    ]


# --- safety ---------------------------------------


def test_df_unchanged(df, inv, trans, sel):
    before = df.copy(deep=True)
    recommend_preprocessing(df, inv, trans, sel, objective="scale features")
    recommend_preprocessing(df, inv, trans, sel)
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, inv, trans, sel, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recommend_preprocessing(df, inv, trans, sel)
    assert list(tmp_path.iterdir()) == []


# --- integration ---------------------------------


def test_merge_into_feature_engineering_spec(df):
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="ds", objective="prepare features")
    )
    inv = inventory_features(df, target=None)
    trans = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, _task())
    spec = spec.model_copy(update={"inventory": inv})
    spec = spec.model_copy(update={"transformations": trans})
    spec = spec.model_copy(update={"selection": sel})
    pre = recommend_preprocessing(df, inv, trans, sel, objective="prepare features")
    merged = spec.model_copy(update={"preprocessing": pre})

    assert merged.inventory == inv
    assert merged.transformations == trans
    assert merged.selection == sel
    assert merged.preprocessing.status is COMPLETED
    assert merged.assessment.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility -------------------


def test_understand_feature_engineering_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_feature_engineering).parameters) == ["request"]


def test_phase_6_1_to_6_4_still_work(df):
    inv = inventory_features(df, target="big_scale")
    assert inv.status is COMPLETED
    trans = recommend_transformations(df, inv)
    assert trans.status is COMPLETED
    sel = recommend_feature_selection(df, inv, _task())
    assert sel.status is COMPLETED


def test_legacy_preprocessing_json_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "required_operations": [],
            "encoding_required": False,
            "scaling_required": False,
            "imputation_required": False,
            "notes": [],
        }
    )
    model = PreprocessingRequirements.model_validate_json(legacy)
    assert model.requirements == []
    assert model.objective_used is False


def test_phase_5_apis_still_work(df):
    from data_engine.problem_understanding import identify_target, infer_task_type

    d = df.rename(columns={"big_scale": "price"})
    t = identify_target(d, objective="predict price")
    assert infer_task_type(d, t, objective="predict price").status.value in {
        "completed",
        "unavailable",
    }
