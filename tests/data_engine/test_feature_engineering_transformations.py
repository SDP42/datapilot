"""Phase 6.3 — deterministic transformation recommendations."""

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
    TransformationRecommendation,
    TransformationRecommendations,
    inventory_features,
    recommend_transformations,
    understand_feature_engineering,
)

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED


@pytest.fixture
def df() -> pd.DataFrame:
    n = 100
    return pd.DataFrame(
        {
            "amount": [1.0] * 90 + [1000.0] * 10,  # strictly positive, strong skew + big range
            "with_zero": [0.0] * 90 + [500.0] * 10,  # non-neg w/ zeros, strong skew -> log1p
            "pos_mild": [1.0] * 40 + [2.0] * 30 + [3.0] * 20 + [6.0] * 10,  # moderate skew -> sqrt
            "delta": ([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] * 15)[:n],  # both signs about 0
            "neg_skew": [-1.0] * 90 + [-1000.0] * 10,  # strictly negative, strong skew
            "steady": list(np.linspace(0.1, 0.9, n)),  # ~uniform, no transform
            "signup_date": pd.date_range("2021-01-01", periods=n, freq="D"),
            "region": (["north", "south", "east", "west"] * 25),
            "flag": ([True, False] * 50),
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "const_col": [7.0] * n,
        }
    )


@pytest.fixture
def inv(df) -> FeatureInventory:
    return inventory_features(df, target=None)


def _ops(result: TransformationRecommendations) -> list[str]:
    return result.recommended_operations


def _for(result: TransformationRecommendations, column: str) -> list[TransformationRecommendation]:
    return [r for r in result.recommendations if r.column == column]


# --- API ------------------------------------------------------------------


def test_public_import():
    assert fe.recommend_transformations is recommend_transformations


def test_return_type(df, inv):
    assert isinstance(recommend_transformations(df, inv), TransformationRecommendations)


def test_structured_recommendation_model(df, inv):
    r = recommend_transformations(df, inv)
    assert all(isinstance(x, TransformationRecommendation) for x in r.recommendations)


def test_enum_reuse(df, inv):
    r = recommend_transformations(df, inv)
    allowed = {
        FeatureOperationType.TRANSFORMATION,
        FeatureOperationType.DATETIME_DERIVATION,
        FeatureOperationType.NUMERICAL_SCALING,
    }
    assert all(x.operation in allowed for x in r.recommendations)


def test_json_round_trip(df, inv):
    r = recommend_transformations(df, inv, objective="reduce skew")
    assert TransformationRecommendations.model_validate_json(r.model_dump_json()) == r


def test_json_primitive_only(df, inv):
    payload = json.loads(recommend_transformations(df, inv).model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


def test_exported_constants():
    assert fe.TRANSFORMATION_SKEW_THRESHOLD == 1.0
    assert fe.TRANSFORMATION_STRONG_SKEW_THRESHOLD == 2.0
    assert fe.TRANSFORMATION_LOG_RANGE_RATIO == 1000.0


# --- numeric transformations --------------------------------------


def test_positive_large_range_gets_log(df, inv):
    recs = _for(recommend_transformations(df, inv), "amount")
    assert [r.description for r in recs] == ["log transform"]
    assert recs[0].operation is FeatureOperationType.TRANSFORMATION


def test_zero_containing_numeric_no_plain_log(df, inv):
    recs = _for(recommend_transformations(df, inv), "with_zero")
    descs = [r.description for r in recs]
    assert "log transform" not in descs
    assert "log1p transform" in descs
    assert any("plain log transform is not applicable" in e for e in recs[0].evidence)


def test_negative_numeric_no_plain_log(df, inv):
    recs = _for(recommend_transformations(df, inv), "neg_skew")
    descs = [r.description for r in recs]
    assert "log transform" not in descs
    assert "log1p transform" not in descs
    assert descs == ["reciprocal transform"]


def test_non_negative_moderate_skew_gets_sqrt(df, inv):
    recs = _for(recommend_transformations(df, inv), "pos_mild")
    assert [r.description for r in recs] == ["square-root transform"]


def test_reciprocal_domain_protection(df, inv):
    # with_zero contains 0.0 -> reciprocal must never be recommended
    assert "reciprocal transform" not in [
        r.description for r in _for(recommend_transformations(df, inv), "with_zero")
    ]


def test_reciprocal_only_for_strictly_negative_no_zero(df, inv):
    recs = _for(recommend_transformations(df, inv), "neg_skew")
    assert recs[0].description == "reciprocal transform"
    assert any("no zero values" in e for e in recs[0].evidence)


def test_deterministic_skew_threshold_respected():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(
        {
            "near_symmetric": np.abs(rng.normal(100.0, 10.0, 200)),  # |skew| < 1 -> nothing
            "moderate_skew": [1.0] * 80 + [2.0] * 60 + [3.0] * 40 + [6.0] * 20,  # skew in [1,2)
            "g": ["a", "b"] * 100,
        }
    )
    r = recommend_transformations(d, inventory_features(d))
    assert _for(r, "near_symmetric") == []
    assert [x.description for x in _for(r, "moderate_skew")] == ["square-root transform"]
    # deterministic across repeated calls
    assert recommend_transformations(d, inventory_features(d)) == r


def test_no_fabricated_transform_for_ordinary_numeric(df, inv):
    assert _for(recommend_transformations(df, inv), "steady") == []


def test_absolute_value_for_signed_centered_feature(df, inv):
    recs = _for(recommend_transformations(df, inv), "delta")
    assert "absolute-value transform" in [r.description for r in recs]


# --- datetime ----------------------------------------------------


def test_datetime_derivation_recommended(df, inv):
    recs = _for(recommend_transformations(df, inv), "signup_date")
    descs = {r.description for r in recs}
    assert {"derive year", "derive month", "derive day_of_week", "derive quarter"} <= descs
    assert all(r.operation is FeatureOperationType.DATETIME_DERIVATION for r in recs)


def test_datetime_no_hour_for_date_only(df, inv):
    descs = {r.description for r in _for(recommend_transformations(df, inv), "signup_date")}
    assert "derive hour" not in descs


def test_datetime_hour_when_time_of_day_present():
    d = pd.DataFrame({"ts": pd.date_range("2021-01-01", periods=60, freq="7h"), "v": range(60)})
    descs = {r.description for r in _for(recommend_transformations(d, inventory_features(d)), "ts")}
    assert "derive hour" in descs
    assert "cyclical (sin/cos) hour" in descs


def test_all_missing_datetime_no_recommendation():
    d = pd.DataFrame({"ts": pd.Series([pd.NaT] * 20, dtype="datetime64[ns]"), "v": range(20)})
    r = recommend_transformations(d, inventory_features(d))
    assert _for(r, "ts") == []  # excluded by inventory as all-missing


def test_datetime_does_not_infer_forecasting(df, inv):
    r = recommend_transformations(df, inv)
    blob = r.model_dump_json().lower()
    assert "forecast" not in blob
    assert "time_series" not in blob


def test_cyclical_recommended(df, inv):
    descs = {r.description for r in _for(recommend_transformations(df, inv), "signup_date")}
    assert "cyclical (sin/cos) month" in descs
    assert "cyclical (sin/cos) day_of_week" in descs


# --- categorical / boolean ------------------------------------


def test_categorical_not_encoded(df, inv):
    r = recommend_transformations(df, inv)
    assert _for(r, "region") == []
    assert not any("encod" in op.lower() for op in r.recommended_operations)
    assert any("categorical encoding is deferred" in n for n in r.notes)


def test_boolean_no_transform(df, inv):
    r = recommend_transformations(df, inv)
    assert _for(r, "flag") == []
    assert any("boolean candidate" in n for n in r.notes)


def test_identifier_like_receives_no_transform(df, inv):
    assert _for(recommend_transformations(df, inv), "customer_id") == []


def test_high_cardinality_categorical_inventory_driven():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    d = pd.DataFrame(
        {"tag": [words[i % len(words)] + str(i % 40) for i in range(200)], "v": range(200)}
    )
    r = recommend_transformations(d, inventory_features(d))
    assert _for(r, "tag") == []


# --- inventory interaction -----------------------------------


def test_inventory_unavailable_is_unavailable(df):
    bad = inventory_features(df, target="does_not_exist")
    r = recommend_transformations(df, bad)
    assert r.status is UNAVAILABLE
    assert "feature inventory" in r.reason
    assert r.recommendations == []


def test_not_yet_inferred_inventory_is_unavailable(df):
    r = recommend_transformations(df, FeatureInventory())
    assert r.status is UNAVAILABLE


def test_excluded_feature_not_recommended(df, inv):
    r = recommend_transformations(df, inv)
    assert _for(r, "const_col") == []
    assert _for(r, "customer_id") == []


def test_declared_target_excluded_not_recommended(df):
    inv = inventory_features(df, target="amount")
    r = recommend_transformations(df, inv)
    assert _for(r, "amount") == []


def test_only_candidate_features_considered(df, inv):
    r = recommend_transformations(df, inv)
    rec_columns = {x.column for x in r.recommendations}
    assert rec_columns <= set(inv.candidate_features)


def test_zero_candidates_completed_empty():
    d = pd.DataFrame({"row_id": [1, 2, 3, 4], "c": [5, 5, 5, 5]})
    r = recommend_transformations(d, inventory_features(d))
    assert r.status is COMPLETED
    assert r.recommendations == []
    assert r.recommended_operations == []
    assert "no structurally eligible feature columns" in r.reason


# --- objective ---------------------------------------------


def test_objective_absent(df, inv):
    r = recommend_transformations(df, inv)
    assert r.objective_used is False


def test_blank_objective(df, inv):
    r = recommend_transformations(df, inv, objective="   ")
    assert r.objective_used is False


def test_objective_normalisation_matches_vocabulary(df, inv):
    r = recommend_transformations(df, inv, objective="Please REDUCE-SKEW here")
    assert r.objective_used is True
    assert any("refinement vocabulary" in n for n in r.notes)


def test_objective_refines_priority_moderate_to_log(df, inv):
    base = _for(recommend_transformations(df, inv), "pos_mild")
    refined = _for(recommend_transformations(df, inv, objective="reduce skew"), "pos_mild")
    assert [r.description for r in base] == ["square-root transform"]
    assert [r.description for r in refined] == ["log transform"]


def test_objective_never_overrides_math_validity(df, inv):
    r = recommend_transformations(df, inv, objective="log transform everything to reduce skew")
    assert "log transform" not in [x.description for x in _for(r, "with_zero")]
    assert "log transform" not in [x.description for x in _for(r, "neg_skew")]


# --- missingness -----------------------------------------


def test_moderate_missingness_still_recommended(df, inv):
    d = df.copy()
    d.loc[d.index[:20], "amount"] = np.nan
    recs = _for(recommend_transformations(d, inventory_features(d)), "amount")
    assert recs and recs[0].description == "log transform"
    assert any("non-missing values" in e for e in recs[0].evidence)


def test_no_imputation_recommendation(df, inv):
    r = recommend_transformations(df, inv)
    assert not any(
        x.operation is FeatureOperationType.MISSING_VALUE_HANDLING for x in r.recommendations
    )
    blob = r.model_dump_json().lower()
    assert "impute" not in blob or "deferred to phase 6.5" in blob


def test_missingness_note_present(df):
    d = df.copy()
    d.loc[d.index[:15], "amount"] = np.nan
    r = recommend_transformations(d, inventory_features(d))
    assert any("Phase 6.5" in n for n in r.notes)


def test_all_missing_column_excluded(df, inv):
    d = df.copy()
    d["amount"] = np.nan
    r = recommend_transformations(d, inventory_features(d))
    assert _for(r, "amount") == []


# --- determinism -----------------------------------------


def test_repeated_calls_identical(df, inv):
    blobs = {recommend_transformations(df, inv).model_dump_json() for _ in range(5)}
    assert len(blobs) == 1


def test_row_shuffle_identical(df, inv):
    base = recommend_transformations(df, inv)
    shuffled = df.sample(frac=1.0, random_state=11)
    assert recommend_transformations(shuffled, inv) == base


def test_column_reorder_identical(df, inv):
    base = recommend_transformations(df, inv)
    reordered = df[list(df.columns)[::-1]]
    assert recommend_transformations(reordered, inv) == base


def test_recommendation_ordering_deterministic(df, inv):
    recs = recommend_transformations(df, inv).recommendations
    keys = [(r.column, r.description) for r in recs]
    assert keys == sorted(keys, key=lambda k: (k[0], k[1])) or all(
        recs[i].column <= recs[i + 1].column for i in range(len(recs) - 1)
    )


def test_recommended_operations_aligned(df, inv):
    r = recommend_transformations(df, inv)
    assert r.recommended_operations == [f"{x.column}: {x.description}" for x in r.recommendations]


# --- safety --------------------------------------------


def test_dataframe_not_mutated(df, inv):
    before = df.copy(deep=True)
    recommend_transformations(df, inv, objective="reduce skew")
    recommend_transformations(df, inv)
    pd.testing.assert_frame_equal(df, before)


def test_inventory_not_mutated(df, inv):
    snapshot = inv.model_dump_json()
    recommend_transformations(df, inv)
    assert inv.model_dump_json() == snapshot


def test_no_files_created(df, inv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recommend_transformations(df, inv)
    assert list(tmp_path.iterdir()) == []


def test_non_dataframe_raises_type_error(inv):
    with pytest.raises(TypeError):
        recommend_transformations({"a": [1]}, inv)


def test_non_inventory_raises_type_error(df):
    with pytest.raises(TypeError):
        recommend_transformations(df, {"status": "completed"})


# --- integration -----------------------------------


def test_merge_into_feature_engineering_spec(df):
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="ds", objective="reduce skew")
    )
    inv = inventory_features(df, target="amount")
    spec = spec.model_copy(update={"inventory": inv})
    tr = recommend_transformations(df, inv, objective="reduce skew")
    merged = spec.model_copy(update={"transformations": tr})

    assert merged.inventory.status is COMPLETED
    assert merged.transformations.status is COMPLETED
    assert merged.selection.status is NOT_YET
    assert merged.preprocessing.status is NOT_YET
    assert merged.assessment.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged
    assert merged.inventory == inv  # inventory unchanged by the merge


# --- backward compatibility ------------------------


def test_understand_feature_engineering_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_feature_engineering).parameters) == ["request"]


def test_inventory_features_still_works(df):
    assert inventory_features(df, target="amount").status is COMPLETED


def test_legacy_transformation_json_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "recommended_operations": [],
            "notes": [],
        }
    )
    model = TransformationRecommendations.model_validate_json(legacy)
    assert model.recommendations == []
    assert model.objective_used is False


def test_phase_5_apis_still_importable(df):
    from data_engine.problem_understanding import (
        assess_feasibility,
        identify_target,
        infer_task_type,
        recommend_metrics,
    )

    t = identify_target(df, objective="predict amount")
    task = infer_task_type(df, t, objective="predict amount")
    m = recommend_metrics(df, task, objective="predict amount")
    assert assess_feasibility(df, t, task, m).status.value in {"completed", "unavailable"}
