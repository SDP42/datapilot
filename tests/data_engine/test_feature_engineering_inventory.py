"""Phase 6.2 — deterministic structural feature inventory."""

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
    inventory_features,
    understand_feature_engineering,
)
from datapilot.contracts import ColumnType

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED
_N = 120


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.uniform(10.0, 500.0, _N),
            "score": rng.integers(0, 100, _N),
            "active": ([True, False] * (_N // 2)),
            "region": (["north", "south", "east", "west"] * (_N // 4)),
            "signup_date": pd.date_range("2021-01-01", periods=_N, freq="D"),
            "customer_id": [f"C{i:05d}" for i in range(_N)],
            "constant_col": ["x"] * _N,
            "empty_col": [np.nan] * _N,
        }
    )


def _by_col(inv: FeatureInventory) -> dict[str, FeatureInventoryCandidate]:
    return {c.column: c for c in inv.candidates}


# --- API -----------------------------------------------------------------


def test_public_import():
    assert fe.inventory_features is inventory_features


def test_public_model_import():
    assert fe.FeatureInventory is FeatureInventory
    assert fe.FeatureInventoryCandidate is FeatureInventoryCandidate


def test_return_type(df):
    assert isinstance(inventory_features(df), FeatureInventory)


def test_enum_reuse(df):
    inv = inventory_features(df)
    assert all(isinstance(c.column_type, ColumnType) for c in inv.candidates)


def test_json_serialisation(df):
    assert isinstance(json.loads(inventory_features(df, target="price").model_dump_json()), dict)


def test_json_round_trip(df):
    inv = inventory_features(df, target="price", objective="predict price")
    assert FeatureInventory.model_validate_json(inv.model_dump_json()) == inv


def test_json_primitive_only(df):
    payload = json.loads(inventory_features(df, target="price").model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


# --- structural inventory --------------------------------------------


def test_numeric_candidate(df):
    assert _by_col(inventory_features(df))["price"].candidate is True


def test_categorical_candidate(df):
    assert _by_col(inventory_features(df))["region"].candidate is True


def test_boolean_candidate(df):
    c = _by_col(inventory_features(df))["active"]
    assert c.candidate is True
    assert c.column_type is ColumnType.BOOLEAN


def test_datetime_candidate(df):
    c = _by_col(inventory_features(df))["signup_date"]
    assert c.candidate is True
    assert c.column_type is ColumnType.DATETIME


def test_unknown_type_handled_conservatively():
    d = pd.DataFrame({"weird": pd.Series([None, None, None], dtype="object"), "x": [1, 2, 3]})
    inv = inventory_features(d)
    weird = _by_col(inv)["weird"]
    # all-missing object column: UNKNOWN type, excluded as entirely missing, explained
    assert weird.column_type is ColumnType.UNKNOWN
    assert weird.candidate is False
    assert any("entirely missing" in r for r in weird.reasons)


def test_constant_exclusion(df):
    c = _by_col(inventory_features(df))["constant_col"]
    assert c.candidate is False
    assert c.constant is True
    assert any("constant" in r for r in c.reasons)
    assert "constant_col" in inventory_features(df).excluded_features


def test_all_missing_exclusion(df):
    c = _by_col(inventory_features(df))["empty_col"]
    assert c.candidate is False
    assert c.all_missing is True
    assert any("entirely missing" in r for r in c.reasons)


def test_identifier_like_exclusion_by_name(df):
    c = _by_col(inventory_features(df))["customer_id"]
    assert c.identifier_like is True
    assert c.candidate is False
    assert any("identifier" in r for r in c.reasons)


def test_near_unique_integer_identifier():
    d = pd.DataFrame({"row_number": range(200), "v": np.random.default_rng(1).normal(size=200)})
    c = _by_col(inventory_features(d))["row_number"]
    assert c.identifier_like is True
    assert c.candidate is False


def test_near_unique_categorical_identifier():
    d = pd.DataFrame({"code": [f"K{i}" for i in range(150)], "v": [1] * 75 + [2] * 75})
    c = _by_col(inventory_features(d))["code"]
    assert c.identifier_like is True
    assert c.candidate is False


def test_high_uniqueness_float_remains_eligible():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"measurement": rng.uniform(0, 1, 300), "grp": ["a", "b"] * 150})
    c = _by_col(inventory_features(d))["measurement"]
    assert c.unique_fraction >= 0.99
    assert c.identifier_like is False
    assert c.candidate is True


def test_declared_target_excluded(df):
    inv = inventory_features(df, target="price")
    assert "price" not in inv.candidate_features
    assert "price" in inv.excluded_features
    tc = _by_col(inv)["price"]
    assert tc.is_target is True
    assert tc.candidate is False
    assert any("prediction target" in r for r in tc.reasons)


def test_target_none_does_not_invent_target(df):
    inv = inventory_features(df, target=None)
    assert not any(c.is_target for c in inv.candidates)
    assert "price" in inv.candidate_features


# --- missingness -------------------------------------------------


def test_zero_missing(df):
    assert _by_col(inventory_features(df))["price"].n_missing == 0


def test_moderate_missingness_remains_candidate(df):
    d = df.copy()
    d.loc[d.index[:40], "price"] = np.nan  # 1/3 missing
    c = _by_col(inventory_features(d))["price"]
    assert c.candidate is True
    assert c.n_missing == 40
    assert any("missing" in r for r in c.reasons)


def test_all_missing_excluded_via_core_rule(df):
    assert _by_col(inventory_features(df))["empty_col"].candidate is False


def test_deterministic_missing_fraction():
    d = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": [1, 2, 3, 4]})
    assert _by_col(inventory_features(d))["a"].missing_fraction == 0.5


# --- determinism ------------------------------------------------


def test_repeated_calls_identical(df):
    a = inventory_features(df, target="price").model_dump_json()
    b = inventory_features(df, target="price").model_dump_json()
    assert a == b


def test_row_shuffle_identical(df):
    base = inventory_features(df, target="price")
    shuffled = df.sample(frac=1.0, random_state=9)
    assert inventory_features(shuffled, target="price") == base


def test_column_reorder_identical(df):
    base = inventory_features(df, target="price")
    reordered = df[list(df.columns)[::-1]]
    assert inventory_features(reordered, target="price") == base


def test_alphabetical_ordering(df):
    inv = inventory_features(df)
    cols = [c.column for c in inv.candidates]
    assert cols == sorted(cols)
    assert inv.candidate_features == sorted(inv.candidate_features)
    assert inv.excluded_features == sorted(inv.excluded_features)


def test_deterministic_reasons(df):
    a = inventory_features(df)
    b = inventory_features(df.sample(frac=1.0, random_state=3))
    assert [c.reasons for c in a.candidates] == [c.reasons for c in b.candidates]


# --- validation ----------------------------------------------


def test_non_dataframe_raises_type_error():
    with pytest.raises(TypeError):
        inventory_features({"a": [1, 2]})


def test_missing_target_unavailable(df):
    inv = inventory_features(df, target="not_a_column")
    assert inv.status is UNAVAILABLE
    assert "not in the DataFrame" in inv.reason
    assert inv.candidates == []


def test_zero_column_dataframe():
    inv = inventory_features(pd.DataFrame())
    assert inv.status is UNAVAILABLE
    assert "no columns" in inv.reason


def test_zero_row_dataframe():
    inv = inventory_features(
        pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
    )
    assert inv.status is UNAVAILABLE
    assert "no rows" in inv.reason


# --- safety ------------------------------------------------


def test_dataframe_not_mutated(df):
    before = df.copy(deep=True)
    inventory_features(df, target="price", objective="predict price")
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inventory_features(df, target="price")
    assert list(tmp_path.iterdir()) == []


def test_objective_not_used(df):
    inv = inventory_features(df, target="price", objective="only keep the region column")
    assert inv.objective_used is False
    assert "region" in inv.candidate_features
    assert "score" in inv.candidate_features


# --- integration ------------------------------------------


def test_merge_into_feature_engineering_spec(df):
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="ds", objective="predict price")
    )
    inv = inventory_features(df, target="price", objective="predict price")
    merged = spec.model_copy(update={"inventory": inv})

    assert merged.inventory.status is COMPLETED
    assert merged.transformations.status is NOT_YET
    assert merged.selection.status is NOT_YET
    assert merged.preprocessing.status is NOT_YET
    assert merged.assessment.status is NOT_YET
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


def test_merge_unavailable_inventory(df):
    spec = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="ds"))
    inv = inventory_features(df, target="missing")
    merged = spec.model_copy(update={"inventory": inv})
    assert merged.inventory.status is UNAVAILABLE
    assert merged.status is NOT_YET


# --- backward compatibility -----------------------------


def test_phase_6_1_foundation_unchanged():
    spec = understand_feature_engineering(FeatureEngineeringRequest(dataset_id="ds"))
    assert spec.status is NOT_YET
    assert spec.inventory.status is NOT_YET
    assert spec.inventory.candidates == []


def test_legacy_feature_inventory_json_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "candidate_features": [],
            "excluded_features": [],
            "notes": [],
        }
    )
    model = FeatureInventory.model_validate_json(legacy)
    assert model.candidates == []
    assert model.objective_used is False


def test_phase_5_apis_unchanged(df):
    from data_engine.problem_understanding import (
        assess_feasibility,
        identify_target,
        infer_task_type,
        recommend_metrics,
    )

    target = identify_target(df, objective="predict price")
    assert target.target_column == "price"
    task = infer_task_type(df, target, objective="predict price")
    metrics = recommend_metrics(df, task, objective="predict price")
    feas = assess_feasibility(df, target, task, metrics)
    assert feas.status.value == "completed"
