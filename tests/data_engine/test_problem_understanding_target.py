"""Phase 5.2 — deterministic target identification."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import data_engine.problem_understanding as pu
from data_engine.problem_understanding import (
    TARGET_SELECTION_MARGIN,
    ColumnType,
    ObjectiveMatchKind,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetCandidate,
    TargetIdentification,
    identify_target,
    understand_problem,
)

_N = 180


@pytest.fixture
def df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": range(_N),
            "age": rng.integers(18, 80, _N),
            "region": (["north", "south", "east", "west"] * (_N // 4)),
            "signup_date": pd.date_range("2021-01-01", periods=_N, freq="D"),
            "is_active": ([True, False] * (_N // 2)),
            "price": rng.normal(100.0, 15.0, _N),
        }
    )


def _cand(result: TargetIdentification, column: str) -> TargetCandidate:
    return next(c for c in result.candidates if c.column == column)


# --- API ---------------------------------------------------------


def test_public_symbols_importable():
    assert pu.identify_target is identify_target
    for name in ("TargetCandidate", "ObjectiveMatchKind", "TARGET_SELECTION_MARGIN"):
        assert hasattr(pu, name)


def test_return_type(df):
    assert isinstance(identify_target(df), TargetIdentification)


def test_json_round_trip(df):
    result = identify_target(df, objective="predict house price")
    assert TargetIdentification.model_validate_json(result.model_dump_json()) == result


def test_model_holds_only_json_primitives(df):
    payload = identify_target(df, objective="predict price").model_dump(mode="json")

    def prim(o: object) -> bool:
        if isinstance(o, dict):
            return all(prim(v) for v in o.values())
        if isinstance(o, list):
            return all(prim(v) for v in o)
        return o is None or isinstance(o, (str, int, float, bool))

    assert prim(payload)


def test_score_is_not_a_probability(df):
    # a documented ranking score can legitimately exceed 100 / go negative
    result = identify_target(df, objective="predict region and price")
    assert any(c.score > 100 for c in result.candidates)
    result2 = identify_target(pd.DataFrame({"id": range(50), "x": range(50)}))
    assert any(c.score < 0 for c in result2.candidates)


# --- basic candidate detection ---------------------------------


def test_numeric_categorical_boolean_datetime_are_all_eligible(df):
    result = identify_target(df)
    kinds = {c.column: c.column_type for c in result.candidates}
    assert kinds["price"] is ColumnType.NUMERIC
    assert kinds["region"] is ColumnType.CATEGORICAL
    assert kinds["is_active"] is ColumnType.BOOLEAN
    assert kinds["signup_date"] is ColumnType.DATETIME
    assert kinds["age"] is ColumnType.NUMERIC


def test_datetime_column_is_not_auto_rejected(df):
    result = identify_target(df, objective="forecast the signup date")
    assert result.target_column == "signup_date"


def test_candidates_are_ranked_best_first_and_sequential(df):
    result = identify_target(df, objective="predict price")
    ranks = [c.rank for c in result.candidates]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.candidate_columns == [c.column for c in result.candidates]


# --- objective-driven identification ---------------------------


def test_exact_objective_name_match_selects_target(df):
    result = identify_target(df, objective="predict house price")
    assert result.status is ProblemUnderstandingStatus.COMPLETED
    assert result.target_column == "price"
    assert result.reason is None
    assert _cand(result, "price").objective_match is ObjectiveMatchKind.EXACT
    assert result.objective_used is True


def test_normalized_name_match(df):
    frame = df.rename(columns={"price": "sale_price"})
    result = identify_target(frame, objective="predict the saleprice")
    assert result.target_column == "sale_price"
    assert _cand(result, "sale_price").objective_match in {
        ObjectiveMatchKind.EXACT,
        ObjectiveMatchKind.NORMALIZED,
    }


def test_token_match_with_shared_prefix(df):
    frame = df.rename(columns={"is_active": "churned"})
    result = identify_target(frame, objective="classify whether a customer will churn")
    assert result.target_column == "churned"
    assert _cand(result, "churned").objective_match is ObjectiveMatchKind.TOKEN


def test_ambiguous_objective_returns_candidates_not_a_guess(df):
    result = identify_target(df, objective="predict region and price")
    assert result.status is ProblemUnderstandingStatus.COMPLETED
    assert result.target_column is None
    assert "matches 2 columns" in result.reason
    assert {c.column for c in result.candidates[:2]} == {"region", "price"}


def test_objective_absent_is_conservative(df):
    result = identify_target(df)
    assert result.objective_used is False
    assert result.target_column is None  # several similarly-plausible columns
    assert "objective" in result.reason
    assert all(c.objective_match is ObjectiveMatchKind.NONE for c in result.candidates)


def test_objective_string_is_never_stored_or_altered(df):
    request = ProblemUnderstandingRequest(dataset_id="ds", objective="predict PRICE now")
    spec = understand_problem(request)
    merged = spec.model_copy(update={"target": identify_target(df, objective=request.objective)})
    assert merged.objective == "predict PRICE now"  # verbatim, unchanged


# --- exclusions / penalties -----------------------------------


def test_constant_and_all_missing_columns_are_excluded(df):
    frame = df.assign(const_col=1.0, empty_col=np.nan)
    result = identify_target(frame)
    assert "const_col" not in result.candidate_columns
    assert "empty_col" not in result.candidate_columns
    assert any("const_col" in n and "constant" in n for n in result.notes)
    assert any("empty_col" in n and "missing" in n for n in result.notes)


def test_identifier_like_columns_are_penalised(df):
    result = identify_target(df)
    cid = _cand(result, "customer_id")
    assert cid.identifier_like is True
    assert cid.rank == len(result.candidates)  # ranked last
    assert cid.score < _cand(result, "price").score


def test_high_uniqueness_float_column_is_not_flagged_as_identifier():
    frame = pd.DataFrame({"measurement": np.linspace(0.0, 1.0, 100), "grp": (["a", "b"] * 50)})
    result = identify_target(frame)
    assert _cand(result, "measurement").identifier_like is False


def test_high_missingness_column_is_deterministically_penalised(df):
    frame = df.copy()
    frame.loc[: _N * 3 // 4, "price"] = np.nan  # ~75% missing
    result = identify_target(frame)
    assert _cand(result, "price").missing_fraction > 0.5
    assert any("mostly missing" in r for r in _cand(result, "price").reasons)
    assert _cand(result, "price").score < _cand(result, "region").score


def test_empty_dataframe_is_unavailable():
    result = identify_target(pd.DataFrame())
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "no columns" in result.reason
    assert result.candidates == []


def test_dataframe_with_no_rows_is_unavailable():
    result = identify_target(pd.DataFrame(columns=["a", "b"]))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "no rows" in result.reason


def test_all_degenerate_columns_is_unavailable():
    result = identify_target(pd.DataFrame({"a": [1, 1, 1], "b": [np.nan, np.nan, np.nan]}))
    assert result.status is ProblemUnderstandingStatus.UNAVAILABLE
    assert "no plausible target" in result.reason


def test_missing_dataset_raises_type_error():
    with pytest.raises(TypeError):
        identify_target(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        identify_target([1, 2, 3])  # type: ignore[arg-type]


# --- determinism / tie-breaking -----------------------------


def test_repeated_calls_produce_identical_json(df):
    dumps = {identify_target(df, objective="predict price").model_dump_json() for _ in range(5)}
    assert len(dumps) == 1


def test_row_shuffle_does_not_change_the_result(df):
    base = identify_target(df, objective="predict price")
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert identify_target(shuffled, objective="predict price").model_dump() == base.model_dump()


def test_column_order_does_not_change_ranking(df):
    base = identify_target(df, objective="predict price")
    reordered = df[list(df.columns)[::-1]]
    result = identify_target(reordered, objective="predict price")
    assert result.candidate_columns == base.candidate_columns
    assert result.target_column == base.target_column


def test_tie_break_is_alphabetical_column_name():
    frame = pd.DataFrame(
        {"zulu": [1, 2, 3, 4, 5] * 12, "alpha": [1, 2, 3, 4, 5] * 12, "mike": [1, 2, 3, 4, 5] * 12}
    )
    result = identify_target(frame)
    scores = {c.score for c in result.candidates}
    assert len(scores) == 1  # identical shape -> identical score
    assert result.candidate_columns == ["alpha", "mike", "zulu"]


def test_selection_margin_constant_is_exposed_and_used():
    assert TARGET_SELECTION_MARGIN == 20.0
    # top leads by well over the margin -> pinned
    frame = pd.DataFrame(
        {"id": range(100), "grade": (["a", "b", "c", "d"] * 25), "notes_id": range(100)}
    )
    result = identify_target(frame)
    assert result.target_column == "grade"


# --- safety -------------------------------------------------


def test_input_dataframe_is_not_mutated(df):
    before = df.copy(deep=True)
    identify_target(df, objective="predict price")
    identify_target(df)
    pd.testing.assert_frame_equal(df, before)


def test_no_files_created(df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    identify_target(df, objective="predict price")
    assert list(tmp_path.iterdir()) == []


def test_no_figure_created(df):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    identify_target(df, objective="predict price")
    assert plt.get_fignums() == before


# --- ProblemSpec integration -----------------------------


def test_merge_into_problem_spec_leaves_other_sections_untouched(df):
    request = ProblemUnderstandingRequest(dataset_id="sales", objective="predict house price")
    spec = understand_problem(request)
    merged = spec.model_copy(update={"target": identify_target(df, objective=request.objective)})
    assert merged.target.status is ProblemUnderstandingStatus.COMPLETED
    assert merged.target.target_column == "price"
    assert merged.task_type.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert merged.metrics.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert merged.feasibility.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert merged.status is ProblemUnderstandingStatus.NOT_YET_INFERRED  # overall unchanged
    # and the whole thing still round-trips
    from data_engine.problem_understanding import ProblemSpec

    assert ProblemSpec.model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility -----------------------------


def test_phase_5_1_foundation_still_works():
    spec = understand_problem(ProblemUnderstandingRequest(dataset_id="ds"))
    assert spec.target.status is ProblemUnderstandingStatus.NOT_YET_INFERRED
    assert spec.target.candidates == []
    assert spec.target.objective_used is False


def test_old_target_identification_json_still_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "target_column": None,
            "candidate_columns": [],
            "notes": [],
        }
    )
    restored = TargetIdentification.model_validate_json(legacy)
    assert restored.candidates == []
    assert restored.objective_used is False


def test_understand_problem_signature_unchanged():
    import inspect

    assert list(inspect.signature(understand_problem).parameters) == ["request"]


def test_existing_engine_imports_unaffected():
    from data_engine.eda import analyze_dataframe
    from data_engine.quality import analyze_dataframe as q  # noqa: F401
    from data_engine.validation import DatasetVersion  # noqa: F401

    report = analyze_dataframe(pd.DataFrame({"x": [1.0, 2, 3], "y": [3.0, 2, 1]}))
    assert report.univariate.numeric
    assert not hasattr(report, "target")
