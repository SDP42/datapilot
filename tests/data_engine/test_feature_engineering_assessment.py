"""Phase 6.6 — deterministic feature-engineering assessment."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import data_engine.feature_engineering as fe
from data_engine.feature_engineering import (
    FeatureEngineeringAssessment,
    FeatureEngineeringCheck,
    FeatureEngineeringCheckOutcome,
    FeatureEngineeringRequest,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
    FeatureOperationType,
    FeatureSelectionAction,
    FeatureSelectionRecommendation,
    FeatureSelectionRecommendations,
    PreprocessingRequirement,
    PreprocessingRequirements,
    TransformationRecommendation,
    TransformationRecommendations,
    assess_feature_engineering,
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
from datapilot.contracts import ColumnType

COMPLETED = FeatureEngineeringStatus.COMPLETED
UNAVAILABLE = FeatureEngineeringStatus.UNAVAILABLE
NOT_YET = FeatureEngineeringStatus.NOT_YET_INFERRED


def _task() -> TaskTypeInference:
    return TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED,
        task_type=TaskType.REGRESSION,
        target_column="target",
    )


@pytest.fixture
def df() -> pd.DataFrame:
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "big": rng.uniform(40000.0, 60000.0, n),  # -> 6.3 numerical_scaling
            "age": ([25.0, 26.0, 27.0] * (n // 3 - 2)) + [np.nan] * 6,  # missing -> imputation
            "city": (["ny", "sf", "la", "chi"] * (n // 4)),  # categorical -> encoding
            "flag": ([True, False] * (n // 2)),
            "when": pd.date_range("2021-01-01", periods=n, freq="D"),
            "cust_id": [f"C{i:05d}" for i in range(n)],
            "const": [1.0] * n,
            "target": np.arange(n, dtype=float),
        }
    )


@pytest.fixture
def inv(df) -> FeatureInventory:
    return inventory_features(df, target="target")


@pytest.fixture
def trans(df, inv) -> TransformationRecommendations:
    return recommend_transformations(df, inv)


@pytest.fixture
def sel(df, inv) -> FeatureSelectionRecommendations:
    return recommend_feature_selection(df, inv, _task())


@pytest.fixture
def pre(df, inv, trans, sel) -> PreprocessingRequirements:
    return recommend_preprocessing(df, inv, trans, sel)


@pytest.fixture
def assessment(df, inv, trans, sel, pre) -> FeatureEngineeringAssessment:
    return assess_feature_engineering(df, inv, trans, sel, pre)


def _blocking_cats(a: FeatureEngineeringAssessment) -> list[str]:
    return [m.split("[", 1)[1].split("]", 1)[0] for m in a.blocking_issues]


# --- API --------------------------------------------------------------


def test_public_import():
    assert fe.assess_feature_engineering is assess_feature_engineering


def test_return_type(assessment):
    assert isinstance(assessment, FeatureEngineeringAssessment)


def test_exact_signature():
    params = list(inspect.signature(assess_feature_engineering).parameters)
    assert params == [
        "df",
        "inventory",
        "transformations",
        "selection",
        "preprocessing",
        "objective",
    ]


def test_structured_output(assessment):
    assert all(isinstance(c, FeatureEngineeringCheck) for c in assessment.checks)
    assert all(isinstance(c.outcome, FeatureEngineeringCheckOutcome) for c in assessment.checks)


def test_json_serialisation(assessment):
    assert isinstance(json.loads(assessment.model_dump_json()), dict)


def test_json_round_trip(df, inv, trans, sel, pre):
    a = assess_feature_engineering(df, inv, trans, sel, pre, objective="prepare features")
    assert FeatureEngineeringAssessment.model_validate_json(a.model_dump_json()) == a


def test_json_primitive_only(assessment):
    payload = json.loads(assessment.model_dump_json())

    def check(v):
        if isinstance(v, dict):
            [check(x) for x in v.values()]
        elif isinstance(v, list):
            [check(x) for x in v]
        else:
            assert v is None or isinstance(v, (str, int, float, bool))

    check(payload)


@pytest.mark.parametrize("bad_index", range(5))
def test_type_guards(df, inv, trans, sel, pre, bad_index):
    args = [df, inv, trans, sel, pre]
    args[bad_index] = object()
    with pytest.raises(TypeError):
        assess_feature_engineering(*args)


# --- upstream handling ----------------------------------------


def test_inventory_unavailable(df, trans, sel, pre):
    a = assess_feature_engineering(
        df, FeatureInventory(status=UNAVAILABLE, reason="x"), trans, sel, pre
    )
    assert a.status is UNAVAILABLE
    assert a.feasible is None
    assert "feature inventory" in a.reason
    assert a.blocking_issues == [] and a.warnings == []


def test_inventory_not_yet_inferred(df, trans, sel, pre):
    a = assess_feature_engineering(df, FeatureInventory(), trans, sel, pre)
    assert a.status is UNAVAILABLE and a.feasible is None


def test_transformations_unavailable(df, inv, sel, pre):
    a = assess_feature_engineering(
        df, inv, TransformationRecommendations(status=UNAVAILABLE, reason="x"), sel, pre
    )
    assert a.status is UNAVAILABLE
    assert "transformation recommendations" in a.reason


def test_transformations_not_yet_inferred(df, inv, sel, pre):
    a = assess_feature_engineering(df, inv, TransformationRecommendations(), sel, pre)
    assert a.status is UNAVAILABLE


def test_selection_unavailable(df, inv, trans, pre):
    a = assess_feature_engineering(
        df, inv, trans, FeatureSelectionRecommendations(status=UNAVAILABLE, reason="x"), pre
    )
    assert a.status is UNAVAILABLE
    assert "feature-selection recommendations" in a.reason


def test_selection_not_yet_inferred(df, inv, trans, pre):
    a = assess_feature_engineering(df, inv, trans, FeatureSelectionRecommendations(), pre)
    assert a.status is UNAVAILABLE


def test_preprocessing_unavailable(df, inv, trans, sel):
    a = assess_feature_engineering(
        df, inv, trans, sel, PreprocessingRequirements(status=UNAVAILABLE, reason="x")
    )
    assert a.status is UNAVAILABLE
    assert "preprocessing requirements" in a.reason


def test_preprocessing_not_yet_inferred(df, inv, trans, sel):
    a = assess_feature_engineering(df, inv, trans, sel, PreprocessingRequirements())
    assert a.status is UNAVAILABLE


def test_deterministic_upstream_precedence(df, trans, sel):
    a = assess_feature_engineering(
        df,
        FeatureInventory(status=UNAVAILABLE, reason="i"),
        TransformationRecommendations(status=UNAVAILABLE, reason="t"),
        FeatureSelectionRecommendations(status=UNAVAILABLE, reason="s"),
        PreprocessingRequirements(status=UNAVAILABLE, reason="p"),
    )
    assert "feature inventory" in a.reason  # inventory reported first


# --- valid completed pipeline -----------------------------


def test_coherent_pipeline_feasible(assessment):
    assert assessment.status is COMPLETED
    assert assessment.feasible is True
    assert assessment.blocking_issues == []
    assert assessment.reason is None


def test_no_op_pipeline_feasible():
    d = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]})
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    pre = recommend_preprocessing(d, inv, trans, sel)
    a = assess_feature_engineering(d, inv, trans, sel, pre)
    assert a.status is COMPLETED
    assert a.feasible is True


def test_warnings_do_not_flip_feasibility(assessment):
    assert assessment.warnings  # there are warnings in the fixture pipeline
    assert assessment.feasible is True


def test_no_execution_notes_present(assessment):
    joined = " ".join(assessment.notes)
    assert "no feature engineering was executed" in joined
    assert "no target or task-type inference was performed" in joined
    assert "no predictive evaluation" in joined


def test_pass_check_when_coherent(assessment):
    assert any(c.outcome is FeatureEngineeringCheckOutcome.PASS for c in assessment.checks)


# --- inventory consistency -------------------------------


def _cand(column, **kw) -> FeatureInventoryCandidate:
    d = {
        "column": column,
        "column_type": ColumnType.NUMERIC,
        "n_observations": 6,
        "n_missing": 0,
        "missing_fraction": 0.0,
        "n_unique": 6,
        "unique_fraction": 1.0,
        "identifier_like": False,
        "constant": False,
        "all_missing": False,
        "is_target": False,
        "candidate": True,
        "reasons": [],
    }
    d.update(kw)
    return FeatureInventoryCandidate(**d)


def _inv(*cands: FeatureInventoryCandidate, excluded=None) -> FeatureInventory:
    return FeatureInventory(
        status=COMPLETED,
        candidate_features=sorted(c.column for c in cands if c.candidate),
        excluded_features=excluded or [],
        candidates=list(cands),
    )


def _sel(selected=(), dropped=(), review=()) -> FeatureSelectionRecommendations:
    recs = (
        [
            FeatureSelectionRecommendation(
                column=c, action=FeatureSelectionAction.RETAIN, reason="k"
            )
            for c in selected
        ]
        + [
            FeatureSelectionRecommendation(column=c, action=FeatureSelectionAction.DROP, reason="d")
            for c in dropped
        ]
        + [
            FeatureSelectionRecommendation(
                column=c, action=FeatureSelectionAction.REVIEW, reason="r"
            )
            for c in review
        ]
    )
    return FeatureSelectionRecommendations(
        status=COMPLETED,
        selected_features=sorted(selected),
        dropped_features=sorted(dropped),
        review_features=sorted(review),
        recommendations=recs,
    )


def _trans(*recs: TransformationRecommendation) -> TransformationRecommendations:
    ordered = sorted(recs, key=lambda r: (r.column, r.description))
    return TransformationRecommendations(
        status=COMPLETED,
        recommended_operations=[f"{r.column}: {r.description}" for r in ordered],
        recommendations=list(ordered),
    )


def _pre(*reqs: PreprocessingRequirement) -> PreprocessingRequirements:
    order = ["missing-value imputation", "categorical encoding", "numerical scaling"]
    ordered = sorted(reqs, key=lambda r: (order.index(r.description), r.column))
    descs = {r.description for r in ordered}
    return PreprocessingRequirements(
        status=COMPLETED,
        required_operations=[o for o in order if o in descs],
        encoding_required="categorical encoding" in descs,
        scaling_required="numerical scaling" in descs,
        imputation_required="missing-value imputation" in descs,
        requirements=list(ordered),
    )


def _assess(d, inv, trans=None, sel=None, pre=None, **kw):
    return assess_feature_engineering(d, inv, trans or _trans(), sel or _sel(), pre or _pre(), **kw)


def test_duplicate_candidate():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x"), _cand("x")))
    assert a.feasible is False
    assert "inventory consistency" in _blocking_cats(a)


def test_missing_candidate_column():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x"), _cand("ghost")), sel=_sel(selected=["x", "ghost"]))
    assert a.feasible is False
    assert any("not a column of the DataFrame" in m for m in a.blocking_issues)


def test_invalid_candidate_statistics():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x", n_missing=3, n_observations=6)))  # 3+6 != 6
    assert a.feasible is False
    assert any("row count" in m for m in a.blocking_issues)


def test_candidate_excluded_overlap():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x"), excluded=["x"]), sel=_sel(selected=["x"]))
    assert a.feasible is False
    assert any("both candidate and excluded" in m for m in a.blocking_issues)


def test_target_marked_candidate():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x", is_target=True)), sel=_sel(selected=["x"]))
    assert a.feasible is False
    assert any("target-marked" in m for m in a.blocking_issues)


def test_all_missing_candidate():
    d = pd.DataFrame({"x": [np.nan] * 6})
    a = _assess(
        d,
        _inv(
            _cand(
                "x",
                all_missing=True,
                n_observations=0,
                n_missing=6,
                missing_fraction=1.0,
                n_unique=0,
                unique_fraction=0.0,
            )
        ),
        sel=_sel(selected=["x"]),
    )
    assert a.feasible is False
    assert any("entirely missing" in m for m in a.blocking_issues)


def test_constant_candidate():
    d = pd.DataFrame({"x": [1] * 6})
    a = _assess(
        d,
        _inv(_cand("x", constant=True, n_unique=1, unique_fraction=1 / 6)),
        sel=_sel(selected=["x"]),
    )
    assert a.feasible is False
    assert any("marked constant" in m for m in a.blocking_issues)


def test_identifier_like_candidate_inconsistency():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x", identifier_like=True)), sel=_sel(selected=["x"]))
    assert a.feasible is False
    assert any("identifier-like" in m for m in a.blocking_issues)


def test_candidate_features_mismatch():
    d = pd.DataFrame({"x": range(6), "y": range(6)})
    bad = FeatureInventory(
        status=COMPLETED,
        candidate_features=["x"],  # omits y
        candidates=[_cand("x"), _cand("y")],
    )
    a = _assess(d, bad, sel=_sel(selected=["x", "y"]))
    assert a.feasible is False
    assert any("candidate_features does not match" in m for m in a.blocking_issues)


# --- selection consistency ------------------------------


def test_selected_absent_from_inventory(df, inv, trans, pre):
    bad = _sel(selected=["not_a_candidate"])
    a = assess_feature_engineering(df, inv, trans, bad, pre)
    assert a.feasible is False
    assert "selection consistency" in _blocking_cats(a)


def test_selected_dropped_overlap():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x")), sel=_sel(selected=["x"], dropped=["x"]))
    assert a.feasible is False
    assert any("both the selected and dropped" in m for m in a.blocking_issues)


def test_selected_review_overlap():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x")), sel=_sel(selected=["x"], review=["x"]))
    assert a.feasible is False


def test_dropped_review_overlap():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(d, _inv(_cand("x")), sel=_sel(dropped=["x"], review=["x"]))
    assert a.feasible is False


def test_selection_recommendation_action_mismatch():
    d = pd.DataFrame({"x": range(6)})
    s = _sel(selected=["x"])
    s = s.model_copy(
        update={
            "recommendations": [
                FeatureSelectionRecommendation(
                    column="x", action=FeatureSelectionAction.DROP, reason="?"
                )
            ]
        }
    )
    a = _assess(d, _inv(_cand("x")), sel=s)
    assert a.feasible is False
    assert any("is drop but it is not in that list" in m for m in a.blocking_issues)


def test_selection_recommendation_unknown_feature():
    d = pd.DataFrame({"x": range(6)})
    s = _sel(selected=["x"])
    s = s.model_copy(
        update={
            "recommendations": list(s.recommendations)
            + [
                FeatureSelectionRecommendation(
                    column="ghost", action=FeatureSelectionAction.RETAIN, reason="?"
                )
            ]
        }
    )
    a = _assess(d, _inv(_cand("x")), sel=s)
    assert a.feasible is False


# --- transformation consistency ------------------------


def _trec(column, description, operation=FeatureOperationType.TRANSFORMATION):
    return TransformationRecommendation(
        column=column, operation=operation, description=description, reason="r"
    )


def test_transformation_unknown_feature():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(
        d, _inv(_cand("x")), sel=_sel(selected=["x"]), trans=_trans(_trec("ghost", "log transform"))
    )
    assert a.feasible is False
    assert "transformation consistency" in _blocking_cats(a)


def test_transformation_on_dropped_feature():
    d = pd.DataFrame({"x": range(6), "y": range(6)})
    a = _assess(
        d,
        _inv(_cand("x"), _cand("y")),
        sel=_sel(selected=["x"], dropped=["y"]),
        trans=_trans(_trec("y", "log transform")),
    )
    assert a.feasible is False
    assert "cross-section consistency" in _blocking_cats(a)


def test_transformation_missing_value_operation():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(
        d,
        _inv(_cand("x")),
        sel=_sel(selected=["x"]),
        trans=_trans(_trec("x", "impute", FeatureOperationType.MISSING_VALUE_HANDLING)),
    )
    assert a.feasible is False
    assert any("missing-value handling" in m for m in a.blocking_issues)


def test_transformation_recommended_operations_mismatch():
    d = pd.DataFrame({"x": range(6)})
    t = _trans(_trec("x", "log transform"))
    t = t.model_copy(update={"recommended_operations": ["x: WRONG"]})
    a = _assess(d, _inv(_cand("x")), sel=_sel(selected=["x"]), trans=t)
    assert a.feasible is False
    assert any("recommended_operations does not match" in m for m in a.blocking_issues)


def test_transformation_categorical_numeric_incompatibility():
    d = pd.DataFrame({"c": list("abcdef")})
    a = _assess(
        d,
        _inv(_cand("c", column_type=ColumnType.CATEGORICAL)),
        sel=_sel(selected=["c"]),
        trans=_trans(_trec("c", "log transform")),
    )
    assert a.feasible is False
    assert any("numeric-only transformation" in m for m in a.blocking_issues)


def test_transformation_datetime_incompatibility():
    d = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=6)})
    a = _assess(
        d,
        _inv(_cand("t", column_type=ColumnType.DATETIME)),
        sel=_sel(selected=["t"]),
        trans=_trans(_trec("t", "log transform")),
    )
    assert a.feasible is False
    assert any("non-datetime-derivation" in m for m in a.blocking_issues)


def test_transformation_duplicate_recommendation():
    d = pd.DataFrame({"x": range(6)})
    t = TransformationRecommendations(
        status=COMPLETED,
        recommended_operations=["x: log transform", "x: log transform"],
        recommendations=[_trec("x", "log transform"), _trec("x", "log transform")],
    )
    a = _assess(d, _inv(_cand("x")), sel=_sel(selected=["x"]), trans=t)
    assert a.feasible is False
    assert any("duplicate transformation" in m for m in a.blocking_issues)


# --- preprocessing consistency -------------------------


def _preq(column, description):
    op = {
        "missing-value imputation": FeatureOperationType.MISSING_VALUE_HANDLING,
        "categorical encoding": FeatureOperationType.CATEGORICAL_ENCODING,
        "numerical scaling": FeatureOperationType.NUMERICAL_SCALING,
    }[description]
    return PreprocessingRequirement(
        column=column, operation=op, description=description, reason="r"
    )


def test_preprocessing_unknown_feature():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(
        d, _inv(_cand("x")), sel=_sel(selected=["x"]), pre=_pre(_preq("ghost", "numerical scaling"))
    )
    assert a.feasible is False
    assert "preprocessing consistency" in _blocking_cats(a)


def test_preprocessing_on_dropped_feature():
    d = pd.DataFrame({"x": range(6), "y": range(6)})
    a = _assess(
        d,
        _inv(_cand("x"), _cand("y")),
        sel=_sel(selected=["x"], dropped=["y"]),
        pre=_pre(_preq("y", "numerical scaling")),
    )
    assert a.feasible is False
    assert "cross-section consistency" in _blocking_cats(a)


def test_preprocessing_wrong_encoding_type():
    d = pd.DataFrame({"x": range(6)})
    a = _assess(
        d,
        _inv(_cand("x", column_type=ColumnType.NUMERIC)),
        sel=_sel(selected=["x"]),
        pre=_pre(_preq("x", "categorical encoding")),
    )
    assert a.feasible is False
    assert any("non-categorical feature" in m for m in a.blocking_issues)


def test_preprocessing_wrong_scaling_type():
    d = pd.DataFrame({"c": list("abcdef")})
    a = _assess(
        d,
        _inv(_cand("c", column_type=ColumnType.CATEGORICAL)),
        sel=_sel(selected=["c"]),
        pre=_pre(_preq("c", "numerical scaling")),
    )
    assert a.feasible is False
    assert any("non-numeric feature" in m for m in a.blocking_issues)


def test_preprocessing_datetime_encoding():
    d = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=6)})
    a = _assess(
        d,
        _inv(_cand("t", column_type=ColumnType.DATETIME)),
        sel=_sel(selected=["t"]),
        pre=_pre(_preq("t", "categorical encoding")),
    )
    assert a.feasible is False


def test_preprocessing_flag_mismatch():
    d = pd.DataFrame({"c": list("abcdef")})
    p = _pre(_preq("c", "categorical encoding"))
    p = p.model_copy(update={"encoding_required": False})
    a = _assess(
        d, _inv(_cand("c", column_type=ColumnType.CATEGORICAL)), sel=_sel(selected=["c"]), pre=p
    )
    assert a.feasible is False
    assert any("encoding_required flag disagrees" in m for m in a.blocking_issues)


def test_preprocessing_required_operations_mismatch():
    d = pd.DataFrame({"c": list("abcdef")})
    p = _pre(_preq("c", "categorical encoding"))
    p = p.model_copy(update={"required_operations": ["numerical scaling"]})
    a = _assess(
        d, _inv(_cand("c", column_type=ColumnType.CATEGORICAL)), sel=_sel(selected=["c"]), pre=p
    )
    assert a.feasible is False


def test_preprocessing_all_missing_imputation():
    d = pd.DataFrame({"x": [np.nan] * 6})
    inv = _inv(
        _cand(
            "x",
            all_missing=True,
            n_observations=0,
            n_missing=6,
            missing_fraction=1.0,
            n_unique=0,
            unique_fraction=0.0,
        )
    )
    # candidate flag intentionally left True to exercise the imputation check
    a = _assess(d, inv, sel=_sel(review=["x"]), pre=_pre(_preq("x", "missing-value imputation")))
    assert a.feasible is False
    assert any("entirely-missing feature" in m for m in a.blocking_issues)


def test_preprocessing_duplicate_requirement():
    d = pd.DataFrame({"c": list("abcdef")})
    p = PreprocessingRequirements(
        status=COMPLETED,
        required_operations=["categorical encoding"],
        encoding_required=True,
        requirements=[_preq("c", "categorical encoding"), _preq("c", "categorical encoding")],
    )
    a = _assess(
        d, _inv(_cand("c", column_type=ColumnType.CATEGORICAL)), sel=_sel(selected=["c"]), pre=p
    )
    assert a.feasible is False
    assert any("duplicate preprocessing requirement" in m for m in a.blocking_issues)


# --- cross-section + target safety --------------------


def test_target_appears_in_selection(df, inv, trans, pre):
    bad = _sel(selected=["target", "big"])
    a = assess_feature_engineering(df, inv, trans, bad, pre)
    assert a.feasible is False
    assert "target safety" in _blocking_cats(a)


def test_target_appears_in_transformation(df, inv, sel, pre):
    bad = _trans(_trec("target", "log transform"))
    a = assess_feature_engineering(df, inv, bad, sel, pre)
    assert a.feasible is False
    assert "target safety" in _blocking_cats(a)


def test_target_appears_in_preprocessing(df, inv, trans, sel):
    bad = _pre(_preq("target", "numerical scaling"))
    a = assess_feature_engineering(df, inv, trans, sel, bad)
    assert a.feasible is False
    assert "target safety" in _blocking_cats(a)


def test_inventory_exclusion_vs_downstream_recommendation():
    d = pd.DataFrame({"x": range(6), "id": range(6)})
    inv = FeatureInventory(
        status=COMPLETED,
        candidate_features=["x"],
        excluded_features=["id"],
        candidates=[_cand("x"), _cand("id", candidate=False, identifier_like=True)],
    )
    a = _assess(d, inv, sel=_sel(selected=["x"]), trans=_trans(_trec("id", "log transform")))
    assert a.feasible is False


def test_blocking_category_ordering():
    d = pd.DataFrame({"x": range(6)})
    # trigger both an inventory issue and a cross-section issue
    inv = _inv(_cand("x"), _cand("x"))
    a = _assess(d, inv, sel=_sel(selected=["x"]))
    ranks = [
        {
            "upstream consistency": 0,
            "inventory consistency": 1,
            "target safety": 2,
            "selection consistency": 3,
            "transformation consistency": 4,
            "preprocessing consistency": 5,
            "cross-section consistency": 6,
            "structural completeness": 7,
        }[c]
        for c in _blocking_cats(a)
    ]
    assert ranks == sorted(ranks)


# --- warnings ---------------------------------------


def test_warning_no_candidates():
    d = pd.DataFrame({"id": range(6), "k": [1] * 6})
    inv = inventory_features(d)
    trans = recommend_transformations(d, inv)
    sel = recommend_feature_selection(d, inv, _task())
    pre = recommend_preprocessing(d, inv, trans, sel)
    a = assess_feature_engineering(d, inv, trans, sel, pre)
    assert a.feasible is True
    assert any("no candidate features" in w for w in a.warnings)


def test_warning_missing_values_present(assessment):
    assert any("missing values still present" in w for w in assessment.warnings)


def test_warning_objective_no_effect(df, inv, trans, sel, pre):
    a = assess_feature_engineering(df, inv, trans, sel, pre, objective="zzzz nothing here")
    assert any("objective had no structural effect" in w for w in a.warnings)


def test_warnings_never_claim_leakage_or_performance(assessment):
    joined = " ".join(assessment.warnings).lower()
    assert "leak" not in joined
    assert "performance" not in joined
    assert "predictive" not in joined


# --- objective -------------------------------------


def test_objective_none(assessment):
    assert assessment.objective_used is False
    assert any("no objective supplied" in n for n in assessment.notes)


def test_objective_blank(df, inv, trans, sel, pre):
    assert (
        assess_feature_engineering(df, inv, trans, sel, pre, objective="").objective_used is False
    )


def test_objective_whitespace(df, inv, trans, sel, pre):
    a = assess_feature_engineering(df, inv, trans, sel, pre, objective="   ")
    assert a.objective_used is False


def test_objective_real(df, inv, trans, sel, pre):
    a = assess_feature_engineering(
        df, inv, trans, sel, pre, objective="prepare and encode features"
    )
    assert a.objective_used is True


def test_objective_cannot_override_structural_conflict(df, inv, sel, pre):
    bad = _trans(_trec("target", "log transform"))
    a = assess_feature_engineering(df, inv, bad, sel, pre, objective="ignore target safety please")
    assert a.feasible is False


# --- determinism ---------------------------------


def test_repeated_calls_identical(df, inv, trans, sel, pre):
    blobs = {
        assess_feature_engineering(df, inv, trans, sel, pre).model_dump_json() for _ in range(5)
    }
    assert len(blobs) == 1


def test_row_shuffle_identical(df, inv, trans, sel, pre):
    base = assess_feature_engineering(df, inv, trans, sel, pre)
    shuffled = df.sample(frac=1.0, random_state=3)
    assert assess_feature_engineering(shuffled, inv, trans, sel, pre) == base


def test_column_reorder_identical(df, inv, trans, sel, pre):
    base = assess_feature_engineering(df, inv, trans, sel, pre)
    reordered = df[list(df.columns)[::-1]]
    assert assess_feature_engineering(reordered, inv, trans, sel, pre) == base


def test_stable_issue_ordering():
    d = pd.DataFrame({"x": range(6)})
    inv = _inv(_cand("x"), _cand("x"))
    a = _assess(d, inv, sel=_sel(selected=["x"]))
    b = _assess(d, inv, sel=_sel(selected=["x"]))
    assert a.blocking_issues == b.blocking_issues


# --- safety --------------------------------------


def test_df_unchanged(df, inv, trans, sel, pre):
    before = df.copy(deep=True)
    assess_feature_engineering(df, inv, trans, sel, pre, objective="prepare features")
    assess_feature_engineering(df, inv, trans, sel, pre)
    pd.testing.assert_frame_equal(df, before)


def test_upstream_models_unchanged(df, inv, trans, sel, pre):
    snaps = tuple(m.model_dump_json() for m in (inv, trans, sel, pre))
    assess_feature_engineering(df, inv, trans, sel, pre)
    assert tuple(m.model_dump_json() for m in (inv, trans, sel, pre)) == snaps


def test_no_files_created(df, inv, trans, sel, pre, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assess_feature_engineering(df, inv, trans, sel, pre)
    assert list(tmp_path.iterdir()) == []


# --- integration --------------------------------


def test_merge_into_feature_engineering_spec(df):
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="ds", objective="prepare features")
    )
    inv = inventory_features(df, target="target")
    trans = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, _task())
    pre = recommend_preprocessing(df, inv, trans, sel)
    spec = spec.model_copy(update={"inventory": inv})
    spec = spec.model_copy(update={"transformations": trans})
    spec = spec.model_copy(update={"selection": sel})
    spec = spec.model_copy(update={"preprocessing": pre})
    a = assess_feature_engineering(df, inv, trans, sel, pre, objective="prepare features")
    merged = spec.model_copy(update={"assessment": a})

    assert merged.inventory == inv
    assert merged.transformations == trans
    assert merged.selection == sel
    assert merged.preprocessing == pre
    assert merged.assessment.status is COMPLETED
    assert merged.assessment.feasible is True
    assert merged.status is NOT_YET
    assert type(merged).model_validate_json(merged.model_dump_json()) == merged


# --- backward compatibility ---------------------


def test_understand_feature_engineering_signature_unchanged():
    assert list(inspect.signature(understand_feature_engineering).parameters) == ["request"]


def test_phase_6_1_to_6_5_still_work(df):
    inv = inventory_features(df, target="target")
    trans = recommend_transformations(df, inv)
    sel = recommend_feature_selection(df, inv, _task())
    pre = recommend_preprocessing(df, inv, trans, sel)
    for section in (inv, trans, sel, pre):
        assert section.status is COMPLETED


def test_legacy_assessment_json_validates():
    legacy = json.dumps(
        {
            "status": "not_yet_inferred",
            "reason": None,
            "feasible": None,
            "blocking_issues": [],
            "warnings": [],
            "notes": [],
        }
    )
    model = FeatureEngineeringAssessment.model_validate_json(legacy)
    assert model.checks == []
    assert model.objective_used is False


def test_phase_5_apis_still_work(df):
    from data_engine.problem_understanding import identify_target, infer_task_type

    d = df.rename(columns={"target": "price"})
    t = identify_target(d, objective="predict price")
    assert infer_task_type(d, t, objective="predict price").status.value in {
        "completed",
        "unavailable",
    }
