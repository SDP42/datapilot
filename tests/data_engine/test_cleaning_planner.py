"""Cleaning planner: one test per required behaviour.

The planner consumes a QualityReport (+ optional DatasetProfile) and
produces a CleaningPlan of *proposals*. It executes nothing.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from data_engine.cleaning import (
    CleaningPlan,
    OperationCategory,
    OperationStatus,
    OperationType,
    plan_cleaning,
    plan_from_dataframe,
)
from data_engine.cleaning.models import ImputationStrategy
from data_engine.profiling import profile_dataframe
from data_engine.quality import analyze_profile
from data_engine.quality.models import (
    FindingType,
    QualityFinding,
    QualityReport,
    QualitySummary,
    Severity,
    SuggestedAction,
)


def _ops_of(plan: CleaningPlan, op_type: OperationType):
    return [op for op in plan.operations if op.operation_type is op_type]


def _report_from(df: pd.DataFrame, target=None):
    profile = profile_dataframe(df, dataset_id="ds-test")
    report = analyze_profile(df, profile, target_column=target)
    return report, profile


# --- 1 / 2: missing values -----------------------------------------------


def test_missing_numeric_proposes_median_imputation():
    df = pd.DataFrame({"age": [10, 20, None, 40, 50, 60, 70, 80, 90, 100]})
    report, profile = _report_from(df)
    plan = plan_cleaning(report, profile=profile)
    ops = _ops_of(plan, OperationType.IMPUTE_MISSING_NUMERIC)
    assert ops and ops[0].strategy is ImputationStrategy.MEDIAN
    assert ops[0].target_columns == ["age"]
    assert ops[0].requires_train_test_split_awareness is True


def test_missing_categorical_proposes_mode_imputation():
    df = pd.DataFrame({"city": ["A", "B", "A", None, "A", "B", "A", "B", "A", "B"]})
    report, profile = _report_from(df)
    plan = plan_cleaning(report, profile=profile)
    ops = _ops_of(plan, OperationType.IMPUTE_MISSING_CATEGORICAL)
    assert ops and ops[0].strategy is ImputationStrategy.MODE


# --- 3: high missingness -> review, never a silent drop ------------------


def test_high_missingness_is_review_required_not_auto_drop():
    df = pd.DataFrame({"x": list(range(10)), "sparse": [1.0] + [None] * 9})
    report, profile = _report_from(df)
    plan = plan_cleaning(report, profile=profile)

    drop_ops = _ops_of(plan, OperationType.DROP_HIGH_MISSING_COLUMN)
    assert drop_ops, "a drop proposal should be surfaced for very high missingness"
    assert drop_ops[0].status is OperationStatus.NOT_SAFE_TO_AUTOMATE
    # It is a *proposal*: nothing is applied and the impute option also exists.
    assert plan.summary.auto_applicable_count == 0


# --- 4: duplicates ------------------------------------------------------


def test_exact_duplicates_propose_removal():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    report, profile = _report_from(df)
    plan = plan_cleaning(report, profile=profile)
    ops = _ops_of(plan, OperationType.REMOVE_EXACT_DUPLICATE_ROWS)
    assert ops and ops[0].status is OperationStatus.RECOMMENDED
    assert ops[0].parameters["scope"] == "exact_full_row_duplicates"


# --- 5 / 6: type conversion -------------------------------------------


def test_numeric_as_text_proposes_type_conversion_with_validation():
    df = pd.DataFrame({"amt": [str(x) for x in range(20)], "name": list("ab") * 10})
    plan = plan_from_dataframe(df)
    ops = _ops_of(plan, OperationType.CONVERT_TEXT_TO_NUMERIC)
    assert ops and ops[0].parameters["validate_before_apply"] is True
    assert ops[0].parameters["on_unparseable"] == "abort_and_report"


def test_datetime_as_text_proposes_datetime_conversion_no_silent_coercion():
    df = pd.DataFrame({"d": [f"2021-02-{i:02d}" for i in range(1, 21)]})
    plan = plan_from_dataframe(df)
    ops = _ops_of(plan, OperationType.CONVERT_TEXT_TO_DATETIME)
    assert ops and ops[0].status is OperationStatus.REVIEW_REQUIRED
    assert ops[0].parameters["on_unparseable"] == "report_do_not_coerce"


# --- 7: categorical standardization ---------------------------------


def test_categorical_variants_propose_standardization_not_semantic_mapping():
    df = pd.DataFrame({"g": (["Male", "male", "MALE"] * 6) + ["Female", "Female"]})
    plan = plan_from_dataframe(df)
    ops = _ops_of(plan, OperationType.STANDARDIZE_CATEGORY_FORMATTING)
    assert ops and ops[0].status is OperationStatus.REVIEW_REQUIRED
    assert ops[0].parameters["semantic_mapping"] is False


def test_whitespace_only_variants_are_recommended_trim():
    df = pd.DataFrame({"g": (["cat", "cat ", " cat"] * 6) + ["dog", "dog"]})
    plan = plan_from_dataframe(df)
    trim = _ops_of(plan, OperationType.TRIM_CATEGORY_WHITESPACE)
    assert trim and trim[0].status is OperationStatus.RECOMMENDED


# --- 8: outliers are never auto-deleted -----------------------------


def test_outliers_do_not_produce_deletion():
    df = pd.DataFrame({"v": list(range(1, 101)) + [1_000_000]})
    plan = plan_from_dataframe(df)
    review = _ops_of(plan, OperationType.REVIEW_OUTLIERS)
    assert review and review[0].category is OperationCategory.INVESTIGATION
    assert review[0].parameters["confirmed_error"] is False
    # No operation anywhere in the plan deletes/caps/replaces rows or values.
    text = " ".join(op.proposed_action.lower() for op in plan.operations)
    assert "delete" not in text.replace("do not delete", "")
    assert not any(
        "drop" in op.operation_type.value and "row" in op.operation_type.value
        for op in plan.operations
    )


# --- 9: unsafe log transform guarded -------------------------------


def test_negative_or_zero_values_block_log_transform():
    rng = np.random.default_rng(1)
    skewed_with_zero = np.concatenate([rng.exponential(1.0, 400), [0.0], [40, 50, 60]])
    df = pd.DataFrame({"z": skewed_with_zero})
    plan = plan_from_dataframe(df)
    assert not _ops_of(plan, OperationType.TRANSFORM_DISTRIBUTION_LOG)
    review = _ops_of(plan, OperationType.REVIEW_DISTRIBUTION_TRANSFORM)
    assert review and review[0].parameters["plain_log_applicable"] is False


def test_strictly_positive_skew_allows_log_transform_as_review():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"p": np.concatenate([rng.exponential(1.0, 400) + 0.01, [40, 50, 60]])})
    plan = plan_from_dataframe(df)
    ops = _ops_of(plan, OperationType.TRANSFORM_DISTRIBUTION_LOG)
    assert ops and ops[0].status is OperationStatus.REVIEW_REQUIRED
    assert ops[0].parameters["transform"] == "log"


# --- 10: class imbalance -> modelling advice, not a transform -----


def test_class_imbalance_produces_modeling_recommendation_only():
    df = pd.DataFrame({"x": range(100), "y": [0] * 92 + [1] * 8})
    plan = plan_from_dataframe(df, target_column="y")
    ops = _ops_of(plan, OperationType.RECOMMEND_IMBALANCE_STRATEGY)
    assert ops and ops[0].category is OperationCategory.MODELING_RECOMMENDATION
    assert ops[0].parameters["is_data_transformation"] is False


# --- 11-14: plan-level guarantees --------------------------------


def test_every_operation_references_its_source_finding():
    df = pd.DataFrame(
        {
            "a": [1, None, 3, 3, 5, 6, 7, 8, 9, 9],
            "g": ["x", "X", "y", "y", "x", "X", "y", "y", "x", "X"],
        }
    )
    report, profile = _report_from(df)
    finding_ids = {f.finding_id for f in report.findings}
    plan = plan_cleaning(report, profile=profile)
    assert plan.operations
    for op in plan.operations:
        assert op.source_finding_id in finding_ids
        assert op.addresses_finding_type in {f.finding_type for f in report.findings}


def test_plan_is_json_serialisable_and_round_trips():
    df = pd.DataFrame({"a": [1, None, 3, 3], "g": ["x", "X", "y", "y"]})
    plan = plan_from_dataframe(df, dataset_id="ds-json")
    payload = plan.model_dump_json()
    restored = CleaningPlan.model_validate_json(payload)
    assert restored.dataset_id == "ds-json"
    assert restored.summary.total_operations == plan.summary.total_operations


def test_planner_does_not_mutate_dataframe():
    df = pd.DataFrame(
        {
            "age": [10, None, 30, 40, 50, 60, 70, 80, 90, 200],
            "g": ["a", "A", "a", "A", "a", "A", "a", "A", "a", "A"],
            "amt": [str(x) for x in range(10)],
        }
    )
    before = df.copy(deep=True)
    plan_from_dataframe(df, target_column="g")
    pd.testing.assert_frame_equal(df, before)


def test_planner_does_not_mutate_quality_report():
    df = pd.DataFrame({"a": [1, None, 3, 3, 5, 6, 7, 8, 9, 9]})
    report, profile = _report_from(df)
    snapshot = report.model_dump_json()
    plan_cleaning(report, profile=profile)
    assert report.model_dump_json() == snapshot


# --- extra: works without a profile (degrades to review) -------


def test_without_profile_missing_values_are_review_required():
    report = QualityReport(
        dataset_id="ds-noprofile",
        generated_at=dt.datetime.now(dt.UTC),
        summary=QualitySummary(
            n_rows=100,
            n_columns=1,
            total_findings=1,
            findings_by_severity={Severity.MEDIUM: 1},
            findings_by_type={FindingType.MISSING_VALUES: 1},
            columns_with_findings=["age"],
            has_critical=False,
            score=94.0,
        ),
        findings=[
            QualityFinding(
                finding_id="missing_values:age",
                finding_type=FindingType.MISSING_VALUES,
                severity=Severity.MEDIUM,
                columns=["age"],
                affected_rows=3,
                affected_percentage=3.0,
                observed={"missing_count": 3, "missing_percentage": 3.0, "fully_missing": False},
                description="Column 'age' has 3 missing value(s) (3.00% of rows).",
                recommended_action=SuggestedAction.HANDLE_MISSING_VALUES,
            )
        ],
    )
    plan = plan_cleaning(report)  # no profile
    assert plan.used_profile is False
    ops = _ops_of(plan, OperationType.HANDLE_MISSING_VALUES)
    assert ops and ops[0].status is OperationStatus.REVIEW_REQUIRED
    assert ops[0].source_finding_id == "missing_values:age"


def test_operations_sorted_safest_first():
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # duplicate -> recommended
            "sparse": [1.0] + [None] * 9,  # high missing -> not_safe drop
        }
    )
    plan = plan_from_dataframe(df)
    order = {
        OperationStatus.RECOMMENDED: 0,
        OperationStatus.REVIEW_REQUIRED: 1,
        OperationStatus.NOT_SAFE_TO_AUTOMATE: 2,
    }
    statuses = [order[op.status] for op in plan.operations]
    assert statuses == sorted(statuses)
