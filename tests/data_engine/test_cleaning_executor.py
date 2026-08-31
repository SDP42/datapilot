"""Safe cleaning executor: basic execution, safety, leakage, validation,
reporting, and an end-to-end integration test."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from data_engine.cleaning import (
    CleaningExecutionReport,
    ExecutionContext,
    ExecutionStatus,
    OperationCategory,
    OperationStatus,
    OperationType,
    ProcessedDataStore,
    execute_cleaning,
    execute_dataframe,
    plan_from_dataframe,
)
from data_engine.cleaning.executor import EXECUTORS
from data_engine.cleaning.models import CleaningOperation, CleaningPlan
from data_engine.cleaning.summary import build_plan_summary
from data_engine.cleaning.validation import validate_after
from data_engine.ingestion import RawDataStore, ingest_dataset
from data_engine.quality.models import FindingType

FULL_FIT = ExecutionContext(allow_full_data_fit=True)


def _op(
    op_type: OperationType,
    *,
    columns: list[str],
    status: OperationStatus = OperationStatus.RECOMMENDED,
    category: OperationCategory = OperationCategory.DATA_TRANSFORMATION,
    params: dict | None = None,
    strategy=None,
    ttsa: bool = False,
    finding_type: FindingType = FindingType.MISSING_VALUES,
) -> CleaningOperation:
    anchor = columns[0] if columns else "_dataset_"
    return CleaningOperation(
        operation_id=f"{op_type.value}:{anchor}",
        operation_type=op_type,
        category=category,
        status=status,
        status_reason="test",
        target_columns=list(columns),
        addresses_finding_type=finding_type,
        source_finding_id=f"{finding_type.value}:{anchor}",
        problem_summary="p",
        proposed_action="a",
        rationale="r",
        strategy=strategy,
        parameters=params or {},
        requires_train_test_split_awareness=ttsa,
    )


def _plan(
    *ops: CleaningOperation, dataset_id: str = "ds-t", target_column: str | None = None
) -> CleaningPlan:
    return CleaningPlan(
        dataset_id=dataset_id,
        generated_at=dt.datetime.now(dt.UTC),
        target_column=target_column,
        based_on_quality_engine_version="1",
        used_profile=True,
        source_findings_considered=len(ops),
        operations=list(ops),
        summary=build_plan_summary(list(ops)),
    )


def _rec(report: CleaningExecutionReport, op_id: str):
    return next(r for r in report.operations if r.operation_id == op_id)


# ======================================================================
# Basic execution
# ======================================================================


def test_numeric_median_imputation_executes():
    df = pd.DataFrame({"age": [10.0, 20, np.nan, 40, 50, 60, 70, 80, 90, np.nan]})
    op = _op(OperationType.IMPUTE_MISSING_NUMERIC, columns=["age"], strategy="median", ttsa=True)
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert rec.fit_details["strategy"] == "median"
    assert res.cleaned["age"].isna().sum() == 0
    assert res.cleaned["age"].iloc[2] == df["age"].median()


def test_categorical_mode_imputation_executes():
    df = pd.DataFrame({"city": ["A", "B", "A", None, "A", "B", "A", "B", "A", None]})
    op = _op(OperationType.IMPUTE_MISSING_CATEGORICAL, columns=["city"], strategy="mode", ttsa=True)
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert rec.fit_details["fit_value"] == "A"
    assert (res.cleaned["city"].iloc[[3, 9]] == "A").all()


def test_exact_duplicate_removal_executes():
    df = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": ["x", "x", "y", "z", "z"]})
    op = _op(
        OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
        columns=[],
        finding_type=FindingType.DUPLICATE_ROWS,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert rec.parameters_used["duplicates_removed"] == 2
    assert len(res.cleaned) == 3


def test_numeric_type_conversion_executes():
    df = pd.DataFrame({"amt": [str(x) for x in range(12)]})
    op = _op(
        OperationType.CONVERT_TEXT_TO_NUMERIC,
        columns=["amt"],
        params={"on_unparseable": "abort_and_report"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert pd.api.types.is_numeric_dtype(res.cleaned["amt"])


def test_datetime_conversion_executes_with_explicit_format():
    df = pd.DataFrame({"d": [f"2021-03-{i:02d}" for i in range(1, 13)]})
    op = _op(
        OperationType.CONVERT_TEXT_TO_DATETIME,
        columns=["d"],
        status=OperationStatus.REVIEW_REQUIRED,
        params={"on_unparseable": "report_do_not_coerce"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(
        df,
        _plan(op),
        approved_operation_ids=[op.operation_id],
        operation_parameter_overrides={op.operation_id: {"format": "%Y-%m-%d"}},
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert pd.api.types.is_datetime64_any_dtype(res.cleaned["d"])
    assert rec.parameters_used["format_used"] == "%Y-%m-%d"


def test_whitespace_trimming_executes():
    df = pd.DataFrame({"g": [" Male ", "Male", "Male  ", " Male", "Female", "Female"]})
    op = _op(
        OperationType.TRIM_CATEGORY_WHITESPACE,
        columns=["g"],
        params={"normalization": ["strip", "collapse_internal_whitespace"]},
        finding_type=FindingType.INCONSISTENT_CATEGORIES,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert set(res.cleaned["g"].unique()) == {"Male", "Female"}
    assert rec.parameters_used["unique_values_before"] > rec.parameters_used["unique_values_after"]


def test_category_standardisation_executes():
    df = pd.DataFrame({"g": ["Male", "male", "MALE", "male", "Female", "Female"]})
    op = _op(
        OperationType.STANDARDIZE_CATEGORY_FORMATTING,
        columns=["g"],
        status=OperationStatus.REVIEW_REQUIRED,
        params={
            "variant_groups": {"male": ["Male", "male", "MALE"]},
            "canonical_choice": "most_frequent_variant",
            "semantic_mapping": False,
        },
        finding_type=FindingType.INCONSISTENT_CATEGORIES,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert set(res.cleaned["g"].unique()) == {"male", "Female"}
    assert rec.parameters_used["mapping"] == {"Male": "male", "MALE": "male"}


def test_positive_log_transform_executes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": np.concatenate([rng.exponential(2.0, 60) + 0.5, [50, 60]])})
    op = _op(
        OperationType.TRANSFORM_DISTRIBUTION_LOG,
        columns=["x"],
        status=OperationStatus.REVIEW_REQUIRED,
        params={"transform": "log"},
        ttsa=True,
        finding_type=FindingType.HIGH_SKEW,
    )
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SUCCESS
    assert rec.parameters_used["minimum_after"] < rec.parameters_used["minimum_before"]
    assert np.allclose(res.cleaned["x"], np.log(df["x"]))


# ======================================================================
# Safety
# ======================================================================


def test_raw_dataframe_is_never_mutated():
    df = pd.DataFrame({"age": [1.0, None, 3, 4, 5], "amt": ["1", "2", "3", "4", "5"]})
    before = df.copy(deep=True)
    plan = plan_from_dataframe(df, dataset_id="ds-x")
    execute_dataframe(
        df,
        plan,
        approved_operation_ids=[o.operation_id for o in plan.operations],
        context=FULL_FIT,
    )
    pd.testing.assert_frame_equal(df, before)


def test_raw_file_is_never_overwritten(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,x\n1,x\n2,y\n", encoding="utf-8")
    raw_store = RawDataStore(tmp_path / "raw")
    ref = ingest_dataset(src, raw_store=raw_store)
    raw_bytes = ref.raw_path.read_bytes()
    raw_mode = ref.raw_path.stat().st_mode

    plan = plan_from_dataframe(pd.read_csv(ref.raw_path), dataset_id=ref.dataset_id)
    dup = next(
        o for o in plan.operations if o.operation_type is OperationType.REMOVE_EXACT_DUPLICATE_ROWS
    )
    execute_cleaning(
        ref,
        plan,
        approved_operation_ids=[dup.operation_id],
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )
    assert ref.raw_path.read_bytes() == raw_bytes
    assert ref.raw_path.stat().st_mode == raw_mode


def test_review_required_without_approval_is_skipped():
    df = pd.DataFrame({"age": [1.0, None, 3, 4, 5, 6, 7, 8, 9, 10]})
    op = _op(
        OperationType.IMPUTE_MISSING_NUMERIC,
        columns=["age"],
        status=OperationStatus.REVIEW_REQUIRED,
        ttsa=True,
    )
    res = execute_dataframe(df, _plan(op), context=FULL_FIT)  # not approved
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SKIPPED
    assert res.cleaned["age"].isna().sum() == 1


def test_not_safe_to_automate_cannot_execute_even_if_approved():
    df = pd.DataFrame({"sparse": [1.0] + [None] * 9, "keep": list(range(10))})
    op = _op(
        OperationType.DROP_HIGH_MISSING_COLUMN,
        columns=["sparse"],
        status=OperationStatus.NOT_SAFE_TO_AUTOMATE,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.FAILED
    assert "not_safe_to_automate" in (rec.error or "")
    assert "sparse" in res.cleaned.columns


def test_unsupported_operation_fails_safely(monkeypatch):
    df = pd.DataFrame({"a": [1, 1, 2]})
    op = _op(
        OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
        columns=[],
        finding_type=FindingType.DUPLICATE_ROWS,
    )
    monkeypatch.delitem(EXECUTORS, OperationType.REMOVE_EXACT_DUPLICATE_ROWS)
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.FAILED
    assert len(res.cleaned) == 3


def test_missing_target_column_fails_safely():
    df = pd.DataFrame({"a": [1, 2, 3]})
    op = _op(
        OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
        columns=[],
        finding_type=FindingType.DUPLICATE_ROWS,
    )
    with pytest.raises(ValueError, match="target_column"):
        execute_dataframe(
            df, _plan(op, target_column="ghost"), approved_operation_ids=[op.operation_id]
        )


def test_missing_operation_column_fails_safely():
    df = pd.DataFrame({"a": [1.0, None, 3]})
    op = _op(OperationType.IMPUTE_MISSING_NUMERIC, columns=["ghost"], strategy="median", ttsa=True)
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.FAILED
    assert any("ghost" in m for m in rec.validation_messages)
    pd.testing.assert_frame_equal(res.cleaned, df)


# ======================================================================
# Conversion safety
# ======================================================================


def test_numeric_conversion_aborts_on_unparseable_without_partial_mutation():
    df = pd.DataFrame({"v": ["10", "20", "30", "N/A", "50"]})
    op = _op(
        OperationType.CONVERT_TEXT_TO_NUMERIC,
        columns=["v"],
        params={"on_unparseable": "abort_and_report"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.ABORTED
    assert res.cleaned["v"].tolist() == ["10", "20", "30", "N/A", "50"]
    assert res.cleaned["v"].isna().sum() == 0


def test_datetime_conversion_report_do_not_coerce_aborts_on_invalid():
    df = pd.DataFrame({"d": ["2021-01-01", "2021-01-02", "not-a-date", "2021-01-04"]})
    op = _op(
        OperationType.CONVERT_TEXT_TO_DATETIME,
        columns=["d"],
        status=OperationStatus.REVIEW_REQUIRED,
        params={"on_unparseable": "report_do_not_coerce"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(
        df,
        _plan(op),
        approved_operation_ids=[op.operation_id],
        operation_parameter_overrides={op.operation_id: {"format": "%Y-%m-%d"}},
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.ABORTED
    assert res.cleaned["d"].tolist() == df["d"].tolist()


def test_ambiguous_datetime_format_is_not_guessed():
    df = pd.DataFrame({"d": ["01/02/2026", "03/04/2026", "05/06/2026"]})
    op = _op(
        OperationType.CONVERT_TEXT_TO_DATETIME,
        columns=["d"],
        status=OperationStatus.REVIEW_REQUIRED,
        params={"on_unparseable": "report_do_not_coerce"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SKIPPED
    assert "guess" in rec.message.lower()
    pd.testing.assert_frame_equal(res.cleaned, df)


# ======================================================================
# Mathematical safety
# ======================================================================


def _log_op(col: str) -> CleaningOperation:
    return _op(
        OperationType.TRANSFORM_DISTRIBUTION_LOG,
        columns=[col],
        status=OperationStatus.REVIEW_REQUIRED,
        params={"transform": "log"},
        ttsa=True,
        finding_type=FindingType.HIGH_SKEW,
    )


def test_log_transform_fails_on_zero():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    op = _log_op("x")
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.ABORTED
    pd.testing.assert_frame_equal(res.cleaned, df)


def test_log_transform_fails_on_negative():
    df = pd.DataFrame({"x": [-1.0, 1.0, 2.0, 3.0, 4.0]})
    op = _log_op("x")
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.ABORTED
    pd.testing.assert_frame_equal(res.cleaned, df)


def test_log_transform_never_silently_switches_to_log1p():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    op = _log_op("x")
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert "log1p" in rec.message
    assert "NOT substitute" in rec.message


# ======================================================================
# Outlier / modeling safety
# ======================================================================


def test_review_outliers_never_modifies_data():
    df = pd.DataFrame({"v": list(range(1, 30)) + [10_000]})
    op = _op(
        OperationType.REVIEW_OUTLIERS,
        columns=["v"],
        status=OperationStatus.REVIEW_REQUIRED,
        category=OperationCategory.INVESTIGATION,
        finding_type=FindingType.POTENTIAL_OUTLIERS,
    )
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SKIPPED
    pd.testing.assert_frame_equal(res.cleaned, df)


def test_class_imbalance_recommendation_never_modifies_data():
    df = pd.DataFrame({"x": range(20), "y": [0] * 17 + [1] * 3})
    op = _op(
        OperationType.RECOMMEND_IMBALANCE_STRATEGY,
        columns=["y"],
        status=OperationStatus.REVIEW_REQUIRED,
        category=OperationCategory.MODELING_RECOMMENDATION,
        finding_type=FindingType.CLASS_IMBALANCE,
    )
    res = execute_dataframe(
        df, _plan(op, target_column="y"), approved_operation_ids=[op.operation_id]
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.SKIPPED
    pd.testing.assert_frame_equal(res.cleaned, df)


# ======================================================================
# Leakage
# ======================================================================


def test_median_imputation_uses_training_data_only():
    df = pd.DataFrame({"age": [10.0, 20, 30, 40, 1000, 1000, 1000, 1000, np.nan, np.nan]})
    op = _op(OperationType.IMPUTE_MISSING_NUMERIC, columns=["age"], strategy="median", ttsa=True)
    ctx = ExecutionContext(train_index=[0, 1, 2, 3])
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id], context=ctx)
    rec = _rec(res.report, op.operation_id)
    assert rec.fit_details["fit_on"] == "train_split"
    assert rec.fit_details["fit_value"] == 25.0  # median of [10,20,30,40], not the full data
    assert res.cleaned["age"].iloc[8] == 25.0


def test_mode_imputation_uses_training_data_only():
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "z", "z", "z", "z", "z", None]})
    op = _op(OperationType.IMPUTE_MISSING_CATEGORICAL, columns=["c"], strategy="mode", ttsa=True)
    ctx = ExecutionContext(train_index=[0, 1, 2, 3])
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id], context=ctx)
    rec = _rec(res.report, op.operation_id)
    assert rec.fit_details["fit_value"] == "a"  # train mode, not the global "z"


def test_learned_transformation_refuses_without_fit_scope():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8]})
    op = _log_op("x")
    res = execute_dataframe(df, _plan(op), approved_operation_ids=[op.operation_id])  # no context
    rec = _rec(res.report, op.operation_id)
    assert rec.status is ExecutionStatus.FAILED
    assert any("training split" in m for m in rec.validation_messages)
    pd.testing.assert_frame_equal(res.cleaned, df)


# ======================================================================
# Validation framework
# ======================================================================


def test_validation_detects_unexpected_nans():
    op = _op(
        OperationType.TRIM_CATEGORY_WHITESPACE,
        columns=["g"],
        finding_type=FindingType.INCONSISTENT_CATEGORIES,
    )
    before = pd.DataFrame({"g": ["a", "b"], "other": [1, 2]})
    after = pd.DataFrame({"g": ["a", "b"], "other": [1, None]})
    errors = validate_after(op, before, after, target_column=None)
    assert any("unexpected missing" in e for e in errors)


def test_validation_detects_unexpected_dtype_change():
    op = _op(
        OperationType.CONVERT_TEXT_TO_NUMERIC,
        columns=["v"],
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    before = pd.DataFrame({"v": ["1", "2"]})
    after = pd.DataFrame({"v": ["1", "2"]})  # still text
    errors = validate_after(op, before, after, target_column=None)
    assert any("not numeric" in e for e in errors)


def test_validation_detects_disappearing_target_column():
    op = _op(
        OperationType.TRIM_CATEGORY_WHITESPACE,
        columns=["g"],
        finding_type=FindingType.INCONSISTENT_CATEGORIES,
    )
    before = pd.DataFrame({"g": ["a"], "target": [1]})
    after = pd.DataFrame({"g": ["a"]})
    errors = validate_after(op, before, after, target_column="target")
    assert any("target" in e for e in errors)


def test_operation_execution_is_atomic_across_a_plan():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "v": ["10", "10", "30", "x"]})
    dedup = _op(
        OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
        columns=[],
        finding_type=FindingType.DUPLICATE_ROWS,
    )
    convert = _op(
        OperationType.CONVERT_TEXT_TO_NUMERIC,
        columns=["v"],
        params={"on_unparseable": "abort_and_report"},
        finding_type=FindingType.POTENTIAL_TYPE_MISMATCH,
    )
    res = execute_dataframe(
        df, _plan(dedup, convert), approved_operation_ids=[dedup.operation_id, convert.operation_id]
    )
    assert _rec(res.report, dedup.operation_id).status is ExecutionStatus.SUCCESS
    assert _rec(res.report, convert.operation_id).status is ExecutionStatus.ABORTED
    # dedup committed; convert aborted without touching 'v'
    assert len(res.cleaned) == 3
    assert res.cleaned["v"].tolist() == ["10", "30", "x"]


# ======================================================================
# Reporting
# ======================================================================


def _small_plan_and_df():
    df = pd.DataFrame(
        {
            "age": [1.0, None, 3, 4, 5, 6, 7, 8, 9, 9],
            "b": ["x", "x", "y", "y", "x", "y", "x", "y", "x", "y"],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    plan = plan_from_dataframe(df, dataset_id="ds-report")
    return df, plan


def test_execution_report_serialises_to_json():
    df, plan = _small_plan_and_df()
    res = execute_dataframe(
        df, plan, approved_operation_ids=[o.operation_id for o in plan.operations], context=FULL_FIT
    )
    json.dumps(res.report.model_dump(mode="json"))


def test_execution_report_round_trips():
    df, plan = _small_plan_and_df()
    res = execute_dataframe(
        df, plan, approved_operation_ids=[o.operation_id for o in plan.operations], context=FULL_FIT
    )
    restored = CleaningExecutionReport.model_validate_json(res.report.model_dump_json())
    assert restored.execution_id == res.report.execution_id
    assert len(restored.operations) == len(res.report.operations)


def test_operation_records_contain_source_ids():
    df, plan = _small_plan_and_df()
    res = execute_dataframe(
        df, plan, approved_operation_ids=[o.operation_id for o in plan.operations], context=FULL_FIT
    )
    for rec in res.report.operations:
        assert rec.operation_id
        assert rec.source_finding_id


def test_before_after_statistics_recorded():
    df = pd.DataFrame({"age": [1.0, None, None, 4, 5, 6, 7, 8, 9, 10]})
    op = _op(OperationType.IMPUTE_MISSING_NUMERIC, columns=["age"], strategy="median", ttsa=True)
    res = execute_dataframe(
        df, _plan(op), approved_operation_ids=[op.operation_id], context=FULL_FIT
    )
    rec = _rec(res.report, op.operation_id)
    assert rec.before_statistics["age"].missing_count == 2
    assert rec.after_statistics["age"].missing_count == 0


def test_lineage_information_present():
    df, plan = _small_plan_and_df()
    res = execute_dataframe(
        df, plan, approved_operation_ids=[o.operation_id for o in plan.operations], context=FULL_FIT
    )
    lineage = res.report.lineage
    assert lineage.raw_dataset_id == "ds-report"
    assert lineage.plan_fingerprint
    assert len(lineage.steps) == len(plan.operations)
    assert all(step.operation_id for step in lineage.steps)


def test_post_cleaning_quality_comparison_present():
    df = pd.DataFrame({"age": [1.0, None, 3, 4, 5, 6, 7, 8, 9, 10]})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    plan = plan_from_dataframe(df, dataset_id="ds-q")
    approved = [
        o.operation_id
        for o in plan.operations
        if o.operation_type
        in (OperationType.REMOVE_EXACT_DUPLICATE_ROWS, OperationType.IMPUTE_MISSING_NUMERIC)
    ]
    res = execute_dataframe(df, plan, approved_operation_ids=approved, context=FULL_FIT)
    assert res.report.quality_comparison is not None
    assert res.report.before_quality_summary["total_missing_cells"] >= 1
    assert res.report.after_quality_summary["total_missing_cells"] == 0
    assert any("missing_values" in imp for imp in res.report.quality_comparison.improvements)


# ======================================================================
# Integration
# ======================================================================


def test_full_pipeline_ingest_to_post_cleaning_quality(tmp_path):
    src = tmp_path / "customers.csv"
    src.write_text(
        "age,city,score\n"
        "34,London,9.5\n"
        "34,London,9.5\n"  # exact duplicate
        ",Paris,7.1\n"
        "29,London,\n"
        "41,Berlin,8.8\n"
        "52,Paris,6.0\n"
        "37,London,7.7\n"
        "45,Berlin,8.1\n",
        encoding="utf-8",
    )
    ref = ingest_dataset(src, raw_store=RawDataStore(tmp_path / "raw"))
    df = pd.read_csv(ref.raw_path)
    plan = plan_from_dataframe(df, dataset_id=ref.dataset_id)

    approved = [
        o.operation_id
        for o in plan.operations
        if o.operation_type
        in (
            OperationType.REMOVE_EXACT_DUPLICATE_ROWS,
            OperationType.IMPUTE_MISSING_NUMERIC,
            OperationType.IMPUTE_MISSING_CATEGORICAL,
        )
    ]
    report = execute_cleaning(
        ref,
        plan,
        approved_operation_ids=approved,
        context=ExecutionContext(allow_full_data_fit=True),
        processed_store=ProcessedDataStore(tmp_path / "processed"),
    )

    assert report.status.value in ("completed", "completed_with_failures")
    assert report.output_dataset_reference is not None
    out_path = report.output_dataset_reference.path
    assert out_path.exists()
    assert out_path.parent.parent.parent == (tmp_path / "processed")

    # raw stays put and immutable
    assert ref.raw_path.exists()
    assert ref.raw_path.stat().st_mode & 0o222 == 0

    processed = pd.read_csv(out_path)
    assert len(processed) < len(df)  # a duplicate was removed
    assert report.quality_comparison is not None
    assert (
        report.after_quality_summary["total_missing_cells"]
        <= report.before_quality_summary["total_missing_cells"]
    )
