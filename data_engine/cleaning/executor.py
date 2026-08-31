"""The safe, deterministic cleaning executor.

    CleaningPlan (+ explicit approval + execution context)
        -> validate -> execute atomically -> validate result -> commit
        -> lineage + processed dataset + before/after quality comparison
        -> CleaningExecutionReport

Principle: planning decides what *may* be done; execution performs only
what was explicitly approved. The executor never invents a cleaning
decision that is not a typed CleaningOperation in the plan.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from data_engine.profiling import load_dataframe, profile_dataframe
from data_engine.profiling.models import DatasetProfile
from data_engine.quality import analyze_profile
from datapilot.contracts import DatasetReference

from .execution_models import (
    CleaningExecutionReport,
    DatasetLineage,
    ExecutionReportStatus,
    ExecutionStatus,
    LineageStep,
    OperationExecution,
    QualityComparison,
    ValidationSummary,
)
from .executors import base
from .executors.base import ExecResult, ExecutionContext
from .executors.categories import execute_standardize_formatting, execute_trim_whitespace
from .executors.distribution import (
    execute_log_transform,
    execute_review_distribution_transform,
)
from .executors.duplicates import execute_remove_exact_duplicates
from .executors.missing_values import (
    execute_categorical,
    execute_datetime,
    execute_drop_high_missing_column,
    execute_generic,
    execute_numeric,
)
from .executors.noop import execute_recommend_imbalance_strategy, execute_review_outliers
from .executors.type_conversion import execute_to_datetime, execute_to_numeric
from .models import (
    CleaningOperation,
    CleaningPlan,
    OperationCategory,
    OperationStatus,
    OperationType,
)
from .processed_store import ProcessedDataStore
from .quality_comparison import compare_quality
from .statistics import column_statistics
from .validation import validate_after, validate_before

Executor = Callable[[pd.DataFrame, CleaningOperation, ExecutionContext, dict], ExecResult]

EXECUTORS: dict[OperationType, Executor] = {
    OperationType.IMPUTE_MISSING_NUMERIC: execute_numeric,
    OperationType.IMPUTE_MISSING_CATEGORICAL: execute_categorical,
    OperationType.IMPUTE_MISSING_DATETIME: execute_datetime,
    OperationType.HANDLE_MISSING_VALUES: execute_generic,
    OperationType.DROP_HIGH_MISSING_COLUMN: execute_drop_high_missing_column,
    OperationType.REMOVE_EXACT_DUPLICATE_ROWS: execute_remove_exact_duplicates,
    OperationType.CONVERT_TEXT_TO_NUMERIC: execute_to_numeric,
    OperationType.CONVERT_TEXT_TO_DATETIME: execute_to_datetime,
    OperationType.TRIM_CATEGORY_WHITESPACE: execute_trim_whitespace,
    OperationType.STANDARDIZE_CATEGORY_FORMATTING: execute_standardize_formatting,
    OperationType.TRANSFORM_DISTRIBUTION_LOG: execute_log_transform,
    OperationType.REVIEW_DISTRIBUTION_TRANSFORM: execute_review_distribution_transform,
    OperationType.REVIEW_OUTLIERS: execute_review_outliers,
    OperationType.RECOMMEND_IMBALANCE_STRATEGY: execute_recommend_imbalance_strategy,
}

_NON_TRANSFORMING = {OperationCategory.INVESTIGATION, OperationCategory.MODELING_RECOMMENDATION}


def available_executors() -> tuple[OperationType, ...]:
    return tuple(EXECUTORS)


@dataclass
class CleaningExecutionResult:
    """Return type of :func:`execute_dataframe` — report plus the cleaned frame."""

    report: CleaningExecutionReport
    cleaned: pd.DataFrame


def _plan_fingerprint(plan: CleaningPlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()[:16]


def _execution_id(
    plan_fingerprint: str,
    approved: set[str],
    ctx: ExecutionContext,
    auto_recommended: bool,
    raw_sha: str,
) -> str:
    h = hashlib.sha256()
    h.update(plan_fingerprint.encode())
    for op_id in sorted(approved):
        h.update(b"\x00")
        h.update(op_id.encode())
    h.update(
        f"|train={ctx.train_index}|full={ctx.allow_full_data_fit}|auto={auto_recommended}|raw={raw_sha}".encode()
    )
    return h.hexdigest()[:16]


def _df_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[
        :16
    ]


def _eligible(op: CleaningOperation, approved: set[str], auto_recommended: bool) -> bool:
    if op.operation_id in approved:
        return True
    return op.status is OperationStatus.RECOMMENDED and auto_recommended


@dataclass
class _Attempt:
    """What the decision logic concluded for one operation — no side effects."""

    status: ExecutionStatus
    message: str
    result: ExecResult | None = None
    validation_messages: list[str] = field(default_factory=list)
    error: str | None = None
    commit: bool = False


def _decide(
    op: CleaningOperation,
    working: pd.DataFrame,
    *,
    approved_flag: bool,
    approved: set[str],
    auto_recommended: bool,
    ctx: ExecutionContext,
    overrides: dict[str, dict[str, Any]],
    supported: set[OperationType],
    target_column: str | None,
) -> _Attempt:
    # 1. non-transforming operations
    if op.category in _NON_TRANSFORMING:
        fn = EXECUTORS.get(op.operation_type)
        res = (
            fn(working.copy(), op, ctx, overrides)
            if fn
            else base.skipped(working, "non-transforming operation")
        )
        return _Attempt(ExecutionStatus.SKIPPED, res.message, result=res)

    # 2. not_safe_to_automate — never executes
    if op.status is OperationStatus.NOT_SAFE_TO_AUTOMATE:
        if approved_flag:
            return _Attempt(
                ExecutionStatus.FAILED,
                "Rejected: operation is 'not_safe_to_automate' and cannot be executed by the "
                "executor. It requires explicit domain context / a different phase.",
                error="not_safe_to_automate",
            )
        return _Attempt(ExecutionStatus.SKIPPED, "not_safe_to_automate; not executed.")

    # 3. approval boundary
    if not _eligible(op, approved, auto_recommended):
        return _Attempt(
            ExecutionStatus.SKIPPED, f"status={op.status.value} and not explicitly approved."
        )

    # 4. before-validation
    pre_errors = validate_before(
        op,
        working,
        has_fit_scope=ctx.has_fit_scope,
        supported=supported,
        overrides=overrides,
        target_column=target_column,
    )
    if pre_errors:
        return _Attempt(
            ExecutionStatus.FAILED,
            "Pre-execution validation failed.",
            validation_messages=pre_errors,
            error="pre_validation_failed",
        )

    # 5. execute on a copy
    fn = EXECUTORS[op.operation_type]
    try:
        res = fn(working.copy(), op, ctx, overrides)
    except Exception as exc:  # noqa: BLE001 - surfaced as a FAILED record, never raised
        return _Attempt(ExecutionStatus.FAILED, "Executor raised an exception.", error=repr(exc))

    if res.status is not ExecutionStatus.SUCCESS:
        return _Attempt(res.status, res.message, result=res)

    # 6. after-validation (before committing)
    post_errors = validate_after(op, working, res.df, target_column=target_column)
    if post_errors:
        return _Attempt(
            ExecutionStatus.FAILED,
            "Post-execution validation failed; the transform was NOT committed.",
            result=res,
            validation_messages=post_errors,
            error="post_validation_failed",
        )

    return _Attempt(ExecutionStatus.SUCCESS, res.message, result=res, commit=True)


def _make_record(
    op: CleaningOperation,
    attempt: _Attempt,
    *,
    execution_id: str,
    approved_flag: bool,
    started: dt.datetime,
    working_before: pd.DataFrame,
    frame_after: pd.DataFrame,
) -> OperationExecution:
    result = attempt.result
    before_stats = {
        c: column_statistics(working_before, c)
        for c in op.target_columns
        if c in working_before.columns
    }
    after_stats = {
        c: column_statistics(frame_after, c) for c in op.target_columns if c in frame_after.columns
    }
    return OperationExecution(
        execution_id=execution_id,
        operation_id=op.operation_id,
        operation_type=op.operation_type,
        operation_category=op.category,
        plan_status=op.status,
        approved=approved_flag,
        status=attempt.status,
        message=attempt.message,
        source_finding_id=op.source_finding_id,
        target_columns=list(op.target_columns),
        rows_before=len(working_before),
        rows_after=len(frame_after),
        columns_before=working_before.shape[1],
        columns_after=frame_after.shape[1],
        affected_rows=result.affected_rows if result else None,
        values_changed=result.values_changed if result else None,
        columns_added=result.columns_added if result else [],
        columns_removed=result.columns_removed if result else [],
        before_statistics=before_stats,
        after_statistics=after_stats,
        fit_details=result.fit_details if result else {},
        parameters_used=result.parameters_used if result else {},
        validation_passed=attempt.status is ExecutionStatus.SUCCESS,
        validation_messages=attempt.validation_messages,
        error=attempt.error,
        started_at=started,
        completed_at=dt.datetime.now(dt.UTC),
    )


def _run_operations(
    df: pd.DataFrame,
    plan: CleaningPlan,
    *,
    approved: set[str],
    auto_recommended: bool,
    ctx: ExecutionContext,
    overrides: dict[str, dict[str, Any]],
    execution_id: str,
) -> tuple[pd.DataFrame, list[OperationExecution]]:
    working = df.copy()
    records: list[OperationExecution] = []
    supported = set(EXECUTORS)

    for op in plan.operations:
        started = dt.datetime.now(dt.UTC)
        approved_flag = op.operation_id in approved
        attempt = _decide(
            op,
            working,
            approved_flag=approved_flag,
            approved=approved,
            auto_recommended=auto_recommended,
            ctx=ctx,
            overrides=overrides,
            supported=supported,
            target_column=plan.target_column,
        )
        committed = attempt.commit and attempt.result is not None
        frame_after = attempt.result.df if committed and attempt.result else working
        records.append(
            _make_record(
                op,
                attempt,
                execution_id=execution_id,
                approved_flag=approved_flag,
                started=started,
                working_before=working,
                frame_after=frame_after,
            )
        )
        if committed and attempt.result is not None:
            working = attempt.result.df

    return working, records


def _report_status(records: list[OperationExecution]) -> ExecutionReportStatus:
    attempted = [
        r
        for r in records
        if r.status in (ExecutionStatus.SUCCESS, ExecutionStatus.FAILED, ExecutionStatus.ABORTED)
    ]
    if not attempted:
        return ExecutionReportStatus.NOTHING_EXECUTED
    if any(r.status in (ExecutionStatus.FAILED, ExecutionStatus.ABORTED) for r in attempted):
        return ExecutionReportStatus.COMPLETED_WITH_FAILURES
    return ExecutionReportStatus.COMPLETED


def _build_report(
    *,
    plan: CleaningPlan,
    records: list[OperationExecution],
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    profile: DatasetProfile | None,
    target_column: str | None,
    approved: set[str],
    auto_recommended: bool,
    execution_id: str,
    plan_fingerprint: str,
    raw_dataset_id: str,
    raw_sha: str | None,
    output_reference: Any,
) -> CleaningExecutionReport:
    succeeded = sum(1 for r in records if r.status is ExecutionStatus.SUCCESS)
    skipped_n = sum(1 for r in records if r.status is ExecutionStatus.SKIPPED)
    failed_n = sum(
        1 for r in records if r.status in (ExecutionStatus.FAILED, ExecutionStatus.ABORTED)
    )
    attempted = succeeded + failed_n

    before_profile = profile or profile_dataframe(original_df, dataset_id=raw_dataset_id)
    before_report = analyze_profile(original_df, before_profile, target_column=target_column)
    after_profile = profile_dataframe(
        cleaned_df, dataset_id=f"{raw_dataset_id}:exec-{execution_id}"
    )
    after_report = analyze_profile(cleaned_df, after_profile, target_column=target_column)

    before, after, improvements, regressions, notes = compare_quality(
        before_report, after_report, original_df, cleaned_df, target_column=target_column
    )
    comparison = QualityComparison(
        before=before, after=after, improvements=improvements, regressions=regressions, notes=notes
    )

    validation_failures = [
        f"{r.operation_id}: {msg}" for r in records for msg in r.validation_messages
    ]
    validation_summary = ValidationSummary(
        operations_validated=attempted,
        all_passed=not validation_failures,
        failures=validation_failures,
    )

    lineage = DatasetLineage(
        raw_dataset_id=raw_dataset_id,
        raw_sha256=raw_sha,
        plan_fingerprint=plan_fingerprint,
        planner_version=plan.planner_version,
        quality_engine_version=plan.based_on_quality_engine_version,
        steps=[
            LineageStep(
                index=i,
                operation_id=r.operation_id,
                operation_type=r.operation_type,
                source_finding_id=r.source_finding_id,
                status=r.status,
                summary=r.message,
            )
            for i, r in enumerate(records)
        ],
        processed_dataset_id=getattr(output_reference, "dataset_id", None),
        processed_sha256=getattr(output_reference, "sha256", None),
    )

    return CleaningExecutionReport(
        dataset_id=raw_dataset_id,
        execution_id=execution_id,
        generated_at=dt.datetime.now(dt.UTC),
        plan_fingerprint=plan_fingerprint,
        planner_version=plan.planner_version,
        based_on_quality_engine_version=plan.based_on_quality_engine_version,
        target_column=target_column,
        status=_report_status(records),
        approved_operation_ids=sorted(approved),
        auto_execute_recommended=auto_recommended,
        operations_attempted=attempted,
        operations_succeeded=succeeded,
        operations_skipped=skipped_n,
        operations_failed=failed_n,
        rows_before=len(original_df),
        rows_after=len(cleaned_df),
        columns_before=int(original_df.shape[1]),
        columns_after=int(cleaned_df.shape[1]),
        operations=records,
        output_dataset_reference=output_reference,
        validation_summary=validation_summary,
        lineage=lineage,
        before_quality_summary=before,
        after_quality_summary=after,
        quality_comparison=comparison,
    )


def execute_dataframe(
    df: pd.DataFrame,
    plan: CleaningPlan,
    *,
    approved_operation_ids: Iterable[str] | None = None,
    auto_execute_recommended: bool = False,
    profile: DatasetProfile | None = None,
    target_column: str | None = None,
    context: ExecutionContext | None = None,
    operation_parameter_overrides: dict[str, dict[str, Any]] | None = None,
) -> CleaningExecutionResult:
    """Execute an approved plan against an in-memory DataFrame.

    ``df`` is never mutated. Returns the report plus the cleaned frame.
    """
    ctx = context or ExecutionContext()
    approved = set(approved_operation_ids or ())
    overrides = operation_parameter_overrides or {}
    target = target_column if target_column is not None else plan.target_column
    if target is not None and target not in df.columns:
        raise ValueError(f"target_column {target!r} is not a column in the dataset")

    plan_fp = _plan_fingerprint(plan)
    raw_sha = _df_sha(df)
    execution_id = _execution_id(plan_fp, approved, ctx, auto_execute_recommended, raw_sha)

    cleaned, records = _run_operations(
        df,
        plan,
        approved=approved,
        auto_recommended=auto_execute_recommended,
        ctx=ctx,
        overrides=overrides,
        execution_id=execution_id,
    )
    report = _build_report(
        plan=plan,
        records=records,
        original_df=df,
        cleaned_df=cleaned,
        profile=profile,
        target_column=target,
        approved=approved,
        auto_recommended=auto_execute_recommended,
        execution_id=execution_id,
        plan_fingerprint=plan_fp,
        raw_dataset_id=plan.dataset_id,
        raw_sha=raw_sha,
        output_reference=None,
    )
    return CleaningExecutionResult(report=report, cleaned=cleaned)


def execute_cleaning(
    reference: DatasetReference,
    plan: CleaningPlan,
    *,
    approved_operation_ids: Iterable[str] | None = None,
    auto_execute_recommended: bool = False,
    profile: DatasetProfile | None = None,
    target_column: str | None = None,
    context: ExecutionContext | None = None,
    operation_parameter_overrides: dict[str, dict[str, Any]] | None = None,
    processed_store: ProcessedDataStore | None = None,
) -> CleaningExecutionReport:
    """Execute an approved plan against an ingested dataset.

    Loads the immutable raw copy read-only, executes approved operations on
    a derived DataFrame, writes a processed dataset version, and returns a
    full :class:`CleaningExecutionReport`.
    """
    ctx = context or ExecutionContext()
    approved = set(approved_operation_ids or ())
    overrides = operation_parameter_overrides or {}
    target = target_column if target_column is not None else plan.target_column

    original_df = load_dataframe(reference)  # read-only
    if target is not None and target not in original_df.columns:
        raise ValueError(f"target_column {target!r} is not a column in the dataset")

    plan_fp = _plan_fingerprint(plan)
    execution_id = _execution_id(plan_fp, approved, ctx, auto_execute_recommended, reference.sha256)

    cleaned, records = _run_operations(
        original_df,
        plan,
        approved=approved,
        auto_recommended=auto_execute_recommended,
        ctx=ctx,
        overrides=overrides,
        execution_id=execution_id,
    )

    store = processed_store or ProcessedDataStore.default()
    output_reference = store.save(
        cleaned,
        parent_dataset_id=reference.dataset_id,
        execution_id=execution_id,
        plan_fingerprint=plan_fp,
        original_filename=reference.original_filename,
    )

    report = _build_report(
        plan=plan,
        records=records,
        original_df=original_df,
        cleaned_df=cleaned,
        profile=profile,
        target_column=target,
        approved=approved,
        auto_recommended=auto_execute_recommended,
        execution_id=execution_id,
        plan_fingerprint=plan_fp,
        raw_dataset_id=reference.dataset_id,
        raw_sha=reference.sha256,
        output_reference=output_reference,
    )
    store.write_execution_report(
        reference.dataset_id, execution_id, report.model_dump_json(indent=2)
    )
    return report
