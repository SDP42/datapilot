"""Plan proposals for a ``missing_values`` finding.

Strategy depends on the column's inferred type (from the profile):

* numeric      -> median imputation
* categorical  -> most-frequent (mode) imputation
* datetime     -> no safe default; review required
* unknown      -> generic "handle missing values", review required

Severity of the proposal tracks the proportion missing. Very high
missingness *additionally* surfaces a "drop the column" proposal, which
is always NOT_SAFE_TO_AUTOMATE.
"""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityFinding
from datapilot.contracts import ColumnType

from ..context import PlanContext
from ..models import (
    CleaningOperation,
    ImputationStrategy,
    OperationCategory,
    OperationStatus,
    OperationType,
)
from ..thresholds import MISSING_HIGH_PCT, MISSING_SAFE_IMPUTE_MAX_PCT

_LEAKAGE_ASSUMPTION = (
    "The imputation value must be computed on the training split only and then applied "
    "to validation/test/production data — otherwise information leaks."
)


def _impute_op(
    finding: QualityFinding,
    column: str,
    op_type: OperationType,
    strategy: ImputationStrategy | None,
    pct: float,
) -> CleaningOperation:
    safe = pct <= MISSING_SAFE_IMPUTE_MAX_PCT and strategy is not None
    status = OperationStatus.RECOMMENDED if safe else OperationStatus.REVIEW_REQUIRED
    if strategy is None:
        status_reason = (
            "No safe default strategy for this column type; a human must choose how to fill "
            "or whether to drop affected rows."
        )
    elif safe:
        status_reason = (
            f"Only {pct:.2f}% of values are missing; {strategy.value} imputation is a "
            "low-distortion, standard choice."
        )
    else:
        status_reason = (
            f"{pct:.2f}% missing is substantial: imputation noticeably reshapes the column "
            "and shrinks its variance, so confirm it is acceptable for the analysis goal."
        )

    return CleaningOperation(
        operation_id=f"{op_type.value}:{column}",
        operation_type=op_type,
        category=OperationCategory.DATA_TRANSFORMATION,
        status=status,
        status_reason=status_reason,
        target_columns=[column],
        addresses_finding_type=FindingType.MISSING_VALUES,
        source_finding_id=finding.finding_id,
        problem_summary=f"Column '{column}' has {pct:.2f}% missing values.",
        proposed_action=(
            f"Fill missing values in '{column}' using the {strategy.value} of the non-null values."
            if strategy
            else f"Decide how to handle missing values in '{column}'."
        ),
        rationale=(
            "Median is robust to skew and outliers for numeric columns; the most-frequent "
            "category is the conventional choice for categoricals."
            if strategy
            else "Column type could not be determined without a DatasetProfile."
        ),
        assumptions=(
            [
                "Values are missing at random with respect to the analysis target.",
                _LEAKAGE_ASSUMPTION,
            ]
            if strategy
            else ["A DatasetProfile is needed to choose a concrete strategy."]
        ),
        risks=[
            "Imputation invents values: it reduces variance and can bias models.",
            "If missingness is informative, filling it hides signal — consider a 'was_missing' flag.",
        ],
        strategy=strategy,
        parameters={"strategy": strategy.value if strategy else None, "fit_on": "train_split"},
        affected_rows=finding.affected_rows,
        affected_percentage=pct,
        confidence=0.6 if safe else 0.4,
        requires_train_test_split_awareness=True,
    )


def _drop_column_op(finding: QualityFinding, column: str, pct: float) -> CleaningOperation:
    return CleaningOperation(
        operation_id=f"{OperationType.DROP_HIGH_MISSING_COLUMN.value}:{column}",
        operation_type=OperationType.DROP_HIGH_MISSING_COLUMN,
        category=OperationCategory.DATA_TRANSFORMATION,
        status=OperationStatus.NOT_SAFE_TO_AUTOMATE,
        status_reason=(
            "Dropping a column solely because it has many missing values can discard a "
            "predictive or legally-required field. This needs an explicit human decision."
        ),
        target_columns=[column],
        addresses_finding_type=FindingType.MISSING_VALUES,
        source_finding_id=finding.finding_id,
        problem_summary=f"Column '{column}' is {pct:.2f}% missing.",
        proposed_action=f"Consider removing column '{column}' from the working dataset.",
        rationale=(
            "A column that is mostly empty often carries little usable signal and complicates "
            "imputation, but that is not guaranteed."
        ),
        assumptions=["The little data present is not disproportionately important."],
        risks=[
            "The column may still be predictive where it is present.",
            "The pattern of missingness itself may be informative.",
        ],
        parameters={"missing_percentage": pct},
        affected_percentage=pct,
        confidence=None,
        requires_train_test_split_awareness=False,
    )


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    column = finding.columns[0]
    pct = finding.affected_percentage if finding.affected_percentage is not None else 0.0
    fully_missing = bool(finding.observed.get("fully_missing", False))

    col_profile = ctx.column_profile(column)
    inferred = col_profile.inferred_type if col_profile else None

    op_type, strategy = OperationType.HANDLE_MISSING_VALUES, None
    if inferred is ColumnType.NUMERIC:
        op_type, strategy = OperationType.IMPUTE_MISSING_NUMERIC, ImputationStrategy.MEDIAN
    elif inferred in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
        op_type, strategy = OperationType.IMPUTE_MISSING_CATEGORICAL, ImputationStrategy.MODE
    elif inferred is ColumnType.DATETIME:
        op_type, strategy = OperationType.IMPUTE_MISSING_DATETIME, None

    operations: list[CleaningOperation] = []
    if not fully_missing:
        operations.append(_impute_op(finding, column, op_type, strategy, pct))
    if fully_missing or pct >= MISSING_HIGH_PCT:
        operations.append(_drop_column_op(finding, column, pct))
    return operations
