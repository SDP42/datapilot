"""Plan proposal for a ``high_skew`` finding.

A log transform is only proposed when the column is *strictly positive*
(verified from the profile's minimum). Otherwise the planner proposes a
review with transform candidates that tolerate zero/negative values
(log1p, Yeo-Johnson, quantile) — it never blindly recommends plain log.
"""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityFinding

from ..context import PlanContext
from ..models import (
    CleaningOperation,
    OperationCategory,
    OperationStatus,
    OperationType,
)

_LEAKAGE_RISK = (
    "Transform parameters (and the decision to transform at all) must be chosen using the "
    "training split only."
)


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    column = finding.columns[0]
    skew = finding.observed.get("skewness")

    col_profile = ctx.column_profile(column)
    minimum = (
        col_profile.numeric_stats.minimum if col_profile and col_profile.numeric_stats else None
    )
    strictly_positive = minimum is not None and minimum > 0

    if strictly_positive:
        return [
            CleaningOperation(
                operation_id=f"{OperationType.TRANSFORM_DISTRIBUTION_LOG.value}:{column}",
                operation_type=OperationType.TRANSFORM_DISTRIBUTION_LOG,
                category=OperationCategory.DATA_TRANSFORMATION,
                status=OperationStatus.REVIEW_REQUIRED,
                status_reason=(
                    "The column is strictly positive, so a log transform is mathematically "
                    "valid, but whether it helps depends on the downstream model — evaluate "
                    "with and without."
                ),
                target_columns=[column],
                addresses_finding_type=FindingType.HIGH_SKEW,
                source_finding_id=finding.finding_id,
                problem_summary=f"Column '{column}' is highly skewed (skewness = {skew}).",
                proposed_action=(
                    f"Consider replacing '{column}' with its natural logarithm (or keeping both). "
                    f"Verified: minimum value is {minimum} (> 0), so log is defined for all rows."
                ),
                rationale=(
                    "Log compresses a long right tail, often making a feature more symmetric and "
                    "linear relationships easier to model."
                ),
                assumptions=[
                    "All values are and will remain strictly positive.",
                    "A more symmetric feature benefits the intended model.",
                ],
                risks=[
                    "Changes feature interpretability and units.",
                    "Not always beneficial for tree-based models.",
                    _LEAKAGE_RISK,
                ],
                parameters={
                    "transform": "log",
                    "base": "e",
                    "requires_strictly_positive": True,
                    "observed_minimum": minimum,
                },
                affected_rows=None,
                affected_percentage=None,
                confidence=0.5,
                requires_train_test_split_awareness=True,
            )
        ]

    reason_detail = (
        f"minimum value is {minimum} (not strictly positive)"
        if minimum is not None
        else "the column minimum is unknown without a DatasetProfile"
    )
    return [
        CleaningOperation(
            operation_id=f"{OperationType.REVIEW_DISTRIBUTION_TRANSFORM.value}:{column}",
            operation_type=OperationType.REVIEW_DISTRIBUTION_TRANSFORM,
            category=OperationCategory.DATA_TRANSFORMATION,
            status=OperationStatus.REVIEW_REQUIRED,
            status_reason=(
                f"A plain log transform is not safe because {reason_detail}. Candidate "
                "transforms that tolerate zero/negative values need a human choice."
            ),
            target_columns=[column],
            addresses_finding_type=FindingType.HIGH_SKEW,
            source_finding_id=finding.finding_id,
            problem_summary=f"Column '{column}' is highly skewed (skewness = {skew}).",
            proposed_action=(
                f"Review a distribution transform for '{column}'. Plain log is NOT applicable "
                f"({reason_detail}). Consider log1p (if values ≥ 0), Yeo-Johnson, or a quantile "
                "transform."
            ),
            rationale="Reducing skew can help some models, but the method must fit the data's range.",
            assumptions=["A more symmetric feature benefits the intended model."],
            risks=[
                "Applying log to zero/negative values produces -inf / NaN.",
                "Changes feature interpretability.",
                _LEAKAGE_RISK,
            ],
            parameters={
                "plain_log_applicable": False,
                "observed_minimum": minimum,
                "candidate_transforms": ["log1p", "yeo_johnson", "quantile"],
            },
            affected_rows=None,
            affected_percentage=None,
            confidence=None,
            requires_train_test_split_awareness=True,
        )
    ]
