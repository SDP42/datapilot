"""Plan proposal for a ``potential_outliers`` finding.

The planner never proposes deleting or capping flagged values. It creates
an INVESTIGATION operation that keeps two ideas apart:

* "outlier detected"  — true, the value is far from the rest
* "outlier is an error" — unknown, needs domain context
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


def plan(finding: QualityFinding, ctx: PlanContext) -> list[CleaningOperation]:
    column = finding.columns[0]
    obs = finding.observed
    return [
        CleaningOperation(
            operation_id=f"{OperationType.REVIEW_OUTLIERS.value}:{column}",
            operation_type=OperationType.REVIEW_OUTLIERS,
            category=OperationCategory.INVESTIGATION,
            status=OperationStatus.REVIEW_REQUIRED,
            status_reason=(
                "An outlier is a value far from the rest of the distribution — that is NOT the "
                "same as an error. Real data has legitimate extremes. Any treatment (capping, "
                "removal, separate modelling) needs a domain reason, so this stays a review task."
            ),
            target_columns=[column],
            addresses_finding_type=FindingType.POTENTIAL_OUTLIERS,
            source_finding_id=finding.finding_id,
            problem_summary=(
                f"Column '{column}' has {finding.affected_rows} value(s) outside the IQR fence "
                f"[{obs.get('lower_fence')}, {obs.get('upper_fence')}]."
            ),
            proposed_action=(
                f"Investigate the flagged values in '{column}' (min {obs.get('min_outlier')}, "
                f"max {obs.get('max_outlier')}). Confirm whether they are data-entry errors, "
                "unit mistakes, or genuine extremes. Do NOT delete, clip, or replace them as "
                "part of automated cleaning."
            ),
            rationale=(
                "Extreme values are often the most informative part of a dataset. Removing them "
                "without cause throws away signal and biases every downstream estimate."
            ),
            assumptions=["None — this is a review task, not a transformation."],
            risks=[
                "Treating outliers as errors removes real signal and understates variance.",
                "Keeping true data-entry errors distorts training just as much.",
            ],
            parameters={
                "method": "iqr",
                "fences": {"lower": obs.get("lower_fence"), "upper": obs.get("upper_fence")},
                "outlier_detected": True,
                "confirmed_error": False,
                "proposed_treatment": None,
            },
            affected_rows=finding.affected_rows,
            affected_percentage=finding.affected_percentage,
            confidence=None,
            requires_train_test_split_awareness=False,
        )
    ]
