"""Plan proposal for a ``class_imbalance`` finding.

This produces a MODELING_RECOMMENDATION, not a data transformation. The
dataset is not changed at this stage — imbalance is handled during model
training (class weights, training-split resampling, threshold tuning).
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
    target = finding.columns[0]
    obs = finding.observed
    return [
        CleaningOperation(
            operation_id=f"{OperationType.RECOMMEND_IMBALANCE_STRATEGY.value}:{target}",
            operation_type=OperationType.RECOMMEND_IMBALANCE_STRATEGY,
            category=OperationCategory.MODELING_RECOMMENDATION,
            status=OperationStatus.REVIEW_REQUIRED,
            status_reason=(
                "Class imbalance is a modelling concern, not a data defect. It is recorded here "
                "so the ML phase can plan for it; no cleaning transformation is proposed."
            ),
            target_columns=[target],
            addresses_finding_type=FindingType.CLASS_IMBALANCE,
            source_finding_id=finding.finding_id,
            problem_summary=(
                f"Target '{target}' is imbalanced "
                f"(ratio {obs.get('imbalance_ratio')}, minority {obs.get('minority_percentage')}%)."
            ),
            proposed_action=(
                "During MODEL TRAINING (not now, not on the raw data): consider class weights, "
                "resampling applied to the training split only (e.g. SMOTE / random under- or "
                "over-sampling), and decision-threshold tuning. Evaluate with precision, recall, "
                "F1 and PR-AUC rather than accuracy."
            ),
            rationale=(
                "On imbalanced targets a model can score high accuracy by always predicting the "
                "majority class while being useless for the minority class."
            ),
            assumptions=[
                (
                    "The observed class ratio is either representative of deployment, or will "
                    "be corrected for deliberately."
                ),
            ],
            risks=[
                (
                    "Resampling the whole dataset (instead of only the training split) leaks "
                    "information and inflates reported metrics."
                ),
                "Oversampling can cause overfitting to the minority examples.",
            ],
            parameters={
                "is_data_transformation": False,
                "class_distribution": obs.get("class_distribution"),
                "candidate_strategies": [
                    "class_weights",
                    "resample_training_split_only",
                    "decision_threshold_tuning",
                ],
                "recommended_metrics": ["precision", "recall", "f1", "pr_auc"],
            },
            affected_rows=None,
            affected_percentage=None,
            confidence=None,
            requires_train_test_split_awareness=True,
        )
    ]
