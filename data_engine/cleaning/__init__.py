"""Cleaning **planning** layer — proposals only, no execution.

Consumes a :class:`QualityReport` (and, ideally, the ``DatasetProfile``)
and produces a machine-readable :class:`CleaningPlan` of typed,
explainable, auditable :class:`CleaningOperation` proposals. Every
operation carries a safety ``status`` (``recommended`` /
``review_required`` / ``not_safe_to_automate``) and points back to the
finding that triggered it.

Nothing here changes data. Execution is a separate, later phase.

    from data_engine.quality import analyze_quality
    from data_engine.cleaning import plan_cleaning

    report = analyze_quality(reference, target_column="churned")
    plan = plan_cleaning(report, profile=profile)
    payload = plan.model_dump(mode="json")
"""

from __future__ import annotations

from .models import (
    CleaningOperation,
    CleaningPlan,
    CleaningPlanSummary,
    ImputationStrategy,
    OperationCategory,
    OperationStatus,
    OperationType,
)
from .planner import (
    RULES,
    plan_cleaning,
    plan_from_dataframe,
    supported_finding_types,
)

__all__ = [
    "RULES",
    "CleaningOperation",
    "CleaningPlan",
    "CleaningPlanSummary",
    "ImputationStrategy",
    "OperationCategory",
    "OperationStatus",
    "OperationType",
    "plan_cleaning",
    "plan_from_dataframe",
    "supported_finding_types",
]
