"""Automated Problem Understanding (Phase 5) — deterministic, analysis-only.

Phase 5 turns a dataset + an explicit objective into a structured
:class:`ProblemSpec`: what ML task the data describes, what the target is,
which metrics make sense, and whether the problem is feasible.

**Phase 5.1 (this increment) is the contract + foundation only.**
:func:`understand_problem` validates an explicit
:class:`ProblemUnderstandingRequest` and returns a ``ProblemSpec`` whose
sections are all ``not_yet_inferred`` — no target, task type, metric, or
feasibility verdict is produced yet, and the user's objective is never
inferred from the data.

    from data_engine.problem_understanding import (
        ProblemUnderstandingRequest,
        understand_problem,
    )

    spec = understand_problem(
        ProblemUnderstandingRequest(dataset_id="sales", objective="predict churn")
    )
    payload = spec.model_dump(mode="json")
"""

from __future__ import annotations

from datapilot.contracts import ColumnType

from .feasibility_assessment import assess_feasibility
from .metrics_recommendation import recommend_metrics
from .models import (
    PROBLEM_UNDERSTANDING_ENGINE_VERSION,
    CandidateMetrics,
    FeasibilityAssessment,
    ObjectiveMatchKind,
    ProblemSpec,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetCandidate,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
)
from .target_identification import (
    HIGH_UNIQUE_ID_THRESHOLD,
    TARGET_SELECTION_MARGIN,
    identify_target,
)
from .task_type_inference import NUMERIC_CLASS_MAX, infer_task_type
from .understanding import understand_problem

__all__ = [
    "HIGH_UNIQUE_ID_THRESHOLD",
    "NUMERIC_CLASS_MAX",
    "PROBLEM_UNDERSTANDING_ENGINE_VERSION",
    "TARGET_SELECTION_MARGIN",
    "CandidateMetrics",
    "ColumnType",
    "FeasibilityAssessment",
    "ObjectiveMatchKind",
    "ProblemSpec",
    "ProblemUnderstandingRequest",
    "ProblemUnderstandingStatus",
    "TargetCandidate",
    "TargetIdentification",
    "TaskType",
    "TaskTypeInference",
    "assess_feasibility",
    "identify_target",
    "infer_task_type",
    "recommend_metrics",
    "understand_problem",
]
