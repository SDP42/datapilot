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

from .models import (
    PROBLEM_UNDERSTANDING_ENGINE_VERSION,
    CandidateMetrics,
    FeasibilityAssessment,
    ProblemSpec,
    ProblemUnderstandingRequest,
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
)
from .understanding import understand_problem

__all__ = [
    "PROBLEM_UNDERSTANDING_ENGINE_VERSION",
    "CandidateMetrics",
    "FeasibilityAssessment",
    "ProblemSpec",
    "ProblemUnderstandingRequest",
    "ProblemUnderstandingStatus",
    "TargetIdentification",
    "TaskType",
    "TaskTypeInference",
    "understand_problem",
]
