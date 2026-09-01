"""Structured contract for automated problem understanding (Phase 5).

Pydantic v2, JSON round-trip safe, JSON-primitive only — no DataFrame,
NumPy array, SciPy object, model instance, figure, or file handle.

This module defines the **contract** only. The four later Phase-5
increments (target identification, task-type inference, candidate
metrics, feasibility checks) will *populate* the nested sections below;
Phase 5.1 leaves every inferred value at
``status = not_yet_inferred`` and never fabricates a task type, a target,
a metric, or a feasibility verdict.

Design rule (mirrors the rest of the engine): the distinction between
**known** (`completed`), **tried and impossible** (`unavailable`), and
**not attempted yet** (`not_yet_inferred`) is explicit — an unknown value
is ``None`` plus a reason, never a fake ``"classification"`` / ``0`` /
``False``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

PROBLEM_UNDERSTANDING_ENGINE_VERSION = "1"


class ProblemUnderstandingStatus(str, Enum):
    """Lifecycle of one piece of the problem understanding."""

    NOT_YET_INFERRED = "not_yet_inferred"  # no Phase-5 increment has attempted it
    COMPLETED = "completed"  # a later increment inferred a value
    UNAVAILABLE = "unavailable"  # a later increment attempted it and could not (see `reason`)


class TaskType(str, Enum):
    """The supervised/unsupervised ML task a dataset + objective describes.

    Defined now so the contract is stable; **not populated** by Phase 5.1.
    """

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    OTHER = "other"


class TargetIdentification(BaseModel):
    """Which column(s) the model should predict. Populated in a later increment."""

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why the value is unavailable; None otherwise."
    )
    target_column: str | None = Field(
        default=None, description="The identified target column; None until inferred."
    )
    candidate_columns: list[str] = Field(
        default_factory=list, description="Columns considered as possible targets, ranked later."
    )
    notes: list[str] = Field(default_factory=list)


class TaskTypeInference(BaseModel):
    """The inferred ML task type. Populated in a later increment."""

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = None
    task_type: TaskType | None = Field(
        default=None, description="The inferred task type; None until inferred."
    )
    notes: list[str] = Field(default_factory=list)


class CandidateMetrics(BaseModel):
    """Evaluation metrics that would make sense for the task. Populated later."""

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = None
    primary_metric: str | None = Field(
        default=None, description="Recommended primary metric; None until inferred."
    )
    metrics: list[str] = Field(
        default_factory=list, description="All candidate metric names; empty until inferred."
    )
    notes: list[str] = Field(default_factory=list)


class FeasibilityAssessment(BaseModel):
    """Whether the problem looks solvable from the available data. Populated later."""

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = None
    feasible: bool | None = Field(
        default=None, description="True/False once assessed; None until then (never a fake False)."
    )
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProblemUnderstandingRequest(BaseModel):
    """The explicit input to :func:`understand_problem`.

    Carries **dataset identity** (reusing the ``dataset_id`` /
    ``dataset_version_id`` convention shared by ``DatasetProfile`` /
    ``QualityReport`` / ``EDAReport``) and, optionally, the **explicit**
    user objective. The objective is never inferred from column names or
    data content.
    """

    dataset_id: str = Field(description="Identifier of the dataset being analysed.")
    dataset_version_id: str | None = Field(
        default=None, description="Registered DatasetVersion id, when the caller has one."
    )
    objective: str | None = Field(
        default=None,
        description="Plain-language analytical goal, exactly as the user supplied it.",
    )


class ProblemSpec(BaseModel):
    """The structured answer to 'what ML problem is this?'.

    Phase 5.1 produces a spec whose overall ``status`` and every nested
    section are ``not_yet_inferred``; the ``dataset_id`` /
    ``dataset_version_id`` / ``objective`` fields echo the request. Later
    increments fill in ``target`` / ``task_type`` / ``metrics`` /
    ``feasibility`` in place, additively.
    """

    problem_understanding_engine_version: str = PROBLEM_UNDERSTANDING_ENGINE_VERSION

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = Field(
        default=None, description="The user's objective, verbatim; None if none was supplied."
    )
    objective_provided: bool = Field(
        description="True iff the request carried a non-empty objective string."
    )

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Explains a non-completed overall status.",
    )

    target: TargetIdentification = Field(default_factory=TargetIdentification)
    task_type: TaskTypeInference = Field(default_factory=TaskTypeInference)
    metrics: CandidateMetrics = Field(default_factory=CandidateMetrics)
    feasibility: FeasibilityAssessment = Field(default_factory=FeasibilityAssessment)

    notes: list[str] = Field(default_factory=list)
