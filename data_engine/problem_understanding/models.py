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

from datapilot.contracts import ColumnType

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


class ObjectiveMatchKind(str, Enum):
    """How strongly a column name matched the (verbatim) user objective."""

    EXACT = "exact"  # the full normalised column name appears as a phrase in the objective
    NORMALIZED = "normalized"  # separator-insensitive substring / all-token match
    TOKEN = "token"  # a significant column-name token appears as an objective token
    NONE = "none"  # no deterministic match, or no objective supplied


class TargetCandidate(BaseModel):
    """One ranked candidate target column, with the evidence for its rank.

    ``score`` is a **deterministic ranking score** — a sum of documented
    structural / objective components. It is **not** a probability and
    **not** a confidence percentage.
    """

    column: str
    rank: int = Field(description="1-based rank within the candidate list; unique and sequential.")
    score: float = Field(description="Deterministic ranking score (not a probability).")

    column_type: ColumnType
    n_observations: int = Field(description="Non-null values.")
    n_missing: int
    missing_fraction: float
    n_unique: int = Field(description="Distinct non-null values.")
    unique_fraction: float = Field(
        description="n_unique / n_observations; 0.0 when no observations."
    )

    identifier_like: bool = Field(
        description="Column name / behaviour looks like a row identifier (penalised as a target)."
    )
    objective_match: ObjectiveMatchKind = ObjectiveMatchKind.NONE
    reasons: list[str] = Field(
        default_factory=list, description="Human-readable evidence for the score, in a fixed order."
    )


class TargetIdentification(BaseModel):
    """Which column(s) the model should predict.

    Populated by :func:`data_engine.problem_understanding.identify_target`
    (Phase 5.2). ``candidates`` / ``objective_used`` are additive and
    defaulted, so a ``TargetIdentification`` serialised by Phase 5.1 still
    validates.
    """

    status: ProblemUnderstandingStatus = ProblemUnderstandingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description=(
            "Why no single target was pinned (ambiguity / no objective), or why the result "
            "is unavailable. None when a single target_column was identified."
        ),
    )
    target_column: str | None = Field(
        default=None,
        description="The single identified target; None when the evidence is ambiguous.",
    )
    candidate_columns: list[str] = Field(
        default_factory=list,
        description="Candidate target column names, best-first (mirrors `candidates`).",
    )
    candidates: list[TargetCandidate] = Field(
        default_factory=list, description="Ranked candidates with per-candidate evidence."
    )
    objective_used: bool = Field(
        default=False, description="True iff a non-blank objective string was supplied."
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
