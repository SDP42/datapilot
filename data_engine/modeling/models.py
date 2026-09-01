"""Structured contract for model development / modeling (Phase 7).

Pydantic v2, JSON round-trip safe, JSON-primitive only — no DataFrame,
NumPy array, SciPy object, fitted estimator, prediction array, figure, or
file handle.

This module defines the **contract** only. Later Phase-7 increments
(model readiness, data-split planning, candidate model families, training,
evaluation, model selection) will *populate* the nested sections below;
**Phase 7.1 infers nothing** and never fabricates a model name, a split
ratio, a metric, a hyperparameter, a fitted estimator, or a readiness
verdict.

Design rule (mirrors Phases 5 and 6): the distinction between **known**
(``completed``), **tried and impossible** (``unavailable``), and **not
attempted yet** (``not_yet_inferred``) is explicit — an unknown value is
``None`` / ``[]`` / ``False`` plus a reason, never a fabricated value.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

MODEL_ENGINE_VERSION = "1"


class ModelingStatus(str, Enum):
    """Lifecycle of one piece of the modeling understanding."""

    NOT_YET_INFERRED = "not_yet_inferred"  # no Phase-7 increment has attempted it
    COMPLETED = "completed"  # a later increment produced a value
    UNAVAILABLE = "unavailable"  # a later increment attempted it and could not (see `reason`)


class ModelFamily(str, Enum):
    """Categories of model a later Phase-7 increment may consider.

    Declarative only — Phase 7.1 trains, recommends, and names **none** of
    these.
    """

    LINEAR = "linear"
    TREE_BASED = "tree_based"
    DISTANCE_BASED = "distance_based"
    PROBABILISTIC = "probabilistic"
    ENSEMBLE = "ensemble"
    NEURAL = "neural"


class ModelReadiness(BaseModel):
    """Whether the data / pipeline is structurally ready for modeling.

    Populated by a later Phase-7 increment. Phase 7.1 leaves it at
    ``not_yet_inferred``.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why readiness is unavailable; None once assessed."
    )
    notes: list[str] = Field(default_factory=list)


class DataSplitPlan(BaseModel):
    """Future train / validation / test split decisions.

    Populated by a later Phase-7 increment. Phase 7.1 chooses no strategy
    and performs no split.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why the split plan is unavailable; None once produced."
    )
    notes: list[str] = Field(default_factory=list)


class ModelCandidates(BaseModel):
    """Future candidate model families.

    Populated by a later Phase-7 increment. Phase 7.1 recommends and
    trains nothing; ``candidates`` is empty.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why candidates are unavailable; None once produced."
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="Candidate model-family identifiers, empty until inferred.",
    )
    notes: list[str] = Field(default_factory=list)


class TrainingOutcome(BaseModel):
    """Future model-training execution.

    Populated by a later Phase-7 increment. Phase 7.1 trains nothing.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why training is unavailable; None once run."
    )
    notes: list[str] = Field(default_factory=list)


class EvaluationResults(BaseModel):
    """Future evaluation / metric results.

    Populated by a later Phase-7 increment. Phase 7.1 calculates no metric.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why evaluation is unavailable; None once produced."
    )
    notes: list[str] = Field(default_factory=list)


class ModelSelection(BaseModel):
    """Future model comparison and selection.

    Populated by a later Phase-7 increment. Phase 7.1 selects no model.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why a selection is unavailable; None once produced."
    )
    notes: list[str] = Field(default_factory=list)


class ModelingRequest(BaseModel):
    """The explicit input to :func:`understand_modeling`.

    Carries **dataset identity** (reusing the ``dataset_id`` /
    ``dataset_version_id`` convention shared by ``DatasetProfile`` /
    ``QualityReport`` / ``EDAReport`` / ``ProblemSpec`` /
    ``FeatureEngineeringSpec``) and, optionally, the **explicit** user
    objective. The objective is preserved verbatim and never inferred
    from column names or data content.
    """

    dataset_id: str = Field(description="Identifier of the dataset being modelled.")
    dataset_version_id: str | None = Field(
        default=None, description="Registered DatasetVersion id, when the caller has one."
    )
    objective: str | None = Field(
        default=None,
        description="Plain-language analytical goal, exactly as the user supplied it.",
    )


class ModelingSpec(BaseModel):
    """The structured answer to 'how should this problem be modelled?'.

    Phase 7.1 produces a spec whose overall ``status`` and every nested
    section are ``not_yet_inferred``; the ``dataset_id`` /
    ``dataset_version_id`` / ``objective`` fields echo the request. Later
    increments fill in ``readiness`` / ``split`` / ``candidates`` /
    ``training`` / ``evaluation`` / ``selection`` in place, additively.
    """

    # ``model_engine_version`` intentionally uses the ``model_`` prefix for
    # naming consistency with the other engine-version fields; opt out of
    # Pydantic's protected ``model_`` namespace so it is a plain data field.
    model_config = ConfigDict(protected_namespaces=())

    model_engine_version: str = MODEL_ENGINE_VERSION

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = Field(
        default=None, description="The user's objective, verbatim; None if none was supplied."
    )
    objective_provided: bool = Field(
        description="True iff the request carried a non-blank objective string (after strip)."
    )

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(default=None, description="Explains a non-completed overall status.")

    readiness: ModelReadiness = Field(default_factory=ModelReadiness)
    split: DataSplitPlan = Field(default_factory=DataSplitPlan)
    candidates: ModelCandidates = Field(default_factory=ModelCandidates)
    training: TrainingOutcome = Field(default_factory=TrainingOutcome)
    evaluation: EvaluationResults = Field(default_factory=EvaluationResults)
    selection: ModelSelection = Field(default_factory=ModelSelection)

    notes: list[str] = Field(default_factory=list)
