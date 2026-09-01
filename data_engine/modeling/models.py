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


class DataSplitStrategy(str, Enum):
    """How a later stage should split the data.

    A **recommendation** only — Phase 7.2 performs no split.
    """

    RANDOM_HOLDOUT = "random_holdout"
    STRATIFIED_HOLDOUT = "stratified_holdout"
    TIME_ORDERED_HOLDOUT = "time_ordered_holdout"
    NOT_APPLICABLE = "not_applicable"


class ModelReadiness(BaseModel):
    """Whether the data / pipeline is **structurally** ready for modeling.

    Populated by
    :func:`data_engine.modeling.assess_model_readiness` (Phase 7.2). All
    fields beyond ``status`` / ``reason`` / ``notes`` are additive and
    defaulted, so a ``ModelReadiness`` serialised by Phase 7.1 still
    validates. ``ready`` means *"the available structural information is
    sufficient to proceed to the next model-development stage"* — never
    *"the dataset will produce a good model"*.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why readiness is unavailable, or why it is not ready; None when ready.",
    )
    ready: bool | None = Field(
        default=None,
        description="Structural readiness verdict; None until assessed (never a fabricated False).",
    )
    target_available: bool = Field(
        default=False, description="True iff a target column is identified (supervised tasks)."
    )
    target_usable: bool = Field(
        default=False,
        description="True iff the target column exists in the DataFrame and is non-constant / "
        "not entirely missing.",
    )
    eligible_feature_count: int = Field(
        default=0, description="Number of structurally eligible candidate feature columns."
    )
    feature_engineering_assessment_usable: bool = Field(
        default=False,
        description="True iff the Phase 6.6 assessment is completed with a boolean verdict.",
    )
    preprocessing_requirements_present: bool = Field(
        default=False,
        description="True iff Phase 6.5 identified at least one preprocessing requirement.",
    )
    sufficient_observations: bool = Field(
        default=False,
        description="True iff the row count meets the structural minimum for modeling.",
    )
    n_observations: int = Field(default=0, description="Row count of the supplied DataFrame.")
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DataSplitPlan(BaseModel):
    """A recommended train / validation / test split.

    Populated by :func:`data_engine.modeling.recommend_data_split`
    (Phase 7.2). All fields beyond ``status`` / ``reason`` / ``notes`` are
    additive and defaulted, so a ``DataSplitPlan`` serialised by Phase 7.1
    still validates. Phase 7.2 **recommends** a strategy and fractions
    only — it never shuffles, stratifies, orders, or copies the
    DataFrame.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None, description="Why the split plan is unavailable; None once produced."
    )
    strategy: DataSplitStrategy | None = Field(
        default=None, description="Recommended split strategy; None until produced."
    )
    train_fraction: float | None = Field(
        default=None, description="Recommended train fraction; None until produced."
    )
    validation_fraction: float | None = Field(
        default=None,
        description="Recommended validation fraction; None when no separate validation split "
        "is recommended.",
    )
    test_fraction: float | None = Field(
        default=None, description="Recommended test fraction; None until produced."
    )
    stratify: bool = Field(
        default=False,
        description="True iff stratifying on the target is recommended (classification only).",
    )
    preserve_temporal_order: bool = Field(
        default=False,
        description="True iff the split must preserve row order (time-series forecasting).",
    )
    shuffle: bool = Field(
        default=False, description="True iff shuffling rows before splitting is recommended."
    )
    notes: list[str] = Field(default_factory=list)


class ModelCandidate(BaseModel):
    """One structurally-suitable candidate model family.

    Produced by :func:`data_engine.modeling.generate_model_candidates`
    (Phase 7.3). ``reason`` / ``evidence`` describe **structural**
    suitability for the inferred task — never a prediction that the family
    will perform well.
    """

    family: ModelFamily
    reason: str = Field(description="The structural reason this family is worth considering.")
    evidence: list[str] = Field(
        default_factory=list, description="Deterministic supporting evidence, fixed order."
    )


class ModelCandidates(BaseModel):
    """Candidate model families for the inferred task.

    Populated by :func:`data_engine.modeling.generate_model_candidates`
    (Phase 7.3). ``candidates_detail`` / ``objective_used`` are additive
    and defaulted, so a ``ModelCandidates`` serialised by Phase 7.1 still
    validates.
    """

    status: ModelingStatus = ModelingStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why candidates are unavailable, or why a completed result is empty; "
        "None otherwise.",
    )
    candidates: list[str] = Field(
        default_factory=list,
        description="Candidate model-family identifiers, deterministically ordered; empty "
        "until inferred / when none apply.",
    )
    candidates_detail: list[ModelCandidate] = Field(
        default_factory=list,
        description="Structured per-family candidates, deterministically ordered; one entry "
        "per name in `candidates`.",
    )
    objective_used: bool = Field(
        default=False, description="True iff a non-blank objective string was supplied."
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
