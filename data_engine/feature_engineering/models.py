"""Structured contract for automated feature engineering (Phase 6).

Pydantic v2, JSON round-trip safe, JSON-primitive only — no DataFrame,
NumPy array, SciPy object, model instance, figure, or file handle.

This module defines the **contract** only. Later Phase-6 increments
(feature inventory, transformation recommendations, feature selection,
preprocessing requirements, feature-engineering feasibility) will
*populate* the nested sections below; **Phase 6.1 infers nothing** and
never fabricates a feature name, a transformation, an encoder, a scaler,
an importance score, or a feasibility verdict.

Design rule (mirrors Phase 5): the distinction between **known**
(``completed``), **tried and impossible** (``unavailable``), and **not
attempted yet** (``not_yet_inferred``) is explicit — an unknown value is
``None`` / ``[]`` / ``False`` plus a reason, never a fabricated name.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from datapilot.contracts import ColumnType

FEATURE_ENGINEERING_ENGINE_VERSION = "1"


class FeatureEngineeringStatus(str, Enum):
    """Lifecycle of one piece of the feature-engineering understanding."""

    NOT_YET_INFERRED = "not_yet_inferred"  # no Phase-6 increment has attempted it
    COMPLETED = "completed"  # a later increment produced a value
    UNAVAILABLE = "unavailable"  # a later increment attempted it and could not (see `reason`)


class FeatureOperationType(str, Enum):
    """Categories of feature-engineering operation.

    Defined now so the contract is stable; **no operation is executed,
    recommended, or even named** by Phase 6.1.
    """

    TRANSFORMATION = "transformation"
    INTERACTION = "interaction"
    AGGREGATION = "aggregation"
    DATETIME_DERIVATION = "datetime_derivation"
    CATEGORICAL_ENCODING = "categorical_encoding"
    NUMERICAL_SCALING = "numerical_scaling"
    MISSING_VALUE_HANDLING = "missing_value_handling"
    FEATURE_SELECTION = "feature_selection"


class FeatureInventoryCandidate(BaseModel):
    """One column's structural feature-inventory evidence.

    Populated by :func:`data_engine.feature_engineering.inventory_features`
    (Phase 6.2). ``candidate`` records only **structural** plausibility —
    Phase 6.2 never decides whether a column is *predictively* useful.
    """

    column: str
    column_type: ColumnType
    n_observations: int = Field(description="Non-null values.")
    n_missing: int
    missing_fraction: float
    n_unique: int = Field(description="Distinct non-null values.")
    unique_fraction: float = Field(
        description="n_unique / n_observations; 0.0 when there are no observations."
    )
    identifier_like: bool = Field(
        description="Name / behaviour looks like a row identifier (excluded from candidates)."
    )
    constant: bool = Field(description="<= 1 distinct non-null value.")
    all_missing: bool = Field(description="Every value is missing.")
    is_target: bool = Field(
        default=False, description="True iff this is the caller-declared prediction target."
    )
    candidate: bool = Field(
        description="True iff the column is a structurally plausible input feature."
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Deterministic evidence for the candidate / excluded decision, fixed order.",
    )


class FeatureInventory(BaseModel):
    """Candidate input features identified from the dataset.

    Populated by :func:`data_engine.feature_engineering.inventory_features`
    (Phase 6.2). ``candidates`` / ``objective_used`` are additive and
    defaulted, so a ``FeatureInventory`` serialised by Phase 6.1 still
    validates.
    """

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why the inventory is unavailable; None once it is produced.",
    )
    candidate_features: list[str] = Field(
        default_factory=list,
        description="Column names that are structurally plausible input features, "
        "alphabetically ordered; empty until inferred.",
    )
    excluded_features: list[str] = Field(
        default_factory=list,
        description="Columns excluded from feature consideration (target / constant / "
        "all-missing / identifier-like), alphabetically ordered.",
    )
    candidates: list[FeatureInventoryCandidate] = Field(
        default_factory=list,
        description="Per-column structural evidence, alphabetically ordered by column name.",
    )
    objective_used: bool = Field(
        default=False,
        description="True iff the objective materially affected an inclusion / exclusion.",
    )
    notes: list[str] = Field(default_factory=list)


class TransformationRecommendation(BaseModel):
    """One structurally-justified transformation worth considering for a column.

    Produced by
    :func:`data_engine.feature_engineering.recommend_transformations`
    (Phase 6.3). ``operation`` is a stable :class:`FeatureOperationType`
    category; ``description`` is the specific human-readable sub-operation
    (e.g. ``"log1p transform"``, ``"derive month"``). A recommendation
    means only *"the observed structure makes this worth considering"* —
    never *"this will improve model performance"*.
    """

    column: str
    operation: FeatureOperationType
    description: str = Field(description="Specific sub-operation, e.g. 'log transform'.")
    reason: str = Field(description="The structural reason this is worth considering.")
    evidence: list[str] = Field(
        default_factory=list, description="Deterministic supporting evidence, fixed order."
    )


class TransformationRecommendations(BaseModel):
    """Recommended per-feature transformations / derivations.

    Populated by
    :func:`data_engine.feature_engineering.recommend_transformations`
    (Phase 6.3). ``recommendations`` / ``objective_used`` are additive and
    defaulted, so a ``TransformationRecommendations`` serialised by Phase
    6.1 still validates.
    """

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why transformations are unavailable, or why an empty completed result "
        "has no recommendations; None otherwise.",
    )
    recommended_operations: list[str] = Field(
        default_factory=list,
        description="'<column>: <description>' for every structured recommendation, "
        "deterministically ordered; empty until inferred / when none apply.",
    )
    recommendations: list[TransformationRecommendation] = Field(
        default_factory=list,
        description="Structured recommendations, deterministically ordered.",
    )
    objective_used: bool = Field(
        default=False, description="True iff a non-blank objective string was supplied."
    )
    notes: list[str] = Field(default_factory=list)


class FeatureSelectionAction(str, Enum):
    """What Phase 6.4 recommends doing with a structurally eligible feature."""

    RETAIN = "retain"  # no deterministic structural reason to exclude
    DROP = "drop"  # a fixed structural rule clearly recommends exclusion
    REVIEW = "review"  # worth a human look; NOT automatically dropped


class FeatureSelectionRecommendation(BaseModel):
    """One structurally-justified feature-selection decision for a column.

    Produced by
    :func:`data_engine.feature_engineering.recommend_feature_selection`
    (Phase 6.4). ``action`` / ``reason`` describe a **structural** decision
    only — never a claim about predictive usefulness or model performance.
    """

    column: str
    action: FeatureSelectionAction
    reason: str = Field(description="The structural reason for the action.")
    evidence: list[str] = Field(
        default_factory=list, description="Deterministic supporting evidence, fixed order."
    )


class FeatureSelectionRecommendations(BaseModel):
    """Recommended feature keep / drop / review decisions.

    Populated by
    :func:`data_engine.feature_engineering.recommend_feature_selection`
    (Phase 6.4). ``recommendations`` / ``review_features`` /
    ``objective_used`` are additive and defaulted, so a
    ``FeatureSelectionRecommendations`` serialised by Phase 6.1 still
    validates.
    """

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why selection is unavailable, or why a completed result is empty; "
        "None otherwise.",
    )
    selected_features: list[str] = Field(
        default_factory=list,
        description="Features to retain (no structural reason to exclude), alphabetical.",
    )
    dropped_features: list[str] = Field(
        default_factory=list,
        description="Features a fixed structural rule recommends excluding, alphabetical.",
    )
    review_features: list[str] = Field(
        default_factory=list,
        description="Features worth a human look but NOT auto-dropped, alphabetical.",
    )
    recommendations: list[FeatureSelectionRecommendation] = Field(
        default_factory=list,
        description="Structured per-column decisions, deterministically ordered.",
    )
    objective_used: bool = Field(
        default=False, description="True iff a non-blank objective string was supplied."
    )
    notes: list[str] = Field(default_factory=list)


class PreprocessingRequirements(BaseModel):
    """Preprocessing a model would require (encoding / scaling / imputation).

    Populated by a later Phase-6 increment. Phase 6.1 leaves it empty.
    """

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why preprocessing requirements are unavailable; None once produced.",
    )
    required_operations: list[str] = Field(
        default_factory=list,
        description="Required preprocessing operation identifiers; empty until inferred.",
    )
    encoding_required: bool = Field(
        default=False, description="True once a later increment confirms encoding is needed."
    )
    scaling_required: bool = Field(
        default=False, description="True once a later increment confirms scaling is needed."
    )
    imputation_required: bool = Field(
        default=False, description="True once a later increment confirms imputation is needed."
    )
    notes: list[str] = Field(default_factory=list)


class FeatureEngineeringAssessment(BaseModel):
    """Whether feature engineering is feasible / safe to proceed with.

    Populated by a later Phase-6 increment. Phase 6.1 leaves it at
    ``not_yet_inferred`` with ``feasible = None`` — never a fabricated
    ``False``.
    """

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Why the assessment is unavailable; None once produced.",
    )
    feasible: bool | None = Field(
        default=None,
        description="True/False once assessed; None until then (never a fabricated False).",
    )
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class FeatureEngineeringRequest(BaseModel):
    """The explicit input to :func:`understand_feature_engineering`.

    Carries **dataset identity** (reusing the ``dataset_id`` /
    ``dataset_version_id`` convention shared by ``DatasetProfile`` /
    ``QualityReport`` / ``EDAReport`` / ``ProblemSpec``) and, optionally,
    the **explicit** user objective. The objective is preserved verbatim
    and never inferred from column names or data content.
    """

    dataset_id: str = Field(description="Identifier of the dataset being analysed.")
    dataset_version_id: str | None = Field(
        default=None, description="Registered DatasetVersion id, when the caller has one."
    )
    objective: str | None = Field(
        default=None,
        description="Plain-language analytical goal, exactly as the user supplied it.",
    )


class FeatureEngineeringSpec(BaseModel):
    """The structured answer to 'what feature engineering does this need?'.

    Phase 6.1 produces a spec whose overall ``status`` and every nested
    section are ``not_yet_inferred``; the ``dataset_id`` /
    ``dataset_version_id`` / ``objective`` fields echo the request. Later
    increments fill in ``inventory`` / ``transformations`` / ``selection``
    / ``preprocessing`` / ``assessment`` in place, additively.
    """

    feature_engineering_engine_version: str = FEATURE_ENGINEERING_ENGINE_VERSION

    dataset_id: str
    dataset_version_id: str | None = None
    objective: str | None = Field(
        default=None, description="The user's objective, verbatim; None if none was supplied."
    )
    objective_provided: bool = Field(
        description="True iff the request carried a non-blank objective string (after strip)."
    )

    status: FeatureEngineeringStatus = FeatureEngineeringStatus.NOT_YET_INFERRED
    reason: str | None = Field(
        default=None,
        description="Explains a non-completed overall status.",
    )

    inventory: FeatureInventory = Field(default_factory=FeatureInventory)
    transformations: TransformationRecommendations = Field(
        default_factory=TransformationRecommendations
    )
    selection: FeatureSelectionRecommendations = Field(
        default_factory=FeatureSelectionRecommendations
    )
    preprocessing: PreprocessingRequirements = Field(default_factory=PreprocessingRequirements)
    assessment: FeatureEngineeringAssessment = Field(default_factory=FeatureEngineeringAssessment)

    notes: list[str] = Field(default_factory=list)
