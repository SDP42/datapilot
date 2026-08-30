"""Machine-readable models for a *proposed* cleaning plan.

The cleaning layer has three strictly separated stages:

    DETECTION   (data_engine.quality)  -> QualityReport
    PLANNING    (this package)          -> CleaningPlan     <-- we are here
    EXECUTION   (future phase)          -> cleaned dataset + lineage

A ``CleaningPlan`` is a list of ``CleaningOperation`` proposals. Nothing
in this package changes data. Every operation is explicit, typed,
explainable, auditable, and traceable back to the ``QualityFinding`` that
caused it.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from data_engine.quality.models import FindingType

PLANNER_VERSION = "1"


class OperationType(str, Enum):
    """The concrete kind of proposed operation."""

    # missing values
    IMPUTE_MISSING_NUMERIC = "impute_missing_numeric"
    IMPUTE_MISSING_CATEGORICAL = "impute_missing_categorical"
    IMPUTE_MISSING_DATETIME = "impute_missing_datetime"
    HANDLE_MISSING_VALUES = "handle_missing_values"  # generic (column type unknown)
    DROP_HIGH_MISSING_COLUMN = "drop_high_missing_column"
    # duplicates
    REMOVE_EXACT_DUPLICATE_ROWS = "remove_exact_duplicate_rows"
    # type conversion
    CONVERT_TEXT_TO_NUMERIC = "convert_text_to_numeric"
    CONVERT_TEXT_TO_DATETIME = "convert_text_to_datetime"
    # categoricals
    TRIM_CATEGORY_WHITESPACE = "trim_category_whitespace"
    STANDARDIZE_CATEGORY_FORMATTING = "standardize_category_formatting"
    # numeric distribution
    REVIEW_OUTLIERS = "review_outliers"
    TRANSFORM_DISTRIBUTION_LOG = "transform_distribution_log"
    REVIEW_DISTRIBUTION_TRANSFORM = "review_distribution_transform"
    # modelling advice (NOT a data transformation)
    RECOMMEND_IMBALANCE_STRATEGY = "recommend_imbalance_strategy"


class OperationCategory(str, Enum):
    """What sort of thing the operation is."""

    DATA_TRANSFORMATION = "data_transformation"  # would change the dataset
    INVESTIGATION = "investigation"  # a human review task, no change proposed
    MODELING_RECOMMENDATION = "modeling_recommendation"  # advice for the ML phase


class OperationStatus(str, Enum):
    """How much human judgement an operation needs before it may run.

    The planner is deliberately conservative: when in doubt it escalates.
    """

    RECOMMENDED = "recommended"  # relatively safe; execute after a glance
    REVIEW_REQUIRED = "review_required"  # a human/domain call is needed first
    NOT_SAFE_TO_AUTOMATE = "not_safe_to_automate"  # needs real context; never auto-run


class ImputationStrategy(str, Enum):
    MEDIAN = "median"
    MODE = "mode"


class CleaningOperation(BaseModel):
    """One proposed cleaning step. A proposal — never executed here."""

    operation_id: str = Field(
        description="Stable id within a plan, e.g. 'impute_missing_numeric:age'."
    )
    operation_type: OperationType
    category: OperationCategory
    status: OperationStatus
    status_reason: str = Field(description="Why the operation has this safety status.")

    target_columns: list[str] = Field(default_factory=list)

    # Traceability back to detection.
    addresses_finding_type: FindingType
    source_finding_id: str = Field(
        description="finding_id of the QualityFinding that triggered this."
    )

    problem_summary: str = Field(description="Restatement of the problem being addressed.")
    proposed_action: str = Field(description="Plain-language description of the proposed change.")
    rationale: str = Field(description="Why this particular action is proposed.")
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list, description="Possible downsides.")

    strategy: ImputationStrategy | None = None
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-readable params for the future executor (JSON primitives only).",
    )

    affected_rows: int | None = None
    affected_percentage: float | None = None
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Planner confidence this is the right action."
    )
    requires_train_test_split_awareness: bool = Field(
        default=False,
        description=(
            "True if execution must fit parameters on the training split only "
            "(imputation values, transform parameters) to avoid data leakage."
        ),
    )


class CleaningPlanSummary(BaseModel):
    total_operations: int
    by_status: dict[OperationStatus, int]
    by_type: dict[OperationType, int]
    by_category: dict[OperationCategory, int]
    columns_affected: list[str]
    auto_applicable_count: int = Field(
        description="Operations that are RECOMMENDED and are actual data transformations."
    )
    notes: list[str] = Field(default_factory=list)


class CleaningPlan(BaseModel):
    """The complete set of proposals derived from one QualityReport."""

    dataset_id: str
    planner_version: str = PLANNER_VERSION
    generated_at: _dt.datetime
    target_column: str | None = None
    based_on_quality_engine_version: str
    used_profile: bool = Field(description="Whether a DatasetProfile informed the planning.")
    source_findings_considered: int
    operations: list[CleaningOperation]
    summary: CleaningPlanSummary
