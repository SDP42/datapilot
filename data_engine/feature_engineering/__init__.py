"""Automated Feature Engineering (Phase 6) — deterministic, analysis-only.

Phase 6 turns a dataset + an explicit objective into a structured
:class:`FeatureEngineeringSpec`: which columns are candidate input
features, which transformations / encoders / scalers / imputers a model
would need, which features to keep or drop, and whether feature
engineering is feasible.

**Implemented so far:**

- **6.1** — the `FeatureEngineeringSpec` contract +
  `understand_feature_engineering` foundation (infers nothing).
- **6.2** — `inventory_features`: a deterministic **structural** feature
  inventory classifying each column as a candidate input feature or an
  excluded column.
- **6.3** — `recommend_transformations`: deterministic, rule-based
  **recommendations** of transformations (log / log1p / sqrt / reciprocal
  / absolute-value, datetime derivations, scaling-as-a-category) that the
  observed structure makes worth considering. It recommends only — it
  never executes a transformation or modifies the DataFrame.

Feature selection, preprocessing requirements, and feature-engineering
feasibility are later Phase-6 increments.

    from data_engine.feature_engineering import (
        FeatureEngineeringRequest,
        inventory_features,
        recommend_transformations,
        understand_feature_engineering,
    )

    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="sales", objective="predict churn")
    )
    inv = inventory_features(df, target="churn")
    spec = spec.model_copy(update={"inventory": inv})
    spec = spec.model_copy(
        update={"transformations": recommend_transformations(df, inv)}
    )
"""

from __future__ import annotations

from .feature_inventory import HIGH_UNIQUE_ID_THRESHOLD, inventory_features
from .feature_selection import (
    FEATURE_SELECTION_HIGH_CARDINALITY,
    FEATURE_SELECTION_HIGH_CORRELATION,
    FEATURE_SELECTION_HIGH_MISSING_THRESHOLD,
    FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE,
    FEATURE_SELECTION_MIN_CORR_OBS,
    recommend_feature_selection,
)
from .models import (
    FEATURE_ENGINEERING_ENGINE_VERSION,
    FeatureEngineeringAssessment,
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
    FeatureOperationType,
    FeatureSelectionAction,
    FeatureSelectionRecommendation,
    FeatureSelectionRecommendations,
    PreprocessingRequirement,
    PreprocessingRequirements,
    TransformationRecommendation,
    TransformationRecommendations,
)
from .preprocessing_requirements import recommend_preprocessing
from .transformation_recommendation import (
    TRANSFORMATION_ABS_SYMMETRY_RATIO,
    TRANSFORMATION_LOG_RANGE_RATIO,
    TRANSFORMATION_MIN_OBS,
    TRANSFORMATION_SCALING_MAGNITUDE,
    TRANSFORMATION_SKEW_THRESHOLD,
    TRANSFORMATION_STRONG_SKEW_THRESHOLD,
    recommend_transformations,
)
from .understanding import understand_feature_engineering

__all__ = [
    "FEATURE_ENGINEERING_ENGINE_VERSION",
    "FEATURE_SELECTION_HIGH_CARDINALITY",
    "FEATURE_SELECTION_HIGH_CORRELATION",
    "FEATURE_SELECTION_HIGH_MISSING_THRESHOLD",
    "FEATURE_SELECTION_LOW_VARIANCE_MAX_UNIQUE",
    "FEATURE_SELECTION_MIN_CORR_OBS",
    "HIGH_UNIQUE_ID_THRESHOLD",
    "TRANSFORMATION_ABS_SYMMETRY_RATIO",
    "TRANSFORMATION_LOG_RANGE_RATIO",
    "TRANSFORMATION_MIN_OBS",
    "TRANSFORMATION_SCALING_MAGNITUDE",
    "TRANSFORMATION_SKEW_THRESHOLD",
    "TRANSFORMATION_STRONG_SKEW_THRESHOLD",
    "FeatureEngineeringAssessment",
    "FeatureEngineeringRequest",
    "FeatureEngineeringSpec",
    "FeatureEngineeringStatus",
    "FeatureInventory",
    "FeatureInventoryCandidate",
    "FeatureOperationType",
    "FeatureSelectionAction",
    "FeatureSelectionRecommendation",
    "FeatureSelectionRecommendations",
    "PreprocessingRequirement",
    "PreprocessingRequirements",
    "TransformationRecommendation",
    "TransformationRecommendations",
    "inventory_features",
    "recommend_feature_selection",
    "recommend_preprocessing",
    "recommend_transformations",
    "understand_feature_engineering",
]
