"""Automated Feature Engineering (Phase 6) — deterministic, analysis-only.

Phase 6 turns a dataset + an explicit objective into a structured
:class:`FeatureEngineeringSpec`: which columns are candidate input
features, which transformations / encoders / scalers / imputers a model
would need, which features to keep or drop, and whether feature
engineering is feasible.

Every function is a standalone, deterministic, **recommendation-only**
step the caller composes into a ``FeatureEngineeringSpec``. Nothing is
executed — no column is created, encoded, scaled, imputed, or dropped.

- **6.1** — `understand_feature_engineering`: the contract + foundation
  (infers nothing).
- **6.2** — `inventory_features`: structural feature inventory
  (candidate vs. excluded).
- **6.3** — `recommend_transformations`: rule-based transformation
  recommendations (log / log1p / sqrt / reciprocal / absolute-value,
  datetime derivations, scaling-as-a-category).
- **6.4** — `recommend_feature_selection`: retain / drop / review from
  fixed structural + redundancy rules.
- **6.5** — `recommend_preprocessing`: identifies the imputation /
  encoding / scaling operations a model would require.
- **forecasting foundation** — `recommend_temporal_features`: lag /
  rolling-window feature recommendations for a time-series-forecasting
  problem (``unavailable`` for every other task).
- **6.6** — `assess_feature_engineering`: structural consistency &
  readiness check over 6.2–6.5 (+ the temporal section).
"""

from __future__ import annotations

from .assessment import assess_feature_engineering
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
    FeatureEngineeringCheck,
    FeatureEngineeringCheckOutcome,
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
    TemporalFeatureRecommendation,
    TemporalFeatureRecommendations,
    TransformationRecommendation,
    TransformationRecommendations,
)
from .preprocessing_requirements import recommend_preprocessing
from .temporal_features import (
    FORECASTING_LAG_ORDERS,
    FORECASTING_MIN_ROWS_FOR_TEMPORAL,
    FORECASTING_ROLLING_WINDOWS,
    FORECASTING_TEMPORAL_ROW_MARGIN,
    recommend_temporal_features,
)
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
    "FORECASTING_LAG_ORDERS",
    "FORECASTING_MIN_ROWS_FOR_TEMPORAL",
    "FORECASTING_ROLLING_WINDOWS",
    "FORECASTING_TEMPORAL_ROW_MARGIN",
    "HIGH_UNIQUE_ID_THRESHOLD",
    "TRANSFORMATION_ABS_SYMMETRY_RATIO",
    "TRANSFORMATION_LOG_RANGE_RATIO",
    "TRANSFORMATION_MIN_OBS",
    "TRANSFORMATION_SCALING_MAGNITUDE",
    "TRANSFORMATION_SKEW_THRESHOLD",
    "TRANSFORMATION_STRONG_SKEW_THRESHOLD",
    "FeatureEngineeringAssessment",
    "FeatureEngineeringCheck",
    "FeatureEngineeringCheckOutcome",
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
    "TemporalFeatureRecommendation",
    "TemporalFeatureRecommendations",
    "TransformationRecommendation",
    "TransformationRecommendations",
    "assess_feature_engineering",
    "inventory_features",
    "recommend_feature_selection",
    "recommend_preprocessing",
    "recommend_temporal_features",
    "recommend_transformations",
    "understand_feature_engineering",
]
