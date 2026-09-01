"""Automated Feature Engineering (Phase 6) — deterministic, analysis-only.

Phase 6 turns a dataset + an explicit objective into a structured
:class:`FeatureEngineeringSpec`: which columns are candidate input
features, which transformations / encoders / scalers / imputers a model
would need, which features to keep or drop, and whether feature
engineering is feasible.

**Implemented so far:** Phase 6.1 (the `FeatureEngineeringSpec` contract +
`understand_feature_engineering` foundation — infers nothing) and Phase
6.2 (`inventory_features` — a deterministic **structural** feature
inventory that classifies each column as a plausible input feature or an
excluded column; it never assesses predictive usefulness). Transformation
recommendations, feature selection, preprocessing requirements, and
feature-engineering feasibility are later Phase-6 increments.

    from data_engine.feature_engineering import (
        FeatureEngineeringRequest,
        inventory_features,
        understand_feature_engineering,
    )

    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="sales", objective="predict churn")
    )
    spec = spec.model_copy(update={"inventory": inventory_features(df, target="churn")})
"""

from __future__ import annotations

from .feature_inventory import HIGH_UNIQUE_ID_THRESHOLD, inventory_features
from .models import (
    FEATURE_ENGINEERING_ENGINE_VERSION,
    FeatureEngineeringAssessment,
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureInventoryCandidate,
    FeatureOperationType,
    FeatureSelectionRecommendations,
    PreprocessingRequirements,
    TransformationRecommendations,
)
from .understanding import understand_feature_engineering

__all__ = [
    "FEATURE_ENGINEERING_ENGINE_VERSION",
    "HIGH_UNIQUE_ID_THRESHOLD",
    "FeatureEngineeringAssessment",
    "FeatureEngineeringRequest",
    "FeatureEngineeringSpec",
    "FeatureEngineeringStatus",
    "FeatureInventory",
    "FeatureInventoryCandidate",
    "FeatureOperationType",
    "FeatureSelectionRecommendations",
    "PreprocessingRequirements",
    "TransformationRecommendations",
    "inventory_features",
    "understand_feature_engineering",
]
