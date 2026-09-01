"""Automated Feature Engineering (Phase 6) — deterministic, analysis-only.

Phase 6 turns a dataset + an explicit objective into a structured
:class:`FeatureEngineeringSpec`: which columns are candidate features,
which transformations / encoders / scalers / imputers are needed, which
features to keep or drop, and whether feature engineering is feasible.

**Phase 6.1 (this increment) is the contract + foundation only.**
:func:`understand_feature_engineering` validates an explicit
:class:`FeatureEngineeringRequest` and returns a ``FeatureEngineeringSpec``
whose sections are all ``not_yet_inferred`` — no feature, transformation,
encoder, scaler, imputer, selection, importance, or feasibility verdict
is produced yet, and the user's objective is never inferred from the
data.

    from data_engine.feature_engineering import (
        FeatureEngineeringRequest,
        understand_feature_engineering,
    )

    spec = understand_feature_engineering(
        FeatureEngineeringRequest(dataset_id="sales", objective="predict churn")
    )
    payload = spec.model_dump(mode="json")
"""

from __future__ import annotations

from .models import (
    FEATURE_ENGINEERING_ENGINE_VERSION,
    FeatureEngineeringAssessment,
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    FeatureEngineeringStatus,
    FeatureInventory,
    FeatureOperationType,
    FeatureSelectionRecommendations,
    PreprocessingRequirements,
    TransformationRecommendations,
)
from .understanding import understand_feature_engineering

__all__ = [
    "FEATURE_ENGINEERING_ENGINE_VERSION",
    "FeatureEngineeringAssessment",
    "FeatureEngineeringRequest",
    "FeatureEngineeringSpec",
    "FeatureEngineeringStatus",
    "FeatureInventory",
    "FeatureOperationType",
    "FeatureSelectionRecommendations",
    "PreprocessingRequirements",
    "TransformationRecommendations",
    "understand_feature_engineering",
]
