"""Model Development / Modeling (Phase 7) — deterministic, analysis-only.

Phase 7 turns a dataset + an explicit objective (and, in later
increments, the upstream Phase-5 / Phase-6 contracts) into a structured
:class:`ModelingSpec`: whether the data is ready for modeling, how to
split it, which model families to consider, the training outcome, the
evaluation results, and the selected model.

**Phase 7.1 (this increment) is the contract + foundation only.**
:func:`understand_modeling` validates an explicit :class:`ModelingRequest`
and returns a ``ModelingSpec`` whose sections are all
``not_yet_inferred`` — no readiness verdict, split, candidate model,
training run, metric, or selection is produced yet, and no DataFrame is
inspected.

    from data_engine.modeling import ModelingRequest, understand_modeling

    spec = understand_modeling(
        ModelingRequest(dataset_id="sales", objective="predict churn")
    )
    payload = spec.model_dump(mode="json")
"""

from __future__ import annotations

from .models import (
    MODEL_ENGINE_VERSION,
    DataSplitPlan,
    EvaluationResults,
    ModelCandidates,
    ModelFamily,
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    TrainingOutcome,
)
from .understanding import understand_modeling

__all__ = [
    "MODEL_ENGINE_VERSION",
    "DataSplitPlan",
    "EvaluationResults",
    "ModelCandidates",
    "ModelFamily",
    "ModelReadiness",
    "ModelSelection",
    "ModelingRequest",
    "ModelingSpec",
    "ModelingStatus",
    "TrainingOutcome",
    "understand_modeling",
]
