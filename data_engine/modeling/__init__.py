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

from .candidate_generation import (
    MODEL_CANDIDATE_NEURAL_MIN_FEATURES,
    MODEL_CANDIDATE_NEURAL_MIN_ROWS,
    generate_model_candidates,
)
from .models import (
    MODEL_ENGINE_VERSION,
    DataSplitPlan,
    DataSplitStrategy,
    EvaluationResults,
    ModelCandidate,
    ModelCandidates,
    ModelFamily,
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    TrainingOutcome,
)
from .readiness import (
    MODEL_READINESS_MIN_ROWS,
    MODEL_READINESS_ROWS_WARNING,
    assess_model_readiness,
)
from .split_planning import (
    DEFAULT_TEST_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY,
    MODEL_SPLIT_MIN_ROWS,
    MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION,
    SMALL_DATA_TEST_FRACTION,
    SMALL_DATA_TRAIN_FRACTION,
    recommend_data_split,
)
from .understanding import understand_modeling

__all__ = [
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_VALIDATION_FRACTION",
    "MODEL_CANDIDATE_NEURAL_MIN_FEATURES",
    "MODEL_CANDIDATE_NEURAL_MIN_ROWS",
    "MODEL_ENGINE_VERSION",
    "MODEL_READINESS_MIN_ROWS",
    "MODEL_READINESS_ROWS_WARNING",
    "MODEL_SPLIT_MIN_CLASS_COUNT_FOR_STRATIFY",
    "MODEL_SPLIT_MIN_ROWS",
    "MODEL_SPLIT_MIN_ROWS_FOR_VALIDATION",
    "SMALL_DATA_TEST_FRACTION",
    "SMALL_DATA_TRAIN_FRACTION",
    "DataSplitPlan",
    "DataSplitStrategy",
    "EvaluationResults",
    "ModelCandidate",
    "ModelCandidates",
    "ModelFamily",
    "ModelReadiness",
    "ModelSelection",
    "ModelingRequest",
    "ModelingSpec",
    "ModelingStatus",
    "TrainingOutcome",
    "assess_model_readiness",
    "generate_model_candidates",
    "recommend_data_split",
    "understand_modeling",
]
