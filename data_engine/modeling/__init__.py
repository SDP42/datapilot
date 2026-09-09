"""Model Development / Modeling (Phase 7) — deterministic.

Phase 7 turns a dataset + an explicit objective + the upstream Phase-5
``ProblemSpec`` / Phase-6 ``FeatureEngineeringSpec`` into a structured
:class:`ModelingSpec`: whether the data is ready for modeling, how to
split it, which model families to consider, the training outcome, the
evaluation summary, and the selected model. Only Phase 7.4
(:func:`train_and_evaluate_models`) fits estimators (conservative
scikit-learn baselines); no increment tunes, cross-validates, or persists
a model.

:func:`understand_modeling` is the inference-free foundation: it validates
an explicit :class:`ModelingRequest` and returns a ``ModelingSpec`` whose
overall status and every section are ``not_yet_inferred``.

:func:`run_modeling_pipeline` is the deterministic end-to-end composition:
it chains the existing Phase-5, Phase-6, and Phase-7 functions into one
fully-populated ``ModelingSpec`` (and is the only producer that sets the
overall ``status`` to ``completed`` / ``unavailable``).

    from data_engine.modeling import ModelingRequest, run_modeling_pipeline

    spec = run_modeling_pipeline(
        df, ModelingRequest(dataset_id="sales", objective="predict churn")
    )
    payload = spec.model_dump(mode="json")
"""

from __future__ import annotations

from .candidate_generation import (
    MODEL_CANDIDATE_NEURAL_MIN_FEATURES,
    MODEL_CANDIDATE_NEURAL_MIN_ROWS,
    generate_model_candidates,
)
from .evaluation import summarize_evaluation
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
    ModelSelectionRank,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
)
from .pipeline import run_modeling_pipeline
from .readiness import (
    MODEL_READINESS_MIN_ROWS,
    MODEL_READINESS_ROWS_WARNING,
    assess_model_readiness,
)
from .selection import select_model
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
from .training import (
    MODEL_TRAINING_FOREST_N_ESTIMATORS,
    MODEL_TRAINING_KNN_N_NEIGHBORS,
    MODEL_TRAINING_METRIC_ROUND,
    MODEL_TRAINING_N_CLUSTERS,
    MODEL_TRAINING_RANDOM_SEED,
    MODEL_TRAINING_TREE_MAX_DEPTH,
    train_and_evaluate_models,
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
    "MODEL_TRAINING_FOREST_N_ESTIMATORS",
    "MODEL_TRAINING_KNN_N_NEIGHBORS",
    "MODEL_TRAINING_METRIC_ROUND",
    "MODEL_TRAINING_N_CLUSTERS",
    "MODEL_TRAINING_RANDOM_SEED",
    "MODEL_TRAINING_TREE_MAX_DEPTH",
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
    "ModelSelectionRank",
    "ModelingRequest",
    "ModelingSpec",
    "ModelingStatus",
    "TrainingOutcome",
    "TrainingRun",
    "TrainingRunStatus",
    "assess_model_readiness",
    "generate_model_candidates",
    "recommend_data_split",
    "run_modeling_pipeline",
    "select_model",
    "summarize_evaluation",
    "train_and_evaluate_models",
    "understand_modeling",
]
