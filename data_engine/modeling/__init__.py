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
    DeepTuneResult,
    EvaluationResults,
    ExpandedCandidateResult,
    ExpandedSearchResult,
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
from .persistence import (
    PREDICTION_ENGINE_VERSION,
    PersistedModelMetadata,
    PredictionResult,
    list_models,
    load_model,
    predict_with_model,
    save_model,
)
from .pipeline import (
    run_clustering_search,
    run_deep_tune,
    run_expanded_model_search,
    run_modeling_pipeline,
    train_and_persist_model,
)
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
from .temporal_execution import (
    build_calendar_features,
    build_temporal_features,
    temporal_feature_spec,
)
from .training import (
    MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS,
    MODEL_TRAINING_FOREST_N_ESTIMATORS,
    MODEL_TRAINING_KNN_N_NEIGHBORS,
    MODEL_TRAINING_METRIC_ROUND,
    MODEL_TRAINING_N_CLUSTERS,
    MODEL_TRAINING_RANDOM_SEED,
    MODEL_TRAINING_TREE_MAX_DEPTH,
    FittedPipeline,
    fit_final_pipeline,
    run_expanded_clustering_search,
    run_expanded_search,
    train_and_evaluate_models,
    tune_best_candidate,
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
    "MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS",
    "MODEL_TRAINING_FOREST_N_ESTIMATORS",
    "MODEL_TRAINING_KNN_N_NEIGHBORS",
    "MODEL_TRAINING_METRIC_ROUND",
    "MODEL_TRAINING_N_CLUSTERS",
    "MODEL_TRAINING_RANDOM_SEED",
    "MODEL_TRAINING_TREE_MAX_DEPTH",
    "PREDICTION_ENGINE_VERSION",
    "SMALL_DATA_TEST_FRACTION",
    "SMALL_DATA_TRAIN_FRACTION",
    "DataSplitPlan",
    "DataSplitStrategy",
    "DeepTuneResult",
    "EvaluationResults",
    "ExpandedCandidateResult",
    "ExpandedSearchResult",
    "FittedPipeline",
    "ModelCandidate",
    "ModelCandidates",
    "ModelFamily",
    "ModelReadiness",
    "ModelSelection",
    "ModelSelectionRank",
    "ModelingRequest",
    "ModelingSpec",
    "ModelingStatus",
    "PersistedModelMetadata",
    "PredictionResult",
    "TrainingOutcome",
    "TrainingRun",
    "TrainingRunStatus",
    "assess_model_readiness",
    "build_calendar_features",
    "build_temporal_features",
    "fit_final_pipeline",
    "generate_model_candidates",
    "list_models",
    "load_model",
    "predict_with_model",
    "recommend_data_split",
    "run_clustering_search",
    "run_deep_tune",
    "run_expanded_clustering_search",
    "run_expanded_model_search",
    "run_expanded_search",
    "run_modeling_pipeline",
    "save_model",
    "select_model",
    "summarize_evaluation",
    "temporal_feature_spec",
    "train_and_evaluate_models",
    "train_and_persist_model",
    "tune_best_candidate",
    "understand_modeling",
]
