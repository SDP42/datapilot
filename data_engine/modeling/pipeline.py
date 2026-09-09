"""Phase 7 (stabilization) — deterministic end-to-end modeling composition.

:func:`run_modeling_pipeline` chains the **existing** Phase-5, Phase-6,
and Phase-7 public functions into one fully-populated
:class:`ModelingSpec`. It is a **composition layer only**:

* it calls only existing public APIs and duplicates none of their logic;
* it adds no inference rule, metric, model family, or heuristic;
* it introduces no randomness beyond the fixed Phase-7.4 training seed;
* it creates no file, persists no model, makes no network / LLM call;
* it never mutates the input DataFrame or any upstream object;
* it preserves every component's own ``completed`` / ``unavailable`` /
  reason semantics — when a stage is unavailable, the later stages report
  their own unavailable state (they are still called, but each returns
  immediately without fitting or computing anything).

The only thing it decides that the individual functions do not is the
**overall** ``ModelingSpec.status``: ``completed`` once a model is
recommended, ``unavailable`` (with a reason pointing at the stage that
stopped the pipeline) otherwise. ``understand_modeling`` itself is
unchanged and still returns an all-``not_yet_inferred`` spec.
"""

from __future__ import annotations

import pandas as pd

from data_engine.feature_engineering import (
    FeatureEngineeringRequest,
    FeatureEngineeringSpec,
    assess_feature_engineering,
    inventory_features,
    recommend_feature_selection,
    recommend_preprocessing,
    recommend_transformations,
    understand_feature_engineering,
)
from data_engine.problem_understanding import (
    ProblemSpec,
    ProblemUnderstandingRequest,
    assess_feasibility,
    identify_target,
    infer_task_type,
    recommend_metrics,
    understand_problem,
)

from .candidate_generation import generate_model_candidates
from .evaluation import summarize_evaluation
from .models import (
    DataSplitPlan,
    EvaluationResults,
    ModelCandidates,
    ModelingRequest,
    ModelingSpec,
    ModelingStatus,
    ModelReadiness,
    ModelSelection,
    TrainingOutcome,
)
from .readiness import assess_model_readiness
from .selection import select_model
from .split_planning import recommend_data_split
from .training import train_and_evaluate_models
from .understanding import understand_modeling

_NOTE_COMPOSED = (
    "ModelingSpec composed by run_modeling_pipeline: Phase 5 (problem understanding) -> "
    "Phase 6 (feature engineering) -> Phase 7.1-7.5, all deterministic; no model was "
    "persisted and the input DataFrame was not modified"
)
_NOTE_EVAL_SOT = (
    "evaluation results live in ModelingSpec.training (TrainingOutcome); ModelingSpec.evaluation "
    "is a summary mirror produced by summarize_evaluation and recomputes nothing"
)


def _build_problem_spec(df: pd.DataFrame, request: ModelingRequest) -> ProblemSpec:
    """Compose the Phase-5 ``ProblemSpec`` from the existing Phase-5 functions."""
    objective = request.objective
    spec = understand_problem(
        ProblemUnderstandingRequest(
            dataset_id=request.dataset_id,
            dataset_version_id=request.dataset_version_id,
            objective=objective,
        )
    )
    target = identify_target(df, objective=objective)
    task_type = infer_task_type(df, target, objective=objective)
    metrics = recommend_metrics(df, task_type, objective=objective)
    feasibility = assess_feasibility(df, target, task_type, metrics, objective=objective)
    return spec.model_copy(
        update={
            "target": target,
            "task_type": task_type,
            "metrics": metrics,
            "feasibility": feasibility,
        }
    )


def _build_feature_engineering_spec(
    df: pd.DataFrame, request: ModelingRequest, problem: ProblemSpec
) -> FeatureEngineeringSpec:
    """Compose the Phase-6 ``FeatureEngineeringSpec`` from the existing Phase-6 functions."""
    objective = request.objective
    spec = understand_feature_engineering(
        FeatureEngineeringRequest(
            dataset_id=request.dataset_id,
            dataset_version_id=request.dataset_version_id,
            objective=objective,
        )
    )
    target_column = problem.target.target_column
    inventory = inventory_features(df, target_column, objective=objective)
    transformations = recommend_transformations(df, inventory, objective=objective)
    selection = recommend_feature_selection(df, inventory, problem.task_type, objective=objective)
    preprocessing = recommend_preprocessing(
        df, inventory, transformations, selection, objective=objective
    )
    assessment = assess_feature_engineering(
        df, inventory, transformations, selection, preprocessing, objective=objective
    )
    return spec.model_copy(
        update={
            "inventory": inventory,
            "transformations": transformations,
            "selection": selection,
            "preprocessing": preprocessing,
            "assessment": assessment,
        }
    )


def _resolve_overall_status(
    readiness: ModelReadiness,
    split: DataSplitPlan,
    candidates: ModelCandidates,
    training: TrainingOutcome,
    evaluation: EvaluationResults,
    selection: ModelSelection,
) -> tuple[ModelingStatus, str | None]:
    """Deterministically resolve the overall ``ModelingSpec.status``.

    ``completed`` iff the pipeline reached a concrete model recommendation;
    otherwise ``unavailable`` with a reason naming the stage that stopped
    it (mirroring the individual components' own precedence).
    """
    unavailable = ModelingStatus.UNAVAILABLE
    completed = ModelingStatus.COMPLETED

    if readiness.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at model readiness: "
            f"{readiness.reason or 'readiness is unavailable'}"
        )
    if readiness.ready is False:
        first = (
            readiness.blocking_issues[0]
            if readiness.blocking_issues
            else (readiness.reason or "the data is not ready for modeling")
        )
        return unavailable, f"model readiness reported the pipeline not ready: {first}"
    if split.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at data-split planning: "
            f"{split.reason or 'the split plan is unavailable'}"
        )
    if candidates.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at model-candidate generation: "
            f"{candidates.reason or 'no candidates are available'}"
        )
    if training.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at model training: "
            f"{training.reason or 'training is unavailable'}"
        )
    if evaluation.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at evaluation summary: "
            f"{evaluation.reason or 'evaluation is unavailable'}"
        )
    if selection.status is not completed:
        return unavailable, (
            f"the modeling pipeline stopped at model selection: "
            f"{selection.reason or 'selection is unavailable'}"
        )
    if selection.selected_family is None:
        return unavailable, (
            "model training completed but no model could be recommended: "
            f"{selection.reason or 'no training run carried the selection metric'}"
        )
    return completed, None


def run_modeling_pipeline(df: pd.DataFrame, request: ModelingRequest) -> ModelingSpec:
    """Deterministically compose the full Phase-5 -> Phase-7 modeling pipeline.

    Parameters
    ----------
    df:
        The dataset (raw or already processed). **Not mutated.** A
        non-DataFrame raises ``TypeError``.
    request:
        A :class:`ModelingRequest` — dataset identity plus an optional,
        **explicit** objective (never inferred from the data). A non-model
        raises ``TypeError``; a blank ``dataset_id`` raises ``ValueError``
        (both via :func:`understand_modeling`).

    Returns
    -------
    ModelingSpec
        A fully-populated spec: ``readiness`` / ``split`` / ``candidates``
        / ``training`` / ``evaluation`` / ``selection`` each carry the
        exact output of the corresponding Phase-7 function, and the
        overall ``status`` is ``completed`` (a model is recommended in
        ``selection``) or ``unavailable`` (``reason`` names the stage that
        stopped the pipeline). ``dataset_id`` / ``dataset_version_id`` /
        ``objective`` / ``objective_provided`` are preserved from the
        Phase-7.1 foundation spec.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"run_modeling_pipeline expects a pandas DataFrame, got {type(df).__name__}"
        )
    # Type / value validation of the request is delegated verbatim to the
    # Phase-7.1 foundation so the contract stays identical.
    spec = understand_modeling(request)
    objective = request.objective

    problem = _build_problem_spec(df, request)
    feature_engineering = _build_feature_engineering_spec(df, request, problem)

    readiness = assess_model_readiness(df, problem, feature_engineering, objective=objective)
    split = recommend_data_split(df, problem, feature_engineering, objective=objective)
    candidates = generate_model_candidates(
        df, problem, feature_engineering, readiness, split, objective=objective
    )
    training = train_and_evaluate_models(
        df, problem, feature_engineering, readiness, split, candidates, objective=objective
    )
    evaluation = summarize_evaluation(training)
    selection = select_model(
        problem, feature_engineering, readiness, split, candidates, training, objective=objective
    )

    status, reason = _resolve_overall_status(
        readiness, split, candidates, training, evaluation, selection
    )

    return spec.model_copy(
        update={
            "status": status,
            "reason": reason,
            "readiness": readiness,
            "split": split,
            "candidates": candidates,
            "training": training,
            "evaluation": evaluation,
            "selection": selection,
            "notes": [_NOTE_COMPOSED, _NOTE_EVAL_SOT],
        }
    )
