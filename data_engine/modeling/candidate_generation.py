"""Phase 7.3 — deterministic, rule-based model-candidate generation.

:func:`generate_model_candidates` turns the Phase-5 task type plus the
Phase-7.2 :class:`ModelReadiness` / :class:`DataSplitPlan` (and the
structural Phase-6 feature information) into a transparent, ordered list
of candidate :class:`ModelFamily` values.

It **generates recommendations only**. It never trains, fits, evaluates,
benchmarks, compares, tunes, or selects a model; never creates an
estimator, a prediction, a metric, or a model artifact; never re-infers
the task type; never uses target correlation, mutual information, ANOVA,
chi-square, feature importance, or SHAP; never executes preprocessing or
Phase-6 transformations; and never modifies the DataFrame or any upstream
model.
"""

from __future__ import annotations

import pandas as pd

from data_engine.feature_engineering import FeatureEngineeringSpec, FeatureEngineeringStatus
from data_engine.problem_understanding import ProblemSpec, ProblemUnderstandingStatus, TaskType
from datapilot.contracts import ColumnType

from .models import (
    DataSplitPlan,
    ModelCandidate,
    ModelCandidates,
    ModelFamily,
    ModelingStatus,
    ModelReadiness,
)

# --- tunables (documented in docs/modeling.md) -------------------------

# A neural family is only offered at a structurally large scale.
MODEL_CANDIDATE_NEURAL_MIN_ROWS = 1000
MODEL_CANDIDATE_NEURAL_MIN_FEATURES = 20

# fixed family ordering for deterministic output
_FAMILY_ORDER: dict[ModelFamily, int] = {
    ModelFamily.LINEAR: 0,
    ModelFamily.TREE_BASED: 1,
    ModelFamily.ENSEMBLE: 2,
    ModelFamily.PROBABILISTIC: 3,
    ModelFamily.DISTANCE_BASED: 4,
    ModelFamily.NEURAL: 5,
}

_CLASSIFICATION_TASKS = frozenset(
    {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}
)
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})
_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED
_NUMERIC_LIKE = frozenset({ColumnType.NUMERIC, ColumnType.BOOLEAN})

_NOTE_RECOMMENDATION_ONLY = (
    "model candidate generation recommends model families only — no model was trained, "
    "fitted, evaluated, compared, tuned, or selected, and the DataFrame was not modified"
)


def _unavailable(reason: str, *, objective_used: bool) -> ModelCandidates:
    return ModelCandidates(
        status=ModelingStatus.UNAVAILABLE,
        reason=reason,
        candidates=[],
        candidates_detail=[],
        objective_used=objective_used,
        notes=[],
    )


def _eligible_features(feature_engineering: FeatureEngineeringSpec) -> list[str]:
    selection = feature_engineering.selection
    if selection.status is _FE_COMPLETED:
        return sorted(set(selection.selected_features) | set(selection.review_features))
    return sorted(feature_engineering.inventory.candidate_features)


def generate_model_candidates(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    *,
    objective: str | None = None,
) -> ModelCandidates:
    """Deterministically recommend candidate model families for the task.

    Parameters
    ----------
    df:
        The dataset. **Not mutated / not inspected for content** (only its
        type is checked). A non-DataFrame raises ``TypeError``.
    problem:
        The **Phase-5** :class:`ProblemSpec` — the sole authority for the
        task type (never re-inferred). A non-model raises ``TypeError``.
    feature_engineering:
        The **Phase-6** :class:`FeatureEngineeringSpec`. A non-model raises
        ``TypeError``; it is **not mutated** and never executed.
    readiness:
        The **Phase-7.2** :class:`ModelReadiness`. A non-model raises
        ``TypeError``.
    split:
        The **Phase-7.2** :class:`DataSplitPlan`. A non-model raises
        ``TypeError``.
    objective:
        The user's objective, **verbatim and optional** — recorded in a
        note only; it never overrides a structural rule.

    Returns
    -------
    ModelCandidates
        ``status = completed`` with an ordered ``candidates`` /
        ``candidates_detail``; ``status = unavailable`` (empty payload)
        when the task type / readiness / split / feature-engineering
        assessment is not usable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"generate_model_candidates expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(problem, ProblemSpec):
        raise TypeError(
            f"generate_model_candidates expects a ProblemSpec, got {type(problem).__name__}"
        )
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            "generate_model_candidates expects a FeatureEngineeringSpec, "
            f"got {type(feature_engineering).__name__}"
        )
    if not isinstance(readiness, ModelReadiness):
        raise TypeError(
            f"generate_model_candidates expects a ModelReadiness, got {type(readiness).__name__}"
        )
    if not isinstance(split, DataSplitPlan):
        raise TypeError(
            f"generate_model_candidates expects a DataSplitPlan, got {type(split).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    # --- deterministic upstream precedence -----------------------------
    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED:
        return _unavailable(
            f"task-type inference is not completed (status = {task_inference.status.value})",
            objective_used=objective_used,
        )
    task = task_inference.task_type
    if task is None:
        return _unavailable(
            "task-type inference completed without a task type", objective_used=objective_used
        )
    if task in _UNSUPPORTED_TASKS:
        return _unavailable(
            f"model candidate generation does not support task type '{task.value}'",
            objective_used=objective_used,
        )
    if readiness.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"model readiness is not completed (status = {readiness.status.value})",
            objective_used=objective_used,
        )
    if readiness.ready is False:
        first = (
            readiness.blocking_issues[0]
            if readiness.blocking_issues
            else (readiness.reason or "no reason given")
        )
        return _unavailable(
            f"candidate generation is blocked by model-readiness issues: {first}",
            objective_used=objective_used,
        )
    if split.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"the data-split plan is not completed (status = {split.status.value})",
            objective_used=objective_used,
        )
    if feature_engineering.assessment.status is not _FE_COMPLETED:
        return _unavailable(
            "feature-engineering assessment is not completed "
            f"(status = {feature_engineering.assessment.status.value})",
            objective_used=objective_used,
        )

    # --- structural feature representation (from the inventory) -------
    eligible = _eligible_features(feature_engineering)
    col_type = {c.column: c.column_type for c in feature_engineering.inventory.candidates}
    eligible_types = {col_type[name] for name in eligible if name in col_type}
    has_numeric = ColumnType.NUMERIC in eligible_types
    has_categorical = ColumnType.CATEGORICAL in eligible_types
    has_boolean = ColumnType.BOOLEAN in eligible_types
    has_datetime = ColumnType.DATETIME in eligible_types
    numeric_only_representation = bool(eligible_types) and eligible_types <= _NUMERIC_LIKE
    n_rows = readiness.n_observations
    n_features = readiness.eligible_feature_count

    split_value = split.strategy.value if split.strategy is not None else "unspecified"
    base_evidence = [
        f"task type is {task.value}",
        "model-readiness assessment completed",
        f"a {split_value} split is available",
    ]

    def _repr_evidence() -> list[str]:
        parts: list[str] = []
        if has_numeric:
            parts.append("numeric feature representation is available")
        if has_categorical:
            parts.append("categorical features are present (encoding is required upstream)")
        if has_boolean:
            parts.append("boolean features are present")
        if has_datetime:
            parts.append("datetime features are present (derivation is required upstream)")
        return parts

    families: dict[ModelFamily, tuple[str, list[str]]] = {}

    def _add(family: ModelFamily, reason: str, extra: list[str] | None = None) -> None:
        families[family] = (reason, base_evidence + _repr_evidence() + (extra or []))

    forecasting_note = (
        "Phase 7.3 does not create lag features, rolling features, forecasting-specific "
        "transformations, or forecasting models"
    )

    if task is TaskType.REGRESSION:
        _add(ModelFamily.LINEAR, "linear models are a standard baseline for regression")
        _add(
            ModelFamily.TREE_BASED,
            "tree-based regressors capture non-linear relationships and mixed feature types "
            "without scaling",
        )
        _add(ModelFamily.ENSEMBLE, "tree ensembles are a strong general-purpose regression family")
        if numeric_only_representation:
            _add(
                ModelFamily.DISTANCE_BASED,
                "every eligible feature is numeric / boolean, so a distance metric is "
                "well-defined for a distance-based regressor",
            )
    elif task in _CLASSIFICATION_TASKS:
        _add(ModelFamily.LINEAR, "linear classifiers are a standard baseline for classification")
        _add(
            ModelFamily.TREE_BASED,
            "tree-based classifiers handle non-linear boundaries and mixed feature types "
            "without scaling",
        )
        _add(
            ModelFamily.ENSEMBLE,
            "tree ensembles are a strong general-purpose classification family",
        )
        _add(
            ModelFamily.PROBABILISTIC,
            "probabilistic classifiers (e.g. the naive Bayes family) are a fast structural "
            "baseline",
        )
        if numeric_only_representation:
            _add(
                ModelFamily.DISTANCE_BASED,
                "every eligible feature is numeric / boolean, so a distance metric is "
                "well-defined for a distance-based classifier",
            )
        if (
            n_rows >= MODEL_CANDIDATE_NEURAL_MIN_ROWS
            and n_features >= MODEL_CANDIDATE_NEURAL_MIN_FEATURES
        ):
            _add(
                ModelFamily.NEURAL,
                f"the dataset is structurally large ({n_rows} rows, {n_features} eligible "
                f"features), a scale at which a neural classifier is reasonable to consider",
            )
    elif task is TaskType.TIME_SERIES_FORECASTING:
        _add(
            ModelFamily.LINEAR,
            "linear models can serve as a forecasting baseline once temporal features exist",
            [forecasting_note],
        )
        _add(
            ModelFamily.TREE_BASED,
            "tree-based regressors can model a forecasting target once temporal features exist",
            [forecasting_note],
        )
        _add(
            ModelFamily.ENSEMBLE,
            "tree ensembles are a strong forecasting family once temporal features exist",
            [forecasting_note],
        )
    elif task is TaskType.CLUSTERING:
        _add(
            ModelFamily.DISTANCE_BASED,
            "distance-based clustering (e.g. k-means / hierarchical) is the standard family "
            "for a clustering task",
        )
        _add(
            ModelFamily.PROBABILISTIC,
            "probabilistic mixture models are a standard clustering family",
        )
    else:  # pragma: no cover - every supported TaskType is handled above
        return _unavailable(
            f"model candidate generation does not support task type '{task.value}'",
            objective_used=objective_used,
        )

    ordered = sorted(families.items(), key=lambda kv: _FAMILY_ORDER[kv[0]])
    candidates_detail = [
        ModelCandidate(family=family, reason=reason, evidence=evidence)
        for family, (reason, evidence) in ordered
    ]
    candidate_names = [c.family.value for c in candidates_detail]

    notes: list[str] = [
        _NOTE_RECOMMENDATION_ONLY,
        f"task type: {task.value}",
        f"data-split strategy: {split_value}",
        f"{len(candidate_names)} candidate model family(ies) recommended",
    ]
    if task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(forecasting_note)
        notes.append(
            "Phase 7.3 does not infer a forecasting task from a datetime column — the task "
            "type came from Phase 5"
        )
    if readiness.preprocessing_requirements_present:
        notes.append(
            "preprocessing requirements exist upstream (Phase 6.5) and must be handled before "
            "model training — Phase 7.3 does not apply them"
        )
    if feature_engineering.transformations.status is _FE_COMPLETED and (
        feature_engineering.transformations.recommendations
    ):
        notes.append(
            "feature-engineering recommendations are available upstream (Phase 6.3 / 6.4) and "
            "are not executed here"
        )
    if objective_used:
        notes.append(
            "an objective was supplied and recorded; it did not change any candidate family"
        )

    if not candidate_names:
        return ModelCandidates(
            status=ModelingStatus.COMPLETED,
            reason="no model family could be justified from the available structural information",
            candidates=[],
            candidates_detail=[],
            objective_used=objective_used,
            notes=notes,
        )

    return ModelCandidates(
        status=ModelingStatus.COMPLETED,
        reason=None,
        candidates=candidate_names,
        candidates_detail=candidates_detail,
        objective_used=objective_used,
        notes=notes,
    )
