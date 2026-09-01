"""Phase 7.2 — deterministic model-readiness assessment.

:func:`assess_model_readiness` answers one narrow question: *is the
available **structural** information (the Phase-5 ``ProblemSpec`` and the
Phase-6 ``FeatureEngineeringSpec``, plus the supplied DataFrame) enough to
proceed to the next model-development stage?*

``ready = True`` means **structurally ready to proceed** — it does **not**
mean "the dataset will produce a good model". Phase 7.2 trains nothing,
fits no estimator, generates no prediction, computes no metric / feature
importance / correlation, performs no preprocessing, and never modifies
the DataFrame or any upstream model.
"""

from __future__ import annotations

import pandas as pd

from data_engine.feature_engineering import FeatureEngineeringSpec, FeatureEngineeringStatus
from data_engine.problem_understanding import (
    ProblemSpec,
    ProblemUnderstandingStatus,
    TaskType,
)

from .models import ModelingStatus, ModelReadiness

# --- tunables (documented in docs/modeling.md) ---------------------------

# Fewer rows than this -> a blocking readiness issue.
MODEL_READINESS_MIN_ROWS = 20
# Fewer rows than this (but >= the minimum) -> a warning.
MODEL_READINESS_ROWS_WARNING = 100

_SUPERVISED_TASKS = frozenset(
    {
        TaskType.REGRESSION,
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.TIME_SERIES_FORECASTING,
    }
)
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})

_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED

_NOTE_STRUCTURAL_ONLY = (
    "model readiness is a structural check only — a ready verdict does not mean the dataset "
    "will produce a good model"
)
_NOTE_NO_EXECUTION = (
    "no model was trained, no estimator fitted, no prediction generated, no metric or feature "
    "importance computed, no preprocessing performed, and the DataFrame was not modified"
)


def _unavailable(reason: str) -> ModelReadiness:
    return ModelReadiness(
        status=ModelingStatus.UNAVAILABLE,
        reason=reason,
        ready=None,
        blocking_issues=[],
        warnings=[],
        notes=[],
    )


def _eligible_features(feature_engineering: FeatureEngineeringSpec) -> list[str]:
    """Structurally eligible feature columns, deterministically ordered."""
    selection = feature_engineering.selection
    if selection.status is _FE_COMPLETED:
        return sorted(set(selection.selected_features) | set(selection.review_features))
    return sorted(feature_engineering.inventory.candidate_features)


def assess_model_readiness(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    *,
    objective: str | None = None,
) -> ModelReadiness:
    """Deterministically assess whether the pipeline is structurally ready.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    problem:
        The **Phase-5** :class:`ProblemSpec`. A non-model raises
        ``TypeError``; it is **not mutated**.
    feature_engineering:
        The **Phase-6** :class:`FeatureEngineeringSpec`. A non-model raises
        ``TypeError``; it is **not mutated**.
    objective:
        The user's objective, **verbatim and optional** — recorded in the
        notes only; it never changes a readiness decision.

    Returns
    -------
    ModelReadiness
        ``status = completed`` with a boolean ``ready`` verdict once the
        upstream task type, feature inventory, and feature-engineering
        assessment are all completed; otherwise ``status = unavailable`` /
        ``ready = None`` with an explicit ``reason``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"assess_model_readiness expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(problem, ProblemSpec):
        raise TypeError(
            f"assess_model_readiness expects a ProblemSpec, got {type(problem).__name__}"
        )
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            "assess_model_readiness expects a FeatureEngineeringSpec, "
            f"got {type(feature_engineering).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED:
        return _unavailable(
            f"task-type inference is not completed (status = {task_inference.status.value})"
        )
    task = task_inference.task_type
    if task is None:
        return _unavailable("task-type inference completed without a task type")
    if task in _UNSUPPORTED_TASKS:
        return _unavailable(f"model readiness does not support task type '{task.value}'")

    if feature_engineering.inventory.status is not _FE_COMPLETED:
        return _unavailable(
            "feature inventory is not completed "
            f"(status = {feature_engineering.inventory.status.value})"
        )
    if feature_engineering.assessment.status is not _FE_COMPLETED:
        return _unavailable(
            "feature-engineering assessment is not completed "
            f"(status = {feature_engineering.assessment.status.value})"
        )

    column_names = [str(c) for c in df.columns]
    column_name_set = set(column_names)
    n_observations = len(df)
    blocking: list[str] = []
    warnings: list[str] = []
    notes: list[str] = [
        _NOTE_STRUCTURAL_ONLY,
        _NOTE_NO_EXECUTION,
        f"task type: {task.value}",
    ]

    # --- dataset size ------------------------------------------------
    sufficient_observations = n_observations >= MODEL_READINESS_MIN_ROWS
    if n_observations < MODEL_READINESS_MIN_ROWS:
        blocking.append(
            f"the dataset has {n_observations} row(s); at least {MODEL_READINESS_MIN_ROWS} "
            "are required to proceed to modeling"
        )
    elif n_observations < MODEL_READINESS_ROWS_WARNING:
        warnings.append(
            f"the dataset has only {n_observations} rows (fewer than "
            f"{MODEL_READINESS_ROWS_WARNING}); model estimates may be unreliable"
        )

    # --- target ----------------------------------------------------
    target_available = False
    target_usable = False
    is_supervised = task in _SUPERVISED_TASKS
    if not is_supervised:
        notes.append(f"task '{task.value}' is targetless; no target column is required")
    else:
        target_section = problem.target
        if target_section.status is not _PU_COMPLETED:
            return _unavailable(
                f"target identification is not completed (status = {target_section.status.value})"
            )
        target_column = target_section.target_column
        if target_column is None:
            blocking.append("no target column was identified for a supervised task")
        else:
            target_available = True
            if target_column not in column_name_set:
                blocking.append(f"the target column '{target_column}' is not in the DataFrame")
            else:
                series = df.iloc[:, column_names.index(target_column)]
                non_null = series.dropna()
                if non_null.empty:
                    blocking.append(f"the target column '{target_column}' is entirely missing")
                elif int(non_null.nunique()) <= 1:
                    blocking.append(f"the target column '{target_column}' is constant")
                else:
                    target_usable = True
                    n_missing = int(series.isna().sum())
                    if n_missing:
                        warnings.append(
                            f"the target column '{target_column}' has {n_missing} missing "
                            "value(s); rows without a target cannot be used for supervised "
                            "training"
                        )

    # --- Phase-5 feasibility (advisory) ---------------------------
    feasibility = problem.feasibility
    if feasibility.status is _PU_COMPLETED:
        if feasibility.feasible is False:
            blocking.append(
                "the Phase 5 feasibility assessment found the problem infeasible: "
                + (feasibility.reason or "no reason given")
            )
        elif feasibility.warnings:
            warnings.append(
                f"the Phase 5 feasibility assessment passed with {len(feasibility.warnings)} "
                "warning(s)"
            )
    else:
        notes.append("Phase 5 feasibility assessment was not run (advisory only)")

    # --- eligible features -------------------------------------
    eligible = _eligible_features(feature_engineering)
    eligible_feature_count = len(eligible)
    if eligible_feature_count == 0:
        blocking.append("no structurally eligible feature columns are available for modeling")

    # --- feature-engineering assessment ----------------------
    fe_assessment = feature_engineering.assessment
    feature_engineering_assessment_usable = fe_assessment.feasible is not None
    if fe_assessment.feasible is False:
        first = (
            fe_assessment.blocking_issues[0]
            if fe_assessment.blocking_issues
            else (fe_assessment.reason or "no reason given")
        )
        blocking.append(
            "the Phase 6.6 feature-engineering assessment found the pipeline structurally "
            f"infeasible: {first}"
        )
    elif fe_assessment.feasible is True and fe_assessment.warnings:
        warnings.append(
            f"the Phase 6.6 feature-engineering assessment passed with "
            f"{len(fe_assessment.warnings)} warning(s)"
        )

    # --- preprocessing requirements (advisory) ---------------
    preprocessing = feature_engineering.preprocessing
    preprocessing_requirements_present = preprocessing.status is _FE_COMPLETED and bool(
        preprocessing.required_operations
    )
    if preprocessing_requirements_present:
        warnings.append(
            "Phase 6.5 identified preprocessing requirements ("
            + ", ".join(preprocessing.required_operations)
            + ") that must be applied before training — Phase 7.2 does not apply them"
        )
    elif preprocessing.status is not _FE_COMPLETED:
        notes.append("Phase 6.5 preprocessing requirements were not identified (advisory only)")

    if objective_used:
        notes.append(
            "an objective was supplied and recorded; it did not change any readiness check"
        )

    blocking.sort()
    warnings.sort()
    ready = len(blocking) == 0

    return ModelReadiness(
        status=ModelingStatus.COMPLETED,
        reason=None if ready else f"{len(blocking)} blocking readiness issue(s): {blocking[0]}",
        ready=ready,
        target_available=target_available,
        target_usable=target_usable,
        eligible_feature_count=eligible_feature_count,
        feature_engineering_assessment_usable=feature_engineering_assessment_usable,
        preprocessing_requirements_present=preprocessing_requirements_present,
        sufficient_observations=sufficient_observations,
        n_observations=n_observations,
        blocking_issues=blocking,
        warnings=warnings,
        notes=notes,
    )
