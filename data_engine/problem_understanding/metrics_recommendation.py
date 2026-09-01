"""Phase 5.4 — deterministic, rule-based candidate-metric recommendation.

:func:`recommend_metrics` turns the **Phase-5.3** :class:`TaskTypeInference`
result (plus the DataFrame and the optional objective) into an ordered list
of evaluation metrics and a single ``primary_metric``.

It uses a **small fixed metric vocabulary** per task — it never generates
metric names dynamically, never trains a model, never predicts, never
cross-validates, and never re-infers the target or the task type. The
target column is taken from ``TaskTypeInference.target_column`` (echoed by
Phase 5.3); ``recommend_metrics`` inspects it only for the ``mape``
compatibility rule and never selects a new one.

Conservative: an unavailable / missing / unsupported task type yields
``status = unavailable`` / ``primary_metric = None`` / ``metrics = []``
with a precise reason — a metric is never fabricated.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .models import (
    CandidateMetrics,
    ProblemUnderstandingStatus,
    TaskType,
    TaskTypeInference,
)

_SEPARATORS = re.compile(r"[\s_\-/]+")


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


# --- fixed metric vocabulary (best-first default order) ---------------------

_REGRESSION_METRICS: tuple[str, ...] = ("rmse", "mae", "r2")  # + "mape" when compatible
_BINARY_METRICS: tuple[str, ...] = ("f1", "roc_auc", "precision", "recall", "accuracy")
_MULTICLASS_METRICS: tuple[str, ...] = ("f1_macro", "accuracy", "precision_macro", "recall_macro")
_CLUSTERING_METRICS: tuple[str, ...] = (
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
)
_FORECASTING_METRICS: tuple[str, ...] = ("mae", "rmse")  # + "mape" when compatible

_MAPE_ELIGIBLE_TASKS = frozenset({TaskType.REGRESSION, TaskType.TIME_SERIES_FORECASTING})
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})


def _base_metrics(task: TaskType) -> tuple[str, ...]:
    return {
        TaskType.REGRESSION: _REGRESSION_METRICS,
        TaskType.BINARY_CLASSIFICATION: _BINARY_METRICS,
        TaskType.MULTICLASS_CLASSIFICATION: _MULTICLASS_METRICS,
        TaskType.CLUSTERING: _CLUSTERING_METRICS,
        TaskType.TIME_SERIES_FORECASTING: _FORECASTING_METRICS,
    }[task]


# --- objective vocabulary (fixed; no NLP / stemming / embeddings) -----------

# (phrase, metric) — first match wins per metric; phrases are checked as
# substrings of the normalised objective.
_OBJECTIVE_METRIC_PHRASES: tuple[tuple[str, str], ...] = (
    ("minimize absolute error", "mae"),
    ("minimise absolute error", "mae"),
    ("mean absolute error", "mae"),
    ("absolute error", "mae"),
    ("minimize squared error", "rmse"),
    ("minimise squared error", "rmse"),
    ("mean squared error", "rmse"),
    ("root mean squared", "rmse"),
    ("squared error", "rmse"),
    ("penalize large errors", "rmse"),
    ("penalise large errors", "rmse"),
    ("penalize outliers", "rmse"),
    ("large errors", "rmse"),
    ("percentage error", "mape"),
    ("percent error", "mape"),
    ("relative error", "mape"),
    ("maximize r2", "r2"),
    ("maximise r2", "r2"),
    ("r squared", "r2"),
    ("explained variance", "r2"),
    ("goodness of fit", "r2"),
    ("avoid false positives", "precision"),
    ("minimize false positives", "precision"),
    ("reduce false positives", "precision"),
    ("false alarms", "precision"),
    ("avoid false negatives", "recall"),
    ("minimize false negatives", "recall"),
    ("reduce false negatives", "recall"),
    ("catch all", "recall"),
    ("catch every", "recall"),
    ("sensitivity", "recall"),
    ("balance precision and recall", "f1"),
    ("balance precision", "f1"),
    ("harmonic mean of precision", "f1"),
)
# bare metric names as whole tokens
_OBJECTIVE_METRIC_TOKENS: dict[str, str] = {
    "mae": "mae",
    "rmse": "rmse",
    "mape": "mape",
    "r2": "r2",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
    "roc": "roc_auc",
    "auc": "roc_auc",
}

_IMBALANCE_PHRASES = (
    "imbalanced",
    "imbalance",
    "class imbalance",
    "rare positive",
    "rare event",
    "rare class",
    "class is rare",
    "classes are rare",
    "minority class",
    "skewed classes",
    "skewed class",
    "few positives",
)
_RANKING_PHRASES = ("ranking", "rank the", "learning to rank", "rank order", "rank ordering")

# f1 for a binary task, f1_macro for a multiclass task
_F1_FOR_TASK = {
    TaskType.BINARY_CLASSIFICATION: "f1",
    TaskType.MULTICLASS_CLASSIFICATION: "f1_macro",
}


def _objective_metric_preferences(objective: str) -> tuple[frozenset[str], bool, bool]:
    """Return (preferred metric names, imbalance-signal, ranking-signal)."""
    normalized = _normalize(objective)
    padded = f" {normalized} "
    tokens = frozenset(t for t in normalized.split() if t)

    preferred: set[str] = set()
    for phrase, metric in _OBJECTIVE_METRIC_PHRASES:
        if phrase in normalized:
            preferred.add(metric)
    for token, metric in _OBJECTIVE_METRIC_TOKENS.items():
        if token in tokens:
            preferred.add(metric)

    imbalance = any(p in padded for p in _IMBALANCE_PHRASES)
    ranking = any(p in padded for p in _RANKING_PHRASES)
    return frozenset(preferred), imbalance, ranking


# --- mape compatibility ---------------------------------------------------


def _mape_compatible(df: pd.DataFrame, target_column: str | None) -> tuple[bool, str]:
    if target_column is None:
        return False, "mape omitted: the target column is unknown"
    names = [str(c) for c in df.columns]
    if target_column not in names:
        return False, f"mape omitted: target column '{target_column}' is not in the DataFrame"
    values = pd.to_numeric(df.iloc[:, names.index(target_column)], errors="coerce").to_numpy(
        dtype=float
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False, f"mape omitted: target column '{target_column}' has no finite numeric values"
    if bool((values == 0.0).any()):
        return False, "mape omitted: the target contains zero values"
    if bool((values < 0.0).any()):
        return False, "mape omitted: the target contains negative values"
    return True, "mape included: the target has only positive non-zero values"


# --- result helpers -----------------------------------------------------


def _unavailable(reason: str, *, objective_used: bool) -> CandidateMetrics:
    return CandidateMetrics(
        status=ProblemUnderstandingStatus.UNAVAILABLE,
        reason=reason,
        primary_metric=None,
        metrics=[],
        objective_used=objective_used,
        notes=[],
    )


# --- public API --------------------------------------------------------


def recommend_metrics(
    df: pd.DataFrame,
    task_type: TaskTypeInference,
    *,
    objective: str | None = None,
) -> CandidateMetrics:
    """Deterministically recommend evaluation metrics for the inferred task.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    task_type:
        The **Phase-5.3** :class:`TaskTypeInference` result — the sole
        authority for the task type and the target column. A non-model
        raises ``TypeError``; it is **not mutated** and never re-inferred.
    objective:
        The user's objective, **verbatim and optional** — matched against a
        small fixed vocabulary for metric preferences. Never parsed for
        meaning.

    Returns
    -------
    CandidateMetrics
        ``status = completed`` with ``metrics`` (best-first) and a
        ``primary_metric`` (always one of ``metrics``); otherwise
        ``status = unavailable`` / ``primary_metric = None`` /
        ``metrics = []`` with an explicit ``reason`` (task inference not
        completed, missing task type, or an unsupported task type).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"recommend_metrics expects a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(task_type, TaskTypeInference):
        raise TypeError(
            f"recommend_metrics expects a TaskTypeInference, got {type(task_type).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    if task_type.status is not ProblemUnderstandingStatus.COMPLETED:
        return _unavailable(
            f"task-type inference is not completed ({task_type.reason or 'no reason given'})",
            objective_used=objective_used,
        )
    task = task_type.task_type
    if task is None:
        return _unavailable(
            "task-type inference completed without a task type", objective_used=objective_used
        )
    if task in _UNSUPPORTED_TASKS:
        return _unavailable(
            f"no metric vocabulary is defined for task type '{task.value}'",
            objective_used=objective_used,
        )

    metrics = list(_base_metrics(task))
    notes: list[str] = []

    preferred, imbalance, ranking = (
        _objective_metric_preferences(objective)
        if objective_used and objective is not None
        else (frozenset(), False, False)
    )
    notes.append(
        f"objective metric preferences: {', '.join(sorted(preferred))}"
        if preferred
        else "no objective metric preference detected"
        if objective_used
        else "no objective supplied"
    )

    # mape compatibility (regression / forecasting only)
    if task in _MAPE_ELIGIBLE_TASKS:
        compatible, mape_note = _mape_compatible(df, task_type.target_column)
        notes.append(mape_note)
        if compatible:
            metrics.append("mape")

    # imbalance -> prioritise f1 / f1_macro over accuracy
    if imbalance and task in _F1_FOR_TASK:
        preferred = preferred | {_F1_FOR_TASK[task]}
        notes.append(
            f"objective indicates class imbalance; prioritising {_F1_FOR_TASK[task]} over accuracy"
        )

    if ranking:
        notes.append(
            f"objective mentions ranking; ranking-specific evaluation is not yet supported — "
            f"returning the standard {task.value} metrics"
        )

    # resolve an objective-driven primary metric
    compatible_preferences = sorted(m for m in preferred if m in metrics)
    incompatible_preferences = sorted(m for m in preferred if m not in metrics)
    if incompatible_preferences:
        notes.append(
            f"objective mentions {', '.join(incompatible_preferences)}, which is not a "
            f"{task.value} metric; ignored"
        )

    if compatible_preferences:
        chosen = compatible_preferences[0]
        if len(compatible_preferences) > 1:
            notes.append(
                f"objective expresses several compatible metric preferences "
                f"({', '.join(compatible_preferences)}); using '{chosen}' "
                "(alphabetically first)"
            )
        primary_reason = f"objective prefers '{chosen}'"
        metrics = [chosen, *(m for m in metrics if m != chosen)]
    else:
        primary_reason = f"default {task.value} priority"

    primary = metrics[0]
    notes.insert(
        0,
        f"task type '{task.value}' -> {len(metrics)} candidate metric(s); "
        f"primary '{primary}' ({primary_reason})",
    )

    return CandidateMetrics(
        status=ProblemUnderstandingStatus.COMPLETED,
        reason=None,
        primary_metric=primary,
        metrics=metrics,
        objective_used=objective_used,
        notes=notes,
    )
