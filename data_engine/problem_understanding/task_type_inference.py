"""Phase 5.3 — deterministic, rule-based ML task-type inference.

:func:`infer_task_type` decides which :class:`TaskType` best describes the
problem, from three inputs only:

* the DataFrame's target column (dtype + class count) and whether the
  frame has any datetime column;
* the **Phase-5.2** :class:`TargetIdentification` result (authoritative
  for *which* column is the target — this function never re-selects one);
* the user's optional **explicit** objective, matched against a small
  fixed vocabulary (no LLM, no embeddings, no stemmer, no fuzzy search).

Conservative: if the evidence is missing or contradictory it returns
``status = unavailable`` / ``task_type = None`` with an explicit reason
rather than guessing a task type.

Analysis-only: it never mutates ``df`` or ``target``, writes a file,
creates a figure, or touches lineage / versions.
"""

from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    ProblemUnderstandingStatus,
    TargetIdentification,
    TaskType,
    TaskTypeInference,
)

# A numeric target with *more* than this many distinct values is never treated
# as an (integer-encoded) classification target, even with a classify objective.
NUMERIC_CLASS_MAX = 10

_SEPARATORS = re.compile(r"[\s_\-/]+")


def _normalize(text: str) -> str:
    return _SEPARATORS.sub(" ", text.strip().lower()).strip()


# --- objective vocabulary (documented in docs/problem-understanding.md) -------

_REGRESSION_WORDS = frozenset(
    {
        "regression",
        "estimate",
        "estimating",
        "estimation",
        "price",
        "prices",
        "amount",
        "amounts",
        "revenue",
        "sales",
        "cost",
        "costs",
        "value",
        "values",
        "continuous",
        "magnitude",
        "duration",
        "score",
        "scores",
        "rating",
        "ratings",
        "quantity",
    }
)
_REGRESSION_PHRASES = ("how much", "how many", "dollar amount", "numeric value")

_CLASSIFICATION_WORDS = frozenset(
    {
        "classify",
        "classifying",
        "classification",
        "categorise",
        "categorize",
        "categorisation",
        "categorization",
        "label",
        "labels",
        "category",
        "categories",
        "churn",
        "churned",
        "fraud",
        "fraudulent",
        "spam",
        "whether",
        "binary",
        "class",
        "classes",
    }
)
_CLASSIFICATION_PHRASES = ("yes or no", "yes / no", "true or false")

_MULTICLASS_PHRASES = (
    "multiclass",
    "multi class",
    "multi-class",
    "several classes",
    "one of several",
    "multiple classes",
)
_MULTILABEL_PHRASES = (
    "multilabel",
    "multi label",
    "multi-label",
    "multiple labels",
    "multiple categories per",
    "several labels",
)

_CLUSTERING_WORDS = frozenset(
    {"cluster", "clusters", "clustering", "segment", "segments", "segmentation", "unsupervised"}
)
_CLUSTERING_PHRASES = (
    "group customers",
    "group users",
    "group the customers",
    "discover groups",
    "find groups",
    "customer groups",
    "user groups",
    "into groups",
)

_FORECASTING_WORDS = frozenset({"forecast", "forecasting", "forecasted"})
_FORECASTING_PHRASES = (
    "next month",
    "next week",
    "next day",
    "next year",
    "next quarter",
    "time series",
    "time-series",
    "future value",
    "future values",
    "over time",
    "in the future",
    "future sales",
    "future demand",
)


def _objective_signals(objective: str) -> frozenset[str]:
    """Deterministic subset of {regression, classification, multiclass,
    multilabel, clustering, forecasting} evidenced by the objective."""
    normalized = _normalize(objective)
    padded = f" {normalized} "
    tokens = frozenset(t for t in normalized.split() if t)

    def has_word(words: frozenset[str]) -> bool:
        return bool(tokens & words)

    def has_phrase(phrases: tuple[str, ...]) -> bool:
        return any(p in padded for p in phrases)

    signals: set[str] = set()
    if has_word(_REGRESSION_WORDS) or has_phrase(_REGRESSION_PHRASES):
        signals.add("regression")
    if has_word(_CLASSIFICATION_WORDS) or has_phrase(_CLASSIFICATION_PHRASES):
        signals.add("classification")
    if has_phrase(_MULTICLASS_PHRASES):
        signals.add("multiclass")
        signals.add("classification")
    if has_phrase(_MULTILABEL_PHRASES):
        signals.add("multilabel")
    if has_word(_CLUSTERING_WORDS) or has_phrase(_CLUSTERING_PHRASES):
        signals.add("clustering")
    if has_word(_FORECASTING_WORDS) or has_phrase(_FORECASTING_PHRASES) or "future" in tokens:
        signals.add("forecasting")
    return frozenset(signals)


# --- result helpers ---------------------------------------------------------


def _unavailable(
    reason: str,
    *,
    objective_used: bool,
    target_column: str | None = None,
    notes: list[str] | None = None,
) -> TaskTypeInference:
    return TaskTypeInference(
        status=ProblemUnderstandingStatus.UNAVAILABLE,
        reason=reason,
        task_type=None,
        target_column=target_column,
        objective_used=objective_used,
        notes=notes or [],
    )


def _completed(
    task_type: TaskType,
    primary: str,
    *,
    objective_used: bool,
    target_column: str | None = None,
    extra: list[str] | None = None,
) -> TaskTypeInference:
    return TaskTypeInference(
        status=ProblemUnderstandingStatus.COMPLETED,
        reason=None,
        task_type=task_type,
        target_column=target_column,
        objective_used=objective_used,
        notes=[primary, *(extra or [])],
    )


def _structural_task(
    column_type: ColumnType,
    n_classes: int,
    is_integer_like: bool,
    has_class_objective: bool,
) -> tuple[TaskType, str]:
    """The task type supported by the target's dtype + class count alone."""
    if column_type is ColumnType.BOOLEAN:
        return TaskType.BINARY_CLASSIFICATION, "boolean target -> binary classification"
    if column_type is ColumnType.CATEGORICAL:
        if n_classes == 2:
            return (
                TaskType.BINARY_CLASSIFICATION,
                "categorical target with 2 distinct classes -> binary classification",
            )
        return (
            TaskType.MULTICLASS_CLASSIFICATION,
            f"categorical target with {n_classes} distinct classes -> multiclass classification",
        )
    # numeric
    if has_class_objective and n_classes == 2:
        return (
            TaskType.BINARY_CLASSIFICATION,
            (
                "numeric target with 2 distinct values + classification objective "
                "-> binary classification"
            ),
        )
    if has_class_objective and is_integer_like and 3 <= n_classes <= NUMERIC_CLASS_MAX:
        return (
            TaskType.MULTICLASS_CLASSIFICATION,
            (
                f"integer-coded target with {n_classes} distinct values + classification "
                "objective -> multiclass classification"
            ),
        )
    return TaskType.REGRESSION, "numeric target -> regression"


# --- public API ------------------------------------------------------------


def infer_task_type(
    df: pd.DataFrame,
    target: TargetIdentification,
    *,
    objective: str | None = None,
) -> TaskTypeInference:
    """Deterministically infer the ML task type.

    Parameters
    ----------
    df:
        The dataset. **Not mutated.** A non-DataFrame raises ``TypeError``.
    target:
        The **Phase-5.2** :class:`TargetIdentification` result — the sole
        authority for which column is the target. A non-model raises
        ``TypeError``; it is **not mutated** and never re-selected from.
    objective:
        The user's objective, **verbatim and optional** — used only for the
        transparent vocabulary matching. Never parsed for meaning.

    Returns
    -------
    TaskTypeInference
        ``status = completed`` + ``task_type`` when the evidence supports a
        single task; otherwise ``status = unavailable`` + ``task_type =
        None`` + an explicit ``reason`` (no target pinned, target column
        missing / all-missing / constant, datetime target without
        forecasting evidence, or an unrecognised target type). Never
        fabricates a task type.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"infer_task_type expects a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(target, TargetIdentification):
        raise TypeError(
            f"infer_task_type expects a TargetIdentification, got {type(target).__name__}"
        )

    resolved_target_column = target.target_column

    objective_used = objective is not None and objective.strip() != ""
    signals = (
        _objective_signals(objective) if objective_used and objective is not None else frozenset()
    )
    signal_note = (
        f"objective signals: {', '.join(sorted(signals))}"
        if signals
        else "no task-type signal detected in the objective"
        if objective_used
        else "no objective supplied"
    )

    # --- no single target column -------------------------------------------
    if target.status is ProblemUnderstandingStatus.UNAVAILABLE:
        return _unavailable(
            f"target identification was unavailable ({target.reason or 'no reason given'})",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )
    if target.target_column is None:
        if "clustering" in signals:
            return _completed(
                TaskType.CLUSTERING,
                "no target column was identified and the objective indicates a clustering "
                "(inherently targetless) task",
                objective_used=objective_used,
                target_column=resolved_target_column,
                extra=[signal_note],
            )
        notes = [signal_note]
        if target.candidate_columns:
            notes.append(
                "target candidates exist but none was decisive; task inference does not "
                "select a target"
            )
        return _unavailable(
            "cannot infer task type because no single target column was identified",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=notes,
        )

    # --- validate the identified target column ----------------------------
    column = target.target_column
    column_names = [str(c) for c in df.columns]
    if column not in column_names:
        return _unavailable(
            f"the identified target column '{column}' is not in the DataFrame",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )
    series = df.iloc[:, column_names.index(column)]
    non_null = series.dropna()
    if non_null.empty:
        return _unavailable(
            f"the target column '{column}' is entirely missing",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )
    n_classes = int(non_null.nunique())
    if n_classes <= 1:
        return _unavailable(
            f"the target column '{column}' is constant",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )

    column_type = infer_column_type(series)
    has_class_objective = "classification" in signals or "multiclass" in signals

    # --- datetime target -------------------------------------------------
    if column_type is ColumnType.DATETIME:
        if "forecasting" in signals:
            return _completed(
                TaskType.TIME_SERIES_FORECASTING,
                f"datetime target '{column}' with an explicit forecasting objective "
                "-> time_series_forecasting",
                objective_used=objective_used,
                target_column=resolved_target_column,
                extra=[signal_note],
            )
        return _unavailable(
            f"the target column '{column}' is a datetime; a datetime target alone is not "
            "sufficient evidence for time_series_forecasting",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )
    if column_type is ColumnType.UNKNOWN:
        return _unavailable(
            f"the target column '{column}' has an unrecognised type",
            objective_used=objective_used,
            target_column=resolved_target_column,
            notes=[signal_note],
        )

    is_integer_like = bool(ptypes.is_integer_dtype(series))
    structural, structural_reason = _structural_task(
        column_type, n_classes, is_integer_like, has_class_objective
    )

    extra: list[str] = [signal_note]
    if (
        column_type is ColumnType.NUMERIC
        and structural is TaskType.REGRESSION
        and has_class_objective
    ):
        extra.append(
            f"the objective suggests classification, but the target '{column}' is continuous "
            f"numeric ({n_classes} distinct values); using the structurally supported task "
            "(regression)"
        )
    if (
        column_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
        and "regression" in signals
        and "classification" not in signals
    ):
        extra.append(
            f"the objective suggests regression, but the target '{column}' is "
            f"{column_type.value}; using the structurally supported task"
        )
    if "multilabel" in signals:
        extra.append(
            "the objective mentions multi-label, but DataPilot's tabular data model provides no "
            "per-row multi-label structural signal; treating as a single-label task"
        )

    task = structural
    primary = structural_reason
    if task is TaskType.REGRESSION and "forecasting" in signals:
        has_datetime_column = any(
            infer_column_type(df.iloc[:, i]) is ColumnType.DATETIME for i in range(df.shape[1])
        )
        if has_datetime_column:
            task = TaskType.TIME_SERIES_FORECASTING
            primary = (
                f"numeric target '{column}' + a forecasting objective + a datetime column present "
                "-> time_series_forecasting"
            )
        else:
            extra.append(
                "the objective mentions forecasting but the DataFrame has no datetime column; "
                "treating as regression"
            )

    return _completed(
        task,
        primary,
        objective_used=objective_used,
        target_column=resolved_target_column,
        extra=extra,
    )
