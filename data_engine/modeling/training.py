"""Phase 7.4 — deterministic baseline model training & evaluation.

:func:`train_and_evaluate_models` is the **first** DataPilot component
allowed to fit estimators and compute evaluation metrics. It:

* consumes the Phase-5 :class:`ProblemSpec`, the Phase-6
  :class:`FeatureEngineeringSpec`, and the Phase-7.2 / 7.3
  :class:`ModelReadiness` / :class:`DataSplitPlan` / :class:`ModelCandidates`;
* performs the actual train / validation / test split **exactly** as the
  :class:`DataSplitPlan` specifies, with a fixed random seed;
* executes only the Phase-6.5 preprocessing requirements, fitted **only**
  on the training partition (leakage-safe within this pipeline);
* fits one conservative baseline scikit-learn estimator per Phase-7.3
  candidate family;
* computes task-appropriate deterministic metrics on the test partition;
* returns a structured, JSON-primitive-only :class:`TrainingOutcome`.

It does **not** select, rank, or recommend a model; tune hyperparameters;
cross-validate; do model-based feature selection, SHAP, feature
importance, or leakage detection; generate features; re-infer the target
or task; use target encoding / SMOTE / PCA; persist any artifact; or
modify the input DataFrame or any upstream model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from time import perf_counter as _perf_counter
from typing import Any

import numpy as np
import pandas as pd

from data_engine.feature_engineering import FeatureEngineeringSpec, FeatureEngineeringStatus
from data_engine.problem_understanding import ProblemSpec, ProblemUnderstandingStatus, TaskType
from data_engine.profiling.type_inference import infer_column_type
from datapilot.contracts import ColumnType

from .models import (
    DataSplitPlan,
    DataSplitStrategy,
    DeepTuneResult,
    ExpandedCandidateResult,
    ExpandedSearchResult,
    ModelCandidates,
    ModelFamily,
    ModelingStatus,
    ModelReadiness,
    TrainingOutcome,
    TrainingRun,
    TrainingRunStatus,
)
from .selection import _TASK_SELECTION_METRIC
from .temporal_execution import (
    build_calendar_features,
    build_temporal_features,
    temporal_feature_spec,
)

# --- fixed, documented tunables --------------------------------------------

# The one random seed used everywhere randomisation is required. A named
# module constant — never generated from the clock, environment, or system
# state.
MODEL_TRAINING_RANDOM_SEED = 42

MODEL_TRAINING_TREE_MAX_DEPTH = 8  # conservative depth cap for a baseline tree
MODEL_TRAINING_FOREST_N_ESTIMATORS = 100  # sklearn default; explicit for reproducibility
MODEL_TRAINING_KNN_N_NEIGHBORS = 5  # sklearn default; explicit
MODEL_TRAINING_N_CLUSTERS = (
    3  # fixed baseline cluster count (cluster-count selection is out of scope)
)
MODEL_TRAINING_LOGREG_MAX_ITER = 1000  # allow convergence on scaled features
MODEL_TRAINING_MLP_MAX_ITER = 200  # modest cap for the optional neural baseline
MODEL_TRAINING_METRIC_ROUND = 6  # decimal places for every reported metric
MODEL_TRAINING_MIN_TRAIN_ROWS = 5  # fewer -> the run is unavailable
MODEL_TRAINING_MIN_TEST_ROWS = 1  # fewer -> the run is unavailable
# Forecasting-execution: after building lag / rolling features and dropping the
# warm-up rows, fewer modelable rows than this -> the whole outcome is unavailable.
MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS = 20

_PU_COMPLETED = ProblemUnderstandingStatus.COMPLETED
_FE_COMPLETED = FeatureEngineeringStatus.COMPLETED
_UNSUPPORTED_TASKS = frozenset({TaskType.MULTILABEL_CLASSIFICATION, TaskType.OTHER})
_TASK_CATEGORY: dict[TaskType, str] = {
    TaskType.REGRESSION: "regression",
    TaskType.TIME_SERIES_FORECASTING: "regression",
    TaskType.BINARY_CLASSIFICATION: "classification",
    TaskType.MULTICLASS_CLASSIFICATION: "classification",
    TaskType.CLUSTERING: "clustering",
}

_OP_IMPUTATION = "missing-value imputation"
_OP_ENCODING = "categorical encoding"
_OP_SCALING = "numerical scaling"

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

try:  # pragma: no cover - environment dependent
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# --- helpers -------------------------------------------------------------


def _normalise_error(message: str) -> str:
    """Strip nondeterministic detail (memory addresses) from an exception message."""
    return _ADDR_RE.sub("0x...", message).strip()


def _verify_chronological_order(df: pd.DataFrame, time_column: str | None) -> tuple[bool, str]:
    """Deterministically check that a forecasting frame's rows are chronological.

    For ``time_ordered_holdout`` the **row order is the time axis** — Phase
    7.4 slices it positionally and never sorts or infers a time column.

    * ``time_column`` given (the Phase-5.3 resolved axis) — that exact column
      must be present and non-decreasing.
    * ``time_column`` ``None`` (a caller not going through Phase 5) — fall back
      to the heuristic: the frame must be non-decreasing on one of its own
      datetime columns.

    Returns ``(ok, detail)`` where ``detail`` is a note on success or an
    explicit reason on failure.
    """
    column_names = [str(c) for c in df.columns]

    if time_column is not None:
        if time_column not in column_names:
            return False, (f"the declared time column '{time_column}' is not in the DataFrame")
        parsed = pd.to_datetime(
            df.iloc[:, column_names.index(time_column)], errors="coerce"
        ).dropna()
        if len(parsed) < 2:
            return False, (
                f"the declared time column '{time_column}' has fewer than 2 usable timestamps"
            )
        if parsed.is_monotonic_increasing:
            return True, (
                f"chronological-order precondition satisfied: the declared time column "
                f"'{time_column}' is non-decreasing across the supplied rows"
            )
        return False, (
            f"the rows are not in chronological order on the declared time column "
            f"'{time_column}'; sort the DataFrame chronologically before modeling — Phase 7.4 "
            "does not reorder rows"
        )

    datetime_cols = sorted(
        name
        for i, name in enumerate(column_names)
        if infer_column_type(df.iloc[:, i]) is ColumnType.DATETIME
    )
    if not datetime_cols:
        return False, (
            "time_series_forecasting uses row order as the time axis, but the DataFrame has no "
            "datetime column to verify that the rows are in chronological order; sort the rows "
            "chronologically and keep the timestamp column in the frame"
        )
    for name in datetime_cols:
        parsed = pd.to_datetime(df.iloc[:, column_names.index(name)], errors="coerce").dropna()
        if len(parsed) >= 2 and parsed.is_monotonic_increasing:
            return True, (
                f"chronological-order precondition satisfied: datetime column '{name}' is "
                "non-decreasing across the supplied rows"
            )
    return False, (
        "time_series_forecasting uses row order as the time axis, but none of the datetime "
        f"column(s) ({', '.join(datetime_cols)}) is non-decreasing across the supplied rows; "
        "sort the DataFrame chronologically before modeling — Phase 7.4 does not reorder rows"
    )


def _unavailable(reason: str, *, objective_used: bool) -> TrainingOutcome:
    return TrainingOutcome(
        status=ModelingStatus.UNAVAILABLE,
        reason=reason,
        runs=[],
        successful_runs=[],
        failed_runs=[],
        objective_used=objective_used,
        notes=[],
    )


def _eligible_features(feature_engineering: FeatureEngineeringSpec) -> list[str]:
    selection = feature_engineering.selection
    if selection.status is _FE_COMPLETED:
        return sorted(set(selection.selected_features) | set(selection.review_features))
    return sorted(feature_engineering.inventory.candidate_features)


def _round(value: float) -> float:
    if not math.isfinite(value):
        return value
    return round(float(value), MODEL_TRAINING_METRIC_ROUND)


def _build_estimator(family: ModelFamily, category: str) -> tuple[str, Any] | None:
    """The fixed, documented family -> concrete baseline estimator mapping."""
    seed = MODEL_TRAINING_RANDOM_SEED
    if family is ModelFamily.LINEAR:
        if category == "regression":
            from sklearn.linear_model import LinearRegression

            return "LinearRegression", LinearRegression()
        if category == "classification":
            from sklearn.linear_model import LogisticRegression

            return "LogisticRegression", LogisticRegression(
                max_iter=MODEL_TRAINING_LOGREG_MAX_ITER, random_state=seed
            )
    elif family is ModelFamily.TREE_BASED:
        if category == "regression":
            from sklearn.tree import DecisionTreeRegressor

            return "DecisionTreeRegressor", DecisionTreeRegressor(
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH, random_state=seed
            )
        if category == "classification":
            from sklearn.tree import DecisionTreeClassifier

            return "DecisionTreeClassifier", DecisionTreeClassifier(
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH, random_state=seed
            )
    elif family is ModelFamily.ENSEMBLE:
        if category == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return "RandomForestRegressor", RandomForestRegressor(
                n_estimators=MODEL_TRAINING_FOREST_N_ESTIMATORS,
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH,
                random_state=seed,
                n_jobs=1,
            )
        if category == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return "RandomForestClassifier", RandomForestClassifier(
                n_estimators=MODEL_TRAINING_FOREST_N_ESTIMATORS,
                max_depth=MODEL_TRAINING_TREE_MAX_DEPTH,
                random_state=seed,
                n_jobs=1,
            )
    elif family is ModelFamily.PROBABILISTIC:
        if category == "classification":
            from sklearn.naive_bayes import GaussianNB

            return "GaussianNB", GaussianNB()
        if category == "clustering":
            from sklearn.mixture import GaussianMixture

            return "GaussianMixture", GaussianMixture(
                n_components=MODEL_TRAINING_N_CLUSTERS, random_state=seed
            )
    elif family is ModelFamily.DISTANCE_BASED:
        if category == "regression":
            from sklearn.neighbors import KNeighborsRegressor

            return "KNeighborsRegressor", KNeighborsRegressor(
                n_neighbors=MODEL_TRAINING_KNN_N_NEIGHBORS, n_jobs=1
            )
        if category == "classification":
            from sklearn.neighbors import KNeighborsClassifier

            return "KNeighborsClassifier", KNeighborsClassifier(
                n_neighbors=MODEL_TRAINING_KNN_N_NEIGHBORS, n_jobs=1
            )
        if category == "clustering":
            from sklearn.cluster import KMeans

            return "KMeans", KMeans(
                n_clusters=MODEL_TRAINING_N_CLUSTERS, random_state=seed, n_init=10
            )
    elif family is ModelFamily.NEURAL:
        if category == "regression":
            from sklearn.neural_network import MLPRegressor

            return "MLPRegressor", MLPRegressor(
                max_iter=MODEL_TRAINING_MLP_MAX_ITER, random_state=seed
            )
        if category == "classification":
            from sklearn.neural_network import MLPClassifier

            return "MLPClassifier", MLPClassifier(
                max_iter=MODEL_TRAINING_MLP_MAX_ITER, random_state=seed
            )
    return None


_MAX_ONEHOT_CATEGORIES = 30
_OUTLIER_HEAVY_THRESHOLD = 0.05


def _skewness(values: Any) -> float:
    """The (biased) sample skewness of `values`, computed directly rather
    than via `pandas.Series.skew` (whose stub return type is too broad for
    `float(...)` to type-check cleanly here). 0.0 for too-few/constant
    values, matching `pandas`' own convention of a non-informative result
    rather than raising."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 3:
        return 0.0
    std = float(arr.std())
    if std == 0.0:
        return 0.0
    return float(np.mean(((arr - arr.mean()) / std) ** 3))


def _is_outlier_heavy(
    x_train: pd.DataFrame, numeric_cols: list[str], threshold: float = _OUTLIER_HEAVY_THRESHOLD
) -> bool:
    """True when more than `threshold` of numeric training values fall outside
    1.5x IQR of their own column — the standard Tukey fence. `StandardScaler`
    centers on the mean / scales by standard deviation, both of which heavy
    outliers distort; `RobustScaler` (median / IQR) is used instead when this
    is true. Computed on the training partition only (leakage-safe)."""
    if not numeric_cols:
        return False
    total = 0
    outliers = 0
    for col in numeric_cols:
        series = pd.to_numeric(x_train[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers += int(((series < lower) | (series > upper)).sum())
        total += len(series)
    return total > 0 and (outliers / total) > threshold


def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    req_by_col: dict[str, set[str]],
    *,
    robust_scaling: bool = False,
    max_onehot_categories: int = _MAX_ONEHOT_CATEGORIES,
) -> tuple[Any, str | None]:
    """A leakage-safe ColumnTransformer built strictly from Phase-6.5 requirements.

    `robust_scaling=True` swaps `StandardScaler` for `RobustScaler` (median /
    IQR based, insensitive to outliers) — see `_is_outlier_heavy`.
    `max_onehot_categories` caps each categorical column's one-hot expansion
    (sklearn's own `OneHotEncoder(max_categories=...)`): a near-unique column
    (e.g. a customer/row id) would otherwise explode into one column per
    distinct value, which both hurts accuracy (the encoding carries no
    generalisable signal) and training speed; categories beyond the cap are
    bucketed into a single "infrequent" column instead of being dropped.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        numeric_ops = {op for col in numeric_cols for op in req_by_col.get(col, set())}
        steps: list[tuple[str, Any]] = []
        if _OP_IMPUTATION in numeric_ops:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        if _OP_SCALING in numeric_ops:
            scaler = RobustScaler() if robust_scaling else StandardScaler()
            steps.append(("scaler", scaler))
        numeric_pipeline = Pipeline(steps) if steps else "passthrough"
        transformers.append(("numeric", numeric_pipeline, sorted(numeric_cols)))

    if categorical_cols:
        categorical_ops = {op for col in categorical_cols for op in req_by_col.get(col, set())}
        if _OP_ENCODING not in categorical_ops:
            return None, (
                "categorical feature(s) are present but Phase 6.5 identified no encoding "
                "requirement; Phase 7.4 does not invent an encoder"
            )
        steps = []
        if _OP_IMPUTATION in categorical_ops:
            steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        steps.append(
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=max_onehot_categories,
                ),
            )
        )
        transformers.append(("categorical", Pipeline(steps), sorted(categorical_cols)))

    if not transformers:
        return "passthrough", None
    return ColumnTransformer(transformers, remainder="drop"), None


def _split_indices(
    n: int, plan: DataSplitPlan, y: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Deterministic train / validation / test index partitions (positions into df_work)."""
    train_f = plan.train_fraction or 0.0
    test_f = plan.test_fraction or 0.0
    val_f = plan.validation_fraction
    notes: list[str] = []

    all_idx = np.arange(n)

    if plan.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT:
        n_train = round(n * train_f)
        n_val = round(n * val_f) if val_f else 0
        n_train = max(0, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        return (
            all_idx[:n_train],
            all_idx[n_train : n_train + n_val],
            all_idx[n_train + n_val :],
            notes,
        )

    stratified = plan.strategy is DataSplitStrategy.STRATIFIED_HOLDOUT and y is not None
    if stratified and y is not None:
        from sklearn.model_selection import train_test_split

        try:
            rest, test_idx = train_test_split(
                all_idx,
                test_size=test_f,
                stratify=y,
                random_state=MODEL_TRAINING_RANDOM_SEED,
                shuffle=True,
            )
            if val_f:
                rel_val = val_f / max(1e-9, (1.0 - test_f))
                train_idx, val_idx = train_test_split(
                    rest,
                    test_size=rel_val,
                    stratify=y[rest],
                    random_state=MODEL_TRAINING_RANDOM_SEED,
                    shuffle=True,
                )
            else:
                train_idx, val_idx = rest, np.empty(0, dtype=int)
            return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx), notes
        except ValueError:
            notes.append(
                "stratified split was not possible (a class has too few members); falling "
                "back to a shuffled random holdout"
            )

    rng = np.random.default_rng(MODEL_TRAINING_RANDOM_SEED)
    perm = rng.permutation(n)
    n_train = round(n * train_f)
    n_val = round(n * val_f) if val_f else 0
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    return (
        np.sort(perm[:n_train]),
        np.sort(perm[n_train : n_train + n_val]),
        np.sort(perm[n_train + n_val :]),
        notes,
    )


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(y_true, y_pred))
    metrics: dict[str, float] = {
        "mse": _round(mse),
        "rmse": _round(math.sqrt(mse)),
        "mae": _round(mean_absolute_error(y_true, y_pred)),
    }
    if float(np.var(y_true)) > 0.0:
        metrics["r2"] = _round(r2_score(y_true, y_pred))
    return metrics


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {
        "accuracy": _round(accuracy_score(y_true, y_pred)),
        "precision": _round(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": _round(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": _round(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None and n_classes == 2 and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = _round(roc_auc_score(y_true, proba))
        except ValueError:
            pass
    return metrics


def _recursive_horizon_metrics(
    pipeline: Any,
    x_all: pd.DataFrame,
    y_all: np.ndarray,
    feature_cols: list[str],
    target_spec: dict[str, tuple[str, str, int]],
    test_start: int,
    n_rows: int,
    horizon: int,
) -> dict[str, float]:
    """Rolling-origin **recursive** multi-step RMSE, one value per horizon step.

    From every origin ``o`` in ``[max(test_start, max_lookback), n_rows - horizon]``
    forecast ``horizon`` steps: a target lag / rolling feature is rebuilt from the
    growing ``actuals[:o] + predictions[o:]`` history; every other feature
    (exogenous, calendar) uses its **actual** row value (the standard assumption
    that regressors and the calendar are known over the forecast window).
    Deterministic; no metric here ever feeds ``select_model`` — the selection
    metric stays the one-step ``rmse``.
    """
    y = np.asarray(y_all, dtype=float)
    actual_features = {c: x_all[c].to_numpy() for c in feature_cols}
    target_kinds = {c: target_spec[c] for c in feature_cols if c in target_spec}
    max_lookback = max((n for _, _, n in target_kinds.values()), default=0)
    start = max(test_start, max_lookback)
    per_h_sq: list[list[float]] = [[] for _ in range(horizon)]

    for origin in range(start, n_rows - horizon + 1):
        preds: list[float] = []
        for step in range(horizon):
            pos = origin + step
            history = np.concatenate([y[:origin], np.asarray(preds, dtype=float)])
            row: dict[str, float] = {}
            for col in feature_cols:
                spec = target_kinds.get(col)
                if spec is None:
                    row[col] = float(actual_features[col][pos])
                    continue
                _, kind, k = spec
                if kind == "lag":
                    row[col] = float(history[pos - k]) if pos - k >= 0 else float("nan")
                else:
                    window = history[pos - k : pos]
                    if window.size == k:
                        row[col] = float(
                            window.mean() if kind == "rollmean" else window.std(ddof=1)
                        )
                    else:
                        row[col] = float("nan")
            pred = float(np.asarray(pipeline.predict(pd.DataFrame([row], columns=feature_cols)))[0])
            preds.append(pred)
            actual = y[pos]
            if math.isfinite(actual) and math.isfinite(pred):
                per_h_sq[step].append((pred - actual) ** 2)

    metrics: dict[str, float] = {}
    for h in range(horizon):
        if per_h_sq[h]:
            metrics[f"rmse_h{h + 1}"] = _round(math.sqrt(sum(per_h_sq[h]) / len(per_h_sq[h])))
    return metrics


def _clustering_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return {}
    return {
        "silhouette_score": _round(silhouette_score(features, labels)),
        "calinski_harabasz_score": _round(calinski_harabasz_score(features, labels)),
        "davies_bouldin_score": _round(davies_bouldin_score(features, labels)),
    }


# --- public API --------------------------------------------------------


def train_and_evaluate_models(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    candidates: ModelCandidates,
    *,
    objective: str | None = None,
    forecast_horizon: int = 1,
) -> TrainingOutcome:
    """Deterministically train & evaluate one baseline estimator per candidate.

    See the module docstring for the full boundary. ``status = unavailable``
    when an upstream contract does not permit execution; ``status =
    completed`` once execution was permitted (even if every individual
    candidate failed). Phase 7.4 never selects a model.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"train_and_evaluate_models expects a pandas DataFrame, got {type(df).__name__}"
        )
    if not isinstance(problem, ProblemSpec):
        raise TypeError(
            f"train_and_evaluate_models expects a ProblemSpec, got {type(problem).__name__}"
        )
    if not isinstance(feature_engineering, FeatureEngineeringSpec):
        raise TypeError(
            "train_and_evaluate_models expects a FeatureEngineeringSpec, "
            f"got {type(feature_engineering).__name__}"
        )
    if not isinstance(readiness, ModelReadiness):
        raise TypeError(
            f"train_and_evaluate_models expects a ModelReadiness, got {type(readiness).__name__}"
        )
    if not isinstance(split, DataSplitPlan):
        raise TypeError(
            f"train_and_evaluate_models expects a DataSplitPlan, got {type(split).__name__}"
        )
    if not isinstance(candidates, ModelCandidates):
        raise TypeError(
            f"train_and_evaluate_models expects a ModelCandidates, got {type(candidates).__name__}"
        )

    objective_used = objective is not None and objective.strip() != ""

    # --- deterministic upstream precedence ---------------------------
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
            f"model training does not support task type '{task.value}'",
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
            f"training is blocked by model-readiness issues: {first}",
            objective_used=objective_used,
        )
    if split.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"the data-split plan is not completed (status = {split.status.value})",
            objective_used=objective_used,
        )
    if candidates.status is not ModelingStatus.COMPLETED:
        return _unavailable(
            f"model candidates are not available (status = {candidates.status.value})",
            objective_used=objective_used,
        )
    if feature_engineering.assessment.status is not _FE_COMPLETED:
        return _unavailable(
            "feature-engineering assessment is not completed "
            f"(status = {feature_engineering.assessment.status.value})",
            objective_used=objective_used,
        )
    if not _SKLEARN_AVAILABLE:
        return _unavailable(
            "scikit-learn is not available in this environment; Phase 7.4 cannot train "
            "baseline models",
            objective_used=objective_used,
        )

    # --- forecasting chronological-order precondition (audit H1) --------
    # A time-ordered holdout slices row order positionally as the time
    # axis. Phase 7.4 never infers a time column, sorts, or reorders — so
    # the rows must already be non-decreasing on one of the frame's own
    # datetime columns, otherwise the split (and every metric) would be
    # silently meaningless.
    chronological_note: str | None = None
    if split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT:
        ordered_ok, ordered_detail = _verify_chronological_order(df, problem.task_type.time_column)
        if not ordered_ok:
            return _unavailable(ordered_detail, objective_used=objective_used)
        chronological_note = ordered_detail

    category = _TASK_CATEGORY[task]
    is_supervised = category in {"regression", "classification"}

    # --- resolve features / target (from the Phase-5 / Phase-6 contracts) ---
    df_columns = [str(c) for c in df.columns]
    column_set = set(df_columns)
    target_column = problem.target.target_column if is_supervised else None

    col_type = {c.column: c.column_type for c in feature_engineering.inventory.candidates}
    eligible = [
        c for c in _eligible_features(feature_engineering) if c in column_set and c != target_column
    ]

    numeric_cols = sorted(
        c for c in eligible if col_type.get(c) in (ColumnType.NUMERIC, ColumnType.BOOLEAN)
    )
    categorical_cols = sorted(c for c in eligible if col_type.get(c) is ColumnType.CATEGORICAL)
    excluded_cols = sorted(
        c for c in eligible if col_type.get(c) in (ColumnType.DATETIME, ColumnType.UNKNOWN, None)
    )
    feature_cols = sorted(numeric_cols + categorical_cols)

    req_by_col: dict[str, set[str]] = {}
    for requirement in feature_engineering.preprocessing.requirements:
        req_by_col.setdefault(requirement.column, set()).add(requirement.description)

    notes: list[str] = [
        (
            "Phase 7.4 trained baseline estimators and computed evaluation metrics; it did "
            "NOT select a final model, rank models, tune hyperparameters, or cross-validate"
        ),
        (
            "preprocessing was executed strictly from the Phase 6.5 requirements and fitted "
            "only on the training partition (leakage-safe within this pipeline)"
        ),
        (
            "no model artifact was persisted; no fitted estimator, pipeline, prediction, or "
            "array is stored in this result"
        ),
        f"random seed: {MODEL_TRAINING_RANDOM_SEED} (fixed)",
        f"split strategy: {split.strategy.value if split.strategy else 'unspecified'}",
        f"task type: {task.value}",
    ]
    if target_column is not None:
        notes.append(f"target column '{target_column}' is excluded from the model features")
    if excluded_cols:
        notes.append(
            f"{len(excluded_cols)} datetime / unrecognised feature column(s) excluded "
            "(derivation / encoding of these is out of Phase-7.4 scope): "
            + ", ".join(excluded_cols)
        )
    if task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(
            "time-series forecasting is trained here as a baseline regression; Phase 7.4 "
            "builds no forecasting-specific model and no forecasting transformation beyond "
            "the Phase-6 lag / rolling recommendations; the task type came from Phase 5, "
            "never a datetime column"
        )
    if chronological_note is not None:
        notes.append(chronological_note)
        notes.append(
            "row order is the time axis for this time-ordered holdout; Phase 7.4 verified it "
            "against a datetime column and did not sort or reorder the rows"
        )
    temporal = feature_engineering.temporal
    if objective_used:
        notes.append("an objective was supplied and recorded; it did not change any training step")

    candidate_families: list[ModelFamily] = []
    for name in candidates.candidates:
        try:
            family = ModelFamily(name)
        except ValueError:
            continue
        if family not in candidate_families:
            candidate_families.append(family)

    if not candidate_families:
        return TrainingOutcome(
            status=ModelingStatus.COMPLETED,
            reason="no candidate model families were provided to train",
            runs=[],
            successful_runs=[],
            failed_runs=[],
            objective_used=objective_used,
            notes=notes,
        )

    # --- build the working frame (a copy; the input is never touched) ---
    is_time_ordered = split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    work = df.copy()
    work.columns = df_columns
    if is_supervised and target_column is not None:
        if is_time_ordered:
            # preserve contiguity for the positional time-ordered split: trim only
            # the leading / trailing run of missing-target rows (internal gaps are
            # blocked at Phase-5 feasibility).
            observed = work[target_column].notna().to_numpy()
            if observed.any():
                first_obs = int(observed.argmax())
                last_obs = len(observed) - 1 - int(observed[::-1].argmax())
                trimmed = len(work) - (last_obs - first_obs + 1)
                work = work.iloc[first_obs : last_obs + 1]
                if trimmed:
                    notes.append(
                        f"{trimmed} leading / trailing row(s) with a missing target were "
                        "trimmed (contiguity preserved for the time-ordered split)"
                    )
        else:
            before = len(work)
            work = work[work[target_column].notna()]
            dropped = before - len(work)
            if dropped:
                notes.append(
                    f"{dropped} row(s) with a missing target were excluded from supervised training"
                )
    work = work.reset_index(drop=True)

    # --- forecasting: execute the Phase-6 lag / rolling / calendar recommendations ---
    temporal_features_built = 0
    calendar_features_built = 0
    rows_consumed_as_history = 0
    forecasting_run = is_time_ordered and task is TaskType.TIME_SERIES_FORECASTING
    if forecasting_run and target_column is not None:
        built_temporal: list[str] = []
        built_calendar: list[str] = []
        try:
            if temporal.status is _FE_COMPLETED and temporal.recommendations:
                work, built_temporal, tf_notes = build_temporal_features(
                    work, temporal.recommendations
                )
                notes.extend(tf_notes)
            transformation_recs = feature_engineering.transformations
            if transformation_recs.status is _FE_COMPLETED and transformation_recs.recommendations:
                work, built_calendar, cal_notes = build_calendar_features(
                    work, transformation_recs.recommendations
                )
                notes.extend(cal_notes)
        except ValueError as exc:
            return _unavailable(
                "forecasting features could not be built: " + _normalise_error(str(exc)),
                objective_used=objective_used,
            )

        built_all = built_temporal + built_calendar
        if built_all:
            before = len(work)
            work = work.dropna(subset=built_all).reset_index(drop=True)
            rows_consumed_as_history = before - len(work)
            temporal_features_built = len(built_temporal)
            calendar_features_built = len(built_calendar)
            numeric_cols = sorted(set(numeric_cols) | set(built_all))
            feature_cols = sorted(set(feature_cols) | set(built_all))
            notes.append(
                f"{rows_consumed_as_history} leading / invalid-timestamp row(s) consumed as "
                "lag / rolling history"
            )
            notes.append(
                "primary evaluation is one-step-ahead: lag / rolling features use observed "
                "actuals; calendar features are stateless functions of the timestamp"
            )
            if forecast_horizon > 1:
                notes.append(
                    f"forecast_horizon = {forecast_horizon}: recursive rolling-origin "
                    "multi-step diagnostics (rmse_h2 …) are added per run; the calendar and "
                    "exogenous features are assumed known over the forecast window; the "
                    "selection metric is still the one-step rmse"
                )
        else:
            notes.append(
                "no Phase-6 lag / rolling / calendar recommendation referenced a column of "
                "the frame; trained as baseline regression on the eligible features"
            )
        if len(work) < MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS:
            return _unavailable(
                f"only {len(work)} modelable row(s) remain after consuming "
                f"{rows_consumed_as_history} row(s) as lag / rolling history; at least "
                f"{MODEL_TRAINING_FORECASTING_MIN_MODELABLE_ROWS} are required",
                objective_used=objective_used,
            )
    elif task is TaskType.TIME_SERIES_FORECASTING:
        notes.append(
            "forecasting without a time-ordered holdout; trained as baseline regression on "
            "the currently-eligible features"
        )

    # canonicalise row order for the non-temporal strategies so the split
    # (and therefore every metric) is invariant to the input row order.
    # Sort by the model columns only (features + target) — orderable dtypes
    # by construction — never by excluded datetime / unknown / object
    # columns, whose comparability varies across pandas versions (audit M9).
    if split.strategy is not DataSplitStrategy.TIME_ORDERED_HOLDOUT and len(work) > 0:
        _feature_set = set(feature_cols)
        canonical_by = sorted(c for c in work.columns if c in _feature_set or c == target_column)
        if canonical_by:
            work = work.sort_values(
                by=canonical_by, kind="stable", ignore_index=True, na_position="last"
            )
        else:
            work = work.reset_index(drop=True)

    n = len(work)
    x_all = work[feature_cols] if feature_cols else work.iloc[:, :0]
    y_all = (
        work[target_column].to_numpy() if (is_supervised and target_column is not None) else None
    )

    stratify_y = y_all if (category == "classification" and y_all is not None) else None
    train_idx, val_idx, test_idx, split_notes = _split_indices(n, split, stratify_y)
    notes.extend(split_notes)

    target_temporal_spec: dict[str, tuple[str, str, int]] = {}
    if forecasting_run and target_column is not None and temporal.status is _FE_COMPLETED:
        target_temporal_spec = {
            name: meta
            for name, meta in temporal_feature_spec(temporal.recommendations).items()
            if meta[0] == target_column and name in feature_cols
        }

    runs: list[TrainingRun] = []
    for family in candidate_families:
        runs.append(
            _run_candidate(
                family=family,
                category=category,
                x_all=x_all,
                y_all=y_all,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                req_by_col=req_by_col,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                forecast_horizon=forecast_horizon if forecasting_run else 1,
                target_temporal_spec=target_temporal_spec,
            )
        )

    if forecasting_run:
        runs = [
            run.model_copy(
                update={
                    "temporal_features_built": temporal_features_built,
                    "calendar_features_built": calendar_features_built,
                    "rows_consumed_as_history": rows_consumed_as_history,
                    "forecast_horizon": forecast_horizon,
                }
            )
            for run in runs
        ]

    successful = [r.family.value for r in runs if r.status is TrainingRunStatus.COMPLETED]
    failed = [r.family.value for r in runs if r.status is not TrainingRunStatus.COMPLETED]

    if successful:
        reason = (
            None
            if not failed
            else f"{len(failed)} of {len(runs)} candidate(s) did not train: {', '.join(failed)}"
        )
    else:
        reason = f"all {len(runs)} candidate model family(ies) failed to train"

    return TrainingOutcome(
        status=ModelingStatus.COMPLETED,
        reason=reason,
        runs=runs,
        successful_runs=successful,
        failed_runs=failed,
        objective_used=objective_used,
        notes=notes,
    )


def _run_candidate(
    *,
    family: ModelFamily,
    category: str,
    x_all: pd.DataFrame,
    y_all: np.ndarray | None,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    req_by_col: dict[str, set[str]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    forecast_horizon: int = 1,
    target_temporal_spec: dict[str, tuple[str, str, int]] | None = None,
) -> TrainingRun:
    built = _build_estimator(family, category)
    if built is None:
        return TrainingRun(
            family=family,
            estimator_name="(none)",
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=(
                f"no dependency-light baseline estimator is defined for the '{family.value}' "
                f"family on a {category} task"
            ),
        )
    estimator_name, estimator = built

    if not feature_cols:
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            test_rows=int(test_idx.size),
            reason="no usable numeric / categorical feature columns are available for training",
        )
    if (
        train_idx.size < MODEL_TRAINING_MIN_TRAIN_ROWS
        or test_idx.size < MODEL_TRAINING_MIN_TEST_ROWS
    ):
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=(
                f"insufficient rows after the split (train = {train_idx.size}, "
                f"test = {test_idx.size})"
            ),
        )

    preprocessor, preproc_error = _build_preprocessor(
        numeric_cols,
        categorical_cols,
        req_by_col,
        robust_scaling=_is_outlier_heavy(x_all.iloc[train_idx], numeric_cols),
    )
    if preproc_error is not None:
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.UNAVAILABLE,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=preproc_error,
        )

    try:
        from sklearn.pipeline import Pipeline

        x_train = x_all.iloc[train_idx]
        x_test = x_all.iloc[test_idx]

        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

        if category == "clustering":
            pipeline.fit(x_train)
            labels = np.asarray(pipeline.predict(x_test))
            transformed = pipeline.named_steps["preprocess"].transform(x_test)
            metrics = _clustering_metrics(np.asarray(transformed, dtype=float), labels)
            notes = (
                []
                if metrics
                else [
                    "fewer than 2 distinct clusters were assigned; no clustering metric is defined"
                ]
            )
        else:
            assert y_all is not None
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]
            pipeline.fit(x_train, y_train)
            y_pred = np.asarray(pipeline.predict(x_test))
            if category == "regression":
                metrics = _regression_metrics(y_test.astype(float), y_pred.astype(float))
                notes = []
                if (
                    forecast_horizon > 1
                    and target_temporal_spec
                    and int(test_idx.size) > forecast_horizon
                ):
                    metrics = {
                        **metrics,
                        **_recursive_horizon_metrics(
                            pipeline,
                            x_all,
                            y_all,
                            feature_cols,
                            target_temporal_spec,
                            test_start=int(test_idx[0]),
                            n_rows=int(x_all.shape[0]),
                            horizon=forecast_horizon,
                        ),
                    }
                    notes.append(
                        f"recursive rolling-origin multi-step diagnostics (rmse_h2 … "
                        f"rmse_h{forecast_horizon}); the selection metric is still the "
                        "one-step rmse"
                    )
            else:
                proba = None
                model = pipeline.named_steps["model"]
                if hasattr(model, "predict_proba"):
                    try:
                        proba_full = np.asarray(pipeline.predict_proba(x_test))
                        if proba_full.ndim == 2 and proba_full.shape[1] == 2:
                            proba = proba_full[:, 1]
                    except (ValueError, AttributeError):
                        proba = None
                n_classes = len(np.unique(y_train))
                metrics = _classification_metrics(y_test, y_pred, proba, n_classes)
                notes = []
    except Exception as exc:  # noqa: BLE001 - deterministic, normalised failure record
        return TrainingRun(
            family=family,
            estimator_name=estimator_name,
            status=TrainingRunStatus.FAILED,
            train_rows=int(train_idx.size),
            validation_rows=int(val_idx.size),
            test_rows=int(test_idx.size),
            reason=f"{type(exc).__name__}: {_normalise_error(str(exc))}",
        )

    return TrainingRun(
        family=family,
        estimator_name=estimator_name,
        status=TrainingRunStatus.COMPLETED,
        train_rows=int(train_idx.size),
        validation_rows=int(val_idx.size),
        test_rows=int(test_idx.size),
        metrics=metrics,
        reason=None,
        notes=notes,
    )


@dataclass(frozen=True)
class _ResolvedContext:
    """The category/target/feature resolution shared by every Phase-7.6/7.7
    function that fits an estimator on the *full* dataset (as opposed to
    :func:`train_and_evaluate_models`'s own train/test-split-scoped copy
    of this same resolution)."""

    category: str
    target_column: str | None
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]


def _resolve_task_and_features(
    df: pd.DataFrame, problem: ProblemSpec, feature_engineering: FeatureEngineeringSpec
) -> _ResolvedContext:
    """Resolve the task category, target column, and feature columns for a
    full-dataset fit. Raises ``ValueError`` when that is not possible —
    the same preconditions :func:`train_and_evaluate_models` checks,
    surfaced as an exception since there is no ``TrainingRun`` to carry an
    ``unavailable`` status here.
    """
    if not _SKLEARN_AVAILABLE:
        raise ValueError("scikit-learn is not available in this environment")

    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED or task_inference.task_type is None:
        raise ValueError("task-type inference is not completed")
    task = task_inference.task_type
    if task in _UNSUPPORTED_TASKS:
        raise ValueError(f"model training does not support task type '{task.value}'")
    category = _TASK_CATEGORY[task]
    is_supervised = category in {"regression", "classification"}

    column_set = {str(c) for c in df.columns}
    target_column = problem.target.target_column if is_supervised else None

    col_type = {c.column: c.column_type for c in feature_engineering.inventory.candidates}
    eligible = [
        c for c in _eligible_features(feature_engineering) if c in column_set and c != target_column
    ]
    numeric_cols = sorted(
        c for c in eligible if col_type.get(c) in (ColumnType.NUMERIC, ColumnType.BOOLEAN)
    )
    categorical_cols = sorted(c for c in eligible if col_type.get(c) is ColumnType.CATEGORICAL)
    feature_cols = sorted(numeric_cols + categorical_cols)
    if not feature_cols:
        raise ValueError("no usable numeric / categorical feature columns are available")

    return _ResolvedContext(
        category=category,
        target_column=target_column,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


def _build_preprocessor_for(
    feature_engineering: FeatureEngineeringSpec,
    ctx: _ResolvedContext,
    *,
    x_train: pd.DataFrame | None = None,
) -> Any:
    """`_build_preprocessor` given an already-resolved `_ResolvedContext`.
    `x_train`, when given, decides `robust_scaling` via `_is_outlier_heavy`.
    Raises ``ValueError`` with the preprocessor's own reason on failure.
    """
    req_by_col: dict[str, set[str]] = {}
    for requirement in feature_engineering.preprocessing.requirements:
        req_by_col.setdefault(requirement.column, set()).add(requirement.description)
    robust_scaling = _is_outlier_heavy(x_train, ctx.numeric_cols) if x_train is not None else False
    preprocessor, preproc_error = _build_preprocessor(
        ctx.numeric_cols,
        ctx.categorical_cols,
        req_by_col,
        robust_scaling=robust_scaling,
    )
    if preproc_error is not None:
        raise ValueError(preproc_error)
    return preprocessor


@dataclass(frozen=True)
class FittedPipeline:
    """A fitted, ready-to-persist scikit-learn pipeline plus the metadata
    :func:`data_engine.modeling.persistence.save_model` needs to run it
    again later on new, unseen rows.

    Holds a live scikit-learn ``Pipeline`` object, so — unlike every other
    result in :mod:`data_engine.modeling` — it is **not** JSON-primitive
    and never crosses the Pydantic / HTTP boundary directly. It exists
    only to hand off to the persistence layer.
    """

    pipeline: Any
    estimator_name: str
    family: ModelFamily
    category: str
    target_column: str | None
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    notes: list[str] = field(default_factory=list)


def fit_final_pipeline(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    family: ModelFamily,
) -> FittedPipeline:
    """Refit the chosen candidate family's baseline pipeline on *all*
    available rows, for persistence and later inference.

    This is the **only** function in Phase 7 allowed to return a live,
    fitted estimator — every evaluation path (:func:`train_and_evaluate_models`)
    deliberately keeps its fitted pipelines train/test-split-scoped and
    JSON-primitive-only. Standard practice once a family has been chosen
    from held-out evaluation: refit on the full dataset for the artifact
    that actually gets deployed. Reuses the exact same estimator mapping
    (:func:`_build_estimator`) and preprocessing construction
    (:func:`_build_preprocessor`) as evaluation, so the fitted pipeline is
    built identically to the one whose metrics were reported — nothing
    about the model definition is reinvented here.

    Raises ``ValueError`` if the task type / feature set makes training
    impossible (mirrors the same preconditions :func:`train_and_evaluate_models`
    checks, surfaced as an exception here since there is no ``TrainingRun``
    to carry an ``unavailable`` status).
    """
    ctx = _resolve_task_and_features(df, problem, feature_engineering)

    built = _build_estimator(family, ctx.category)
    if built is None:
        raise ValueError(
            f"no dependency-light baseline estimator is defined for the '{family.value}' "
            f"family on a {ctx.category} task"
        )
    estimator_name, estimator = built
    x_all = df[ctx.feature_cols]
    preprocessor = _build_preprocessor_for(feature_engineering, ctx, x_train=x_all)

    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    notes = [
        f"refit on all {len(df)} available rows after family '{family.value}' was selected "
        "from held-out evaluation; see the ModelingSpec that produced this selection for "
        "the metrics that justified it",
        f"random seed: {MODEL_TRAINING_RANDOM_SEED} (fixed)",
    ]
    if ctx.category == "clustering":
        pipeline.fit(x_all)
    else:
        y_all = df[ctx.target_column].to_numpy()
        pipeline.fit(x_all, y_all)

    return FittedPipeline(
        pipeline=pipeline,
        estimator_name=estimator_name,
        family=family,
        category=ctx.category,
        target_column=ctx.target_column,
        feature_cols=ctx.feature_cols,
        numeric_cols=ctx.numeric_cols,
        categorical_cols=ctx.categorical_cols,
        notes=notes,
    )


@dataclass(frozen=True)
class _CandidateSpec:
    family: ModelFamily
    estimator_name: str
    hyperparameters: dict[str, Any]
    build: Any  # Callable[[], Any] — a zero-arg estimator factory


def _sweep(
    specs: list[_CandidateSpec],
    family: ModelFamily,
    name: str,
    param_sets: list[dict[str, Any]],
    factory: Any,
) -> None:
    """Append one `_CandidateSpec` per entry in `param_sets`, each built by
    `factory(**params)`. The closure captures `params` by default-argument
    binding (`p=params`) so every candidate gets its own exact hyperparameters
    rather than all sharing the loop variable's final value."""
    for params in param_sets:
        specs.append(_CandidateSpec(family, name, dict(params), lambda p=params: factory(**p)))


def _expanded_catalog(category: str) -> list[_CandidateSpec]:
    """Phase 7.7's expanded candidate catalog for one task category.

    A **fixed, documented grid** — never a randomized search — so the
    same data and the same fixed random seed always produce the same
    ranked result. Each entry is a concrete scikit-learn estimator plus
    the exact hyperparameters it was given; nothing here is tuned
    adaptively from the data. 100+ candidates per category — every
    scikit-learn estimator family with a dependency-light, deterministic
    `fit`, swept across a documented hyperparameter grid.

    Every new estimator type here is mapped onto the existing, fixed
    6-value `ModelFamily` enum (declarative since Phase 7.1) rather than
    growing that enum — e.g. SVM under `DISTANCE_BASED` (margin/kernel
    methods, the closest existing bucket), SGD/Huber/Perceptron/
    PassiveAggressive/RidgeClassifier/BayesianRidge under `LINEAR`,
    AdaBoost/Bagging/HistGradientBoosting under `ENSEMBLE`, and
    discriminant-analysis/BernoulliNB under `PROBABILISTIC` alongside
    GaussianNB.
    """
    seed = MODEL_TRAINING_RANDOM_SEED
    specs: list[_CandidateSpec] = []

    _TREE_DEPTHS: list[dict[str, Any]] = [
        {"max_depth": d} for d in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, None)
    ]
    _FOREST_GRID: list[dict[str, Any]] = [
        {"n_estimators": n, "max_depth": d}
        for n, d in (
            (50, 4),
            (50, 8),
            (50, None),
            (100, 6),
            (100, 10),
            (100, None),
            (150, 8),
            (200, 10),
            (200, None),
        )
    ]
    _BOOST_GRID: list[dict[str, Any]] = [
        {"n_estimators": n, "learning_rate": lr}
        for n, lr in (
            (50, 0.1),
            (100, 0.1),
            (100, 0.05),
            (150, 0.05),
            (200, 0.05),
            (200, 0.03),
            (300, 0.02),
        )
    ]
    _KNN_GRID: list[dict[str, Any]] = [
        {"n_neighbors": k} for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30)
    ]
    _MLP_GRID: list[dict[str, Any]] = [
        {"hidden_layer_sizes": list(h)}
        for h in ((16,), (32,), (64,), (128,), (32, 16), (64, 32), (128, 64), (64, 64, 32))
    ]

    if category == "regression":
        from sklearn.ensemble import (
            AdaBoostRegressor,
            BaggingRegressor,
            ExtraTreesRegressor,
            GradientBoostingRegressor,
            HistGradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.linear_model import (
            BayesianRidge,
            ElasticNet,
            HuberRegressor,
            Lasso,
            LinearRegression,
            PassiveAggressiveRegressor,
            Ridge,
            SGDRegressor,
        )
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor

        specs.append(_CandidateSpec(ModelFamily.LINEAR, "LinearRegression", {}, LinearRegression))
        specs.append(
            _CandidateSpec(ModelFamily.LINEAR, "BayesianRidge", {}, lambda: BayesianRidge())
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "Ridge",
            [{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
            lambda alpha: Ridge(alpha=alpha, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "Lasso",
            [{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0, 10.0)],
            lambda alpha: Lasso(alpha=alpha, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "ElasticNet",
            [{"alpha": a, "l1_ratio": r} for a in (0.1, 1.0, 10.0) for r in (0.2, 0.5, 0.8)],
            lambda alpha, l1_ratio: ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "SGDRegressor",
            [
                {"alpha": a, "penalty": p}
                for a, p in ((0.0001, "l2"), (0.001, "l2"), (0.0001, "l1"), (0.001, "elasticnet"))
            ],
            lambda alpha, penalty: SGDRegressor(alpha=alpha, penalty=penalty, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "HuberRegressor",
            [{"epsilon": e} for e in (1.1, 1.35, 1.5, 2.0)],
            lambda epsilon: HuberRegressor(epsilon=epsilon),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "PassiveAggressiveRegressor",
            [{"C": c} for c in (0.1, 1.0, 10.0)],
            lambda C: PassiveAggressiveRegressor(C=C, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.TREE_BASED,
            "DecisionTreeRegressor",
            _TREE_DEPTHS,
            lambda max_depth: DecisionTreeRegressor(max_depth=max_depth, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "RandomForestRegressor",
            _FOREST_GRID,
            lambda n_estimators, max_depth: RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=1
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "ExtraTreesRegressor",
            [{"n_estimators": n} for n in (50, 100, 150, 200)],
            lambda n_estimators: ExtraTreesRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=1
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "GradientBoostingRegressor",
            _BOOST_GRID,
            lambda n_estimators, learning_rate: GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "AdaBoostRegressor",
            [{"n_estimators": n} for n in (50, 100, 200)],
            lambda n_estimators: AdaBoostRegressor(n_estimators=n_estimators, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "BaggingRegressor",
            [{"n_estimators": n} for n in (10, 50, 100)],
            lambda n_estimators: BaggingRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=1
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "HistGradientBoostingRegressor",
            [
                {"max_iter": m, "learning_rate": lr}
                for m, lr in ((100, 0.1), (200, 0.1), (100, 0.05), (300, 0.05))
            ],
            lambda max_iter, learning_rate: HistGradientBoostingRegressor(
                max_iter=max_iter, learning_rate=learning_rate, random_state=seed
            ),
        )
        _sweep(
            specs,
            ModelFamily.DISTANCE_BASED,
            "KNeighborsRegressor",
            _KNN_GRID,
            lambda n_neighbors: KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=1),
        )
        _sweep(
            specs,
            ModelFamily.DISTANCE_BASED,
            "SVR",
            [
                {"kernel": k, "C": c}
                for k, c in (
                    ("rbf", 0.1),
                    ("rbf", 1.0),
                    ("rbf", 10.0),
                    ("linear", 1.0),
                    ("linear", 10.0),
                    ("poly", 1.0),
                )
            ],
            lambda kernel, C: SVR(kernel=kernel, C=C),
        )
        _sweep(
            specs,
            ModelFamily.NEURAL,
            "MLPRegressor",
            _MLP_GRID,
            lambda hidden_layer_sizes: MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=MODEL_TRAINING_MLP_MAX_ITER,
                random_state=seed,
            ),
        )

    elif category == "classification":
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
            QuadraticDiscriminantAnalysis,
        )
        from sklearn.ensemble import (
            AdaBoostClassifier,
            BaggingClassifier,
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.linear_model import (
            LogisticRegression,
            PassiveAggressiveClassifier,
            Perceptron,
            RidgeClassifier,
            SGDClassifier,
        )
        from sklearn.naive_bayes import BernoulliNB, GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        _sweep(
            specs,
            ModelFamily.LINEAR,
            "LogisticRegression",
            [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
            lambda C: LogisticRegression(
                C=C,
                max_iter=MODEL_TRAINING_LOGREG_MAX_ITER,
                random_state=seed,
                class_weight="balanced",
            ),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "LogisticRegression",
            [{"C": c, "penalty": "l1", "solver": "liblinear"} for c in (0.01, 0.1, 1.0, 10.0)],
            lambda C, penalty, solver: LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=MODEL_TRAINING_LOGREG_MAX_ITER,
                random_state=seed,
                class_weight="balanced",
            ),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "RidgeClassifier",
            [{"alpha": a} for a in (0.1, 1.0, 10.0)],
            lambda alpha: RidgeClassifier(alpha=alpha, random_state=seed, class_weight="balanced"),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "SGDClassifier",
            [
                {"alpha": a, "loss": loss}
                for a, loss in (
                    (0.0001, "log_loss"),
                    (0.001, "log_loss"),
                    (0.0001, "hinge"),
                    (0.001, "modified_huber"),
                )
            ],
            lambda alpha, loss: SGDClassifier(
                alpha=alpha, loss=loss, random_state=seed, class_weight="balanced"
            ),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "PassiveAggressiveClassifier",
            [{"C": c} for c in (0.1, 1.0, 10.0)],
            lambda C: PassiveAggressiveClassifier(C=C, random_state=seed, class_weight="balanced"),
        )
        _sweep(
            specs,
            ModelFamily.LINEAR,
            "Perceptron",
            [{"alpha": a} for a in (0.0001, 0.001, 0.01)],
            lambda alpha: Perceptron(alpha=alpha, random_state=seed, class_weight="balanced"),
        )
        _sweep(
            specs,
            ModelFamily.TREE_BASED,
            "DecisionTreeClassifier",
            _TREE_DEPTHS,
            lambda max_depth: DecisionTreeClassifier(
                max_depth=max_depth, random_state=seed, class_weight="balanced"
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "RandomForestClassifier",
            _FOREST_GRID,
            lambda n_estimators, max_depth: RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=1,
                class_weight="balanced",
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "ExtraTreesClassifier",
            [{"n_estimators": n} for n in (50, 100, 150, 200)],
            lambda n_estimators: ExtraTreesClassifier(
                n_estimators=n_estimators, random_state=seed, n_jobs=1, class_weight="balanced"
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "GradientBoostingClassifier",
            _BOOST_GRID,
            lambda n_estimators, learning_rate: GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "AdaBoostClassifier",
            [{"n_estimators": n} for n in (50, 100, 200)],
            lambda n_estimators: AdaBoostClassifier(n_estimators=n_estimators, random_state=seed),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "BaggingClassifier",
            [{"n_estimators": n} for n in (10, 50, 100)],
            lambda n_estimators: BaggingClassifier(
                n_estimators=n_estimators, random_state=seed, n_jobs=1
            ),
        )
        _sweep(
            specs,
            ModelFamily.ENSEMBLE,
            "HistGradientBoostingClassifier",
            [
                {"max_iter": m, "learning_rate": lr}
                for m, lr in ((100, 0.1), (200, 0.1), (100, 0.05), (300, 0.05))
            ],
            lambda max_iter, learning_rate: HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                random_state=seed,
                class_weight="balanced",
            ),
        )
        _sweep(
            specs,
            ModelFamily.PROBABILISTIC,
            "GaussianNB",
            [{"var_smoothing": v} for v in (1e-9, 1e-8, 1e-7, 1e-6)],
            lambda var_smoothing: GaussianNB(var_smoothing=var_smoothing),
        )
        _sweep(
            specs,
            ModelFamily.PROBABILISTIC,
            "BernoulliNB",
            [{"alpha": a} for a in (0.1, 0.5, 1.0)],
            lambda alpha: BernoulliNB(alpha=alpha),
        )
        specs.append(
            _CandidateSpec(
                ModelFamily.PROBABILISTIC,
                "LinearDiscriminantAnalysis",
                {},
                lambda: LinearDiscriminantAnalysis(),
            )
        )
        specs.append(
            _CandidateSpec(
                ModelFamily.PROBABILISTIC,
                "QuadraticDiscriminantAnalysis",
                {},
                lambda: QuadraticDiscriminantAnalysis(),
            )
        )
        _sweep(
            specs,
            ModelFamily.DISTANCE_BASED,
            "KNeighborsClassifier",
            _KNN_GRID,
            lambda n_neighbors: KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1),
        )
        _sweep(
            specs,
            ModelFamily.DISTANCE_BASED,
            "SVC",
            [
                {"kernel": k, "C": c}
                for k, c in (
                    ("rbf", 0.1),
                    ("rbf", 1.0),
                    ("rbf", 10.0),
                    ("linear", 1.0),
                    ("linear", 10.0),
                    ("poly", 1.0),
                )
            ],
            lambda kernel, C: SVC(kernel=kernel, C=C, random_state=seed, class_weight="balanced"),
        )
        _sweep(
            specs,
            ModelFamily.NEURAL,
            "MLPClassifier",
            _MLP_GRID,
            lambda hidden_layer_sizes: MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=MODEL_TRAINING_MLP_MAX_ITER,
                random_state=seed,
            ),
        )

    return specs


_ENSEMBLE_TOP_K = 5
_CV_FOLDS = 5
_CV_SCORING: dict[str, str] = {
    "rmse": "neg_root_mean_squared_error",
    "f1": "f1_macro",
}


class _PreparationError(Exception):
    """Raised by `_prepare_supervised_run` for any precondition failure.
    Both `run_expanded_search` and `tune_best_candidate` catch this and
    translate it into their own `status=unavailable` result type."""

    def __init__(self, reason: str, task_type: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.task_type = task_type


@dataclass(frozen=True)
class _PreparedSupervisedRun:
    """Everything `run_expanded_search` and `tune_best_candidate` both need
    before their own per-candidate logic begins — task validation, the
    missing-target-aware train/test split, the outlier-aware preprocessor,
    and the skewed-target log-transform decision. Factored out so the two
    functions can't silently drift apart on how a dataset is prepared."""

    task: TaskType
    category: str
    ctx: _ResolvedContext
    catalog: list[_CandidateSpec]
    preprocessor: Any
    x_all: pd.DataFrame
    y_all: np.ndarray
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    selection_metric: str | None
    direction: str | None
    target_log_transform: bool
    dropped_missing_target: int
    is_time_ordered: bool


def _prepare_supervised_run(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    *,
    require_catalog: bool = True,
) -> _PreparedSupervisedRun:
    task_inference = problem.task_type
    if task_inference.status is not _PU_COMPLETED or task_inference.task_type is None:
        raise _PreparationError("task-type inference is not completed")
    task = task_inference.task_type
    if task in _UNSUPPORTED_TASKS:
        raise _PreparationError(
            f"model training does not support task type '{task.value}'", task.value
        )
    category = _TASK_CATEGORY[task]
    if category not in ("regression", "classification"):
        raise _PreparationError(
            "the expanded search covers regression and classification only", task.value
        )
    if readiness.status is not ModelingStatus.COMPLETED or readiness.ready is False:
        first = (
            readiness.blocking_issues[0]
            if readiness.blocking_issues
            else (readiness.reason or "the data is not ready for modeling")
        )
        raise _PreparationError(
            f"training is blocked by model-readiness issues: {first}", task.value
        )
    if split.status is not ModelingStatus.COMPLETED:
        raise _PreparationError("the data-split plan is not completed", task.value)
    if not _SKLEARN_AVAILABLE:
        raise _PreparationError("scikit-learn is not available in this environment", task.value)

    try:
        ctx = _resolve_task_and_features(df, problem, feature_engineering)
    except ValueError as exc:
        raise _PreparationError(str(exc), task.value) from exc

    catalog = _expanded_catalog(category)
    if require_catalog and not catalog:
        raise _PreparationError(
            f"no expanded-search catalog is defined for category '{category}'", task.value
        )

    assert ctx.target_column is not None
    is_time_ordered = split.strategy is DataSplitStrategy.TIME_ORDERED_HOLDOUT
    work = df
    dropped_missing_target = 0
    if is_time_ordered:
        observed = df[ctx.target_column].notna().to_numpy()
        if observed.any():
            first_obs = int(observed.argmax())
            last_obs = len(observed) - 1 - int(observed[::-1].argmax())
            dropped_missing_target = len(df) - (last_obs - first_obs + 1)
            work = df.iloc[first_obs : last_obs + 1].reset_index(drop=True)
    else:
        before = len(df)
        work = df[df[ctx.target_column].notna()].reset_index(drop=True)
        dropped_missing_target = before - len(work)

    if work.empty:
        raise _PreparationError(
            "every row has a missing target value; nothing is left to train on", task.value
        )

    x_all = work[ctx.feature_cols]
    y_all = work[ctx.target_column].to_numpy()
    train_idx, _val_idx, test_idx, _split_notes = _split_indices(len(work), split, y_all)

    selection_metric, direction = _TASK_SELECTION_METRIC.get(task, (None, None))

    x_train, x_test = x_all.iloc[train_idx], x_all.iloc[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    try:
        preprocessor = _build_preprocessor_for(feature_engineering, ctx, x_train=x_train)
    except ValueError as exc:
        raise _PreparationError(str(exc), task.value) from exc

    target_log_transform = (
        category == "regression"
        and _skewness(y_train) > 1.0
        and float(np.asarray(y_all, dtype=float).min()) >= 0.0
    )

    return _PreparedSupervisedRun(
        task=task,
        category=category,
        ctx=ctx,
        catalog=catalog,
        preprocessor=preprocessor,
        x_all=x_all,
        y_all=y_all,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        selection_metric=selection_metric,
        direction=direction,
        target_log_transform=target_log_transform,
        dropped_missing_target=dropped_missing_target,
        is_time_ordered=is_time_ordered,
    )


def run_expanded_search(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    *,
    objective: str | None = None,
    use_cross_validation: bool = False,
) -> ExpandedSearchResult:
    """Fit and evaluate every candidate in Phase 7.7's expanded catalog —
    100+ concrete (estimator, hyperparameter) combinations, not one
    baseline per family — and return every result, ranked.

    This is the **hyperparameter-tuning entry point**: every candidate is
    a fixed, documented (estimator, hyperparameter) pair (see
    :func:`_expanded_catalog`), fit on the exact same train/test split
    :func:`train_and_evaluate_models` would use (same `DataSplitPlan`,
    same fixed random seed) and ranked by the exact same per-task
    selection metric :func:`data_engine.modeling.selection.select_model`
    uses. Unlike Phase 7.4/7.5 (one estimator per family, one winner
    surfaced), this returns **every** candidate's result so a caller can
    compare the full field, not just the one DataPilot would have picked.

    ``use_cross_validation`` (default ``False``, since it multiplies
    runtime by roughly ``_CV_FOLDS``): when ``True``, every candidate
    *also* gets a fixed ``_CV_FOLDS``-fold cross-validation score for the
    selection metric (``KFold`` for regression, ``StratifiedKFold`` for
    classification; shuffled with the one fixed random seed this module
    always uses) over the *full* dataset — a single train/test split can
    be an unreliable, high-variance estimate of how a model generalizes;
    averaging over several folds is the standard fix. When enabled,
    candidates are ranked by the cross-validated mean instead of the
    single-split score, since it is the more reliable number. A
    candidate for which cross-validation itself fails (e.g. a class with
    too few members to stratify) keeps its single-split result and
    simply has no ``cv_*`` metrics — it is never dropped or marked
    failed over this.

    Supervised tasks only (regression / classification) — clustering and
    unsupported task types return ``status = unavailable``.
    """
    try:
        prep = _prepare_supervised_run(df, problem, feature_engineering, readiness, split)
    except _PreparationError as exc:
        return ExpandedSearchResult(
            status=ModelingStatus.UNAVAILABLE, reason=exc.reason, task_type=exc.task_type
        )

    from sklearn.pipeline import Pipeline

    task = prep.task
    category = prep.category
    catalog = prep.catalog
    preprocessor = prep.preprocessor
    x_all, y_all = prep.x_all, prep.y_all
    x_train, x_test = prep.x_train, prep.x_test
    y_train, y_test = prep.y_train, prep.y_test
    selection_metric, direction = prep.selection_metric, prep.direction
    target_log_transform = prep.target_log_transform
    dropped_missing_target = prep.dropped_missing_target
    is_time_ordered = prep.is_time_ordered

    results: list[tuple[ExpandedCandidateResult, float | None]] = []

    cv_splitter = None
    if use_cross_validation and selection_metric in _CV_SCORING:
        from sklearn.model_selection import KFold, StratifiedKFold

        cv_splitter = (
            KFold(n_splits=_CV_FOLDS, shuffle=True, random_state=MODEL_TRAINING_RANDOM_SEED)
            if category == "regression"
            else StratifiedKFold(
                n_splits=_CV_FOLDS, shuffle=True, random_state=MODEL_TRAINING_RANDOM_SEED
            )
        )

    def _wrap_target(estimator: Any) -> Any:
        if not target_log_transform:
            return estimator
        from sklearn.compose import TransformedTargetRegressor

        return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)

    def _score_candidate(
        estimator: Any, metrics: dict[str, float]
    ) -> tuple[float | None, dict[str, float]]:
        """Runs CV (if enabled) for an already-fitted-shape `estimator` and
        returns the score to rank by, mutating `metrics` in place with the
        `cv_*` entries when CV succeeds."""
        cv_mean: float | None = None
        if cv_splitter is not None and selection_metric is not None:
            try:
                from sklearn.model_selection import cross_val_score

                scoring = _CV_SCORING[selection_metric]
                cv_scores = cross_val_score(
                    estimator, x_all, y_all, cv=cv_splitter, scoring=scoring
                )
                cv_mean = float(
                    -cv_scores.mean() if scoring.startswith("neg_") else cv_scores.mean()
                )
                metrics[f"cv_{selection_metric}_mean"] = _round(cv_mean)
                metrics[f"cv_{selection_metric}_std"] = _round(float(cv_scores.std()))
            except Exception:  # noqa: BLE001 - CV is a best-effort addition, never fatal
                cv_mean = None
        score = (
            cv_mean
            if cv_mean is not None
            else (metrics.get(selection_metric) if selection_metric else None)
        )
        return score, metrics

    for spec in catalog:
        candidate_start = _perf_counter()
        try:
            pipeline = Pipeline([("preprocess", preprocessor), ("model", spec.build())])
            estimator = _wrap_target(pipeline)
            estimator.fit(x_train, y_train)
            y_pred = np.asarray(estimator.predict(x_test))
            if category == "regression":
                metrics = _regression_metrics(y_test.astype(float), y_pred.astype(float))
            else:
                proba = None
                model = pipeline.named_steps["model"]
                if hasattr(model, "predict_proba"):
                    try:
                        proba_full = np.asarray(pipeline.predict_proba(x_test))
                        if proba_full.ndim == 2 and proba_full.shape[1] == 2:
                            proba = proba_full[:, 1]
                    except (ValueError, AttributeError):
                        proba = None
                metrics = _classification_metrics(
                    y_test, y_pred, proba, n_classes=len(np.unique(y_train))
                )

            score, metrics = _score_candidate(estimator, metrics)
            results.append(
                (
                    ExpandedCandidateResult(
                        rank=0,
                        family=spec.family,
                        estimator_name=spec.estimator_name,
                        hyperparameters=spec.hyperparameters,
                        status=TrainingRunStatus.COMPLETED,
                        metrics=metrics,
                        fit_seconds=round(_perf_counter() - candidate_start, 4),
                    ),
                    score,
                )
            )
        except Exception as exc:  # noqa: BLE001 - deterministic, normalised failure record
            results.append(
                (
                    ExpandedCandidateResult(
                        rank=0,
                        family=spec.family,
                        estimator_name=spec.estimator_name,
                        hyperparameters=spec.hyperparameters,
                        status=TrainingRunStatus.FAILED,
                        reason=f"{type(exc).__name__}: {_normalise_error(str(exc))}",
                        fit_seconds=round(_perf_counter() - candidate_start, 4),
                    ),
                    None,
                )
            )

    # a voting ensemble over the top-performing candidates, fit once more and
    # scored exactly like any other candidate (ranked alongside everything
    # else, dropped if it doesn't actually score better) — a single "winner"
    # rarely beats a small ensemble of genuinely different models, so this is
    # offered as one more candidate rather than assumed to be the answer.
    ensemble_members = sorted(
        (
            (spec, score)
            for spec, (result, score) in zip(catalog, results, strict=True)
            if result.status is TrainingRunStatus.COMPLETED and score is not None
        ),
        key=lambda item: -item[1] if direction == "maximize" else item[1],
    )[:_ENSEMBLE_TOP_K]
    if len(ensemble_members) >= 2:
        ensemble_start = _perf_counter()
        member_names = [spec.estimator_name for spec, _score in ensemble_members]
        try:
            from sklearn.base import clone
            from sklearn.ensemble import VotingClassifier, VotingRegressor

            members = [
                (f"m{i}", Pipeline([("preprocess", clone(preprocessor)), ("model", spec.build())]))
                for i, (spec, _score) in enumerate(ensemble_members)
            ]
            voting_estimator: Any = (
                VotingRegressor(estimators=members)
                if category == "regression"
                else VotingClassifier(estimators=members, voting="hard")
            )
            estimator = _wrap_target(voting_estimator)
            estimator.fit(x_train, y_train)
            y_pred = np.asarray(estimator.predict(x_test))
            if category == "regression":
                metrics = _regression_metrics(y_test.astype(float), y_pred.astype(float))
            else:
                metrics = _classification_metrics(
                    y_test, y_pred, None, n_classes=len(np.unique(y_train))
                )
            score, metrics = _score_candidate(estimator, metrics)
            results.append(
                (
                    ExpandedCandidateResult(
                        rank=0,
                        family=ModelFamily.ENSEMBLE,
                        estimator_name=f"VotingEnsemble(top-{len(ensemble_members)})",
                        hyperparameters={"members": member_names},
                        status=TrainingRunStatus.COMPLETED,
                        metrics=metrics,
                        fit_seconds=round(_perf_counter() - ensemble_start, 4),
                    ),
                    score,
                )
            )
        except Exception as exc:  # noqa: BLE001 - deterministic, normalised failure record
            results.append(
                (
                    ExpandedCandidateResult(
                        rank=0,
                        family=ModelFamily.ENSEMBLE,
                        estimator_name=f"VotingEnsemble(top-{len(ensemble_members)})",
                        hyperparameters={"members": member_names},
                        status=TrainingRunStatus.FAILED,
                        reason=f"{type(exc).__name__}: {_normalise_error(str(exc))}",
                        fit_seconds=round(_perf_counter() - ensemble_start, 4),
                    ),
                    None,
                )
            )

    def _sort_key(item: tuple[ExpandedCandidateResult, float | None]) -> tuple[int, float]:
        result, score = item
        if score is None:
            return (1, 0.0)
        signed = -score if direction == "maximize" else score
        return (0, signed)

    results.sort(key=_sort_key)
    ranked = [
        result.model_copy(update={"rank": i + 1}) for i, (result, _score) in enumerate(results)
    ]

    cv_active = cv_splitter is not None
    return ExpandedSearchResult(
        status=ModelingStatus.COMPLETED,
        task_type=task.value,
        selection_metric=selection_metric,
        candidate_count=len(ranked),
        total_fit_seconds=round(sum(r.fit_seconds for r, _ in results), 4),
        cross_validation_enabled=cv_active,
        candidates=ranked,
        notes=[
            f"{len(catalog)} (estimator, hyperparameter) candidates from the fixed Phase 7.7 "
            "catalog, each fit once on a single train/test split — see each candidate's own "
            "metrics for its test-partition performance"
            + (
                f"; additionally scored with {_CV_FOLDS}-fold cross-validation "
                f"(cv_{selection_metric}_mean / cv_{selection_metric}_std on each candidate) "
                "and ranked by that more reliable estimate instead"
                if cv_active
                else " (no cross-validation this run)"
            ),
            (
                f"ranked by {'cross-validated ' if cv_active else ''}'{selection_metric}' "
                f"({direction})"
                if selection_metric
                else "unranked"
            ),
            f"random seed: {MODEL_TRAINING_RANDOM_SEED} (fixed)",
            "fit_seconds on every candidate and total_fit_seconds here are real wall-clock "
            "timings, not estimates — these are classical scikit-learn estimators"
            + (
                f", each cross-validated over {_CV_FOLDS} folds this run, which is why "
                "fit_seconds is noticeably larger than a single-split-only run would show"
                if cv_active
                else " on a single train/test split (no cross-validation, no deep learning), "
                "which is why 100+ candidates typically complete in single-digit seconds on a "
                "dataset of a few hundred to a few thousand rows; a slower or much larger "
                "dataset will show correspondingly larger fit_seconds values here"
            ),
            "no model artifact was persisted here; use "
            "data_engine.modeling.persistence.save_model with fit_final_pipeline (or an "
            "expanded-search-specific fit) to persist a chosen candidate",
            "classifiers that accept it were built with class_weight='balanced'; a "
            "categorical column is one-hot encoded with sklearn's own max_categories cap "
            "(infrequent/high-cardinality values bucketed together) rather than exploding "
            "into one column per distinct value; numeric scaling switches from "
            "StandardScaler to RobustScaler automatically when the training partition is "
            "outlier-heavy (see training._is_outlier_heavy)",
        ]
        + (
            [
                "the target is strongly right-skewed and non-negative, so every candidate "
                "was fit on log1p(target) and its predictions were inverse-transformed back "
                "before any metric below was computed (sklearn's TransformedTargetRegressor) "
                "— every metric is on the original target scale either way"
            ]
            if target_log_transform
            else []
        )
        + (
            [
                f"a VotingEnsemble over the top {_ENSEMBLE_TOP_K} candidates (by the same "
                "ranking metric) was also fit and scored as one more candidate — see its "
                "'members' hyperparameter for which ones"
            ]
            if len(ensemble_members) >= 2
            else []
        )
        + (
            [
                f"{dropped_missing_target} row(s) with a missing target were excluded "
                + (
                    "(leading / trailing rows trimmed to preserve contiguity for the "
                    "time-ordered split)"
                    if is_time_ordered
                    else "from training"
                )
            ]
            if dropped_missing_target
            else []
        ),
    )


# --- Phase 14.12 — opt-in deep-tune over one named catalog estimator -------

_DEEP_TUNE_N_ITER = 30
_DEEP_TUNE_CV_FOLDS = 5

# A deliberately curated subset of the catalog — the estimator families
# with enough real hyperparameters to make a randomized search worthwhile.
# Each value is a `model__`-unprefixed `RandomizedSearchCV`-style parameter
# distribution (a list is sampled uniformly; `scipy.stats` distributions
# would work too but are not needed for the ranges used here). An
# estimator_name not present here (e.g. `LinearRegression`, `GaussianNB`,
# anything with no real hyperparameters to search) reports `unavailable`
# with an explicit reason rather than silently tuning nothing.
_DEEP_TUNE_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "RandomForestRegressor": {
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": [None, 4, 6, 8, 10, 12, 16, 20, 24],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "RandomForestClassifier": {
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": [None, 4, 6, 8, 10, 12, 16, 20, 24],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "ExtraTreesRegressor": {
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 10],
    },
    "ExtraTreesClassifier": {
        "n_estimators": list(range(50, 401, 25)),
        "max_depth": [None, 4, 6, 8, 10, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 10],
    },
    "GradientBoostingRegressor": {
        "n_estimators": list(range(50, 401, 25)),
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "max_depth": [2, 3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "GradientBoostingClassifier": {
        "n_estimators": list(range(50, 401, 25)),
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "max_depth": [2, 3, 4, 5, 6],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "HistGradientBoostingRegressor": {
        "max_iter": list(range(50, 401, 25)),
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "max_depth": [None, 3, 4, 5, 6, 8, 10],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
    },
    "HistGradientBoostingClassifier": {
        "max_iter": list(range(50, 401, 25)),
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
        "max_depth": [None, 3, 4, 5, 6, 8, 10],
        "l2_regularization": [0.0, 0.01, 0.1, 1.0],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
        "min_samples_split": [2, 4, 6, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20],
        "min_samples_split": [2, 4, 6, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "KNeighborsRegressor": {
        "n_neighbors": list(range(2, 41)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "KNeighborsClassifier": {
        "n_neighbors": list(range(2, 41)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVR": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "SVC": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel": ["rbf", "linear", "poly"],
        "gamma": ["scale", "auto"],
    },
    "Ridge": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
    "RidgeClassifier": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    "Lasso": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    },
    "LogisticRegression": {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
    "MLPRegressor": {
        "hidden_layer_sizes": [
            (16,),
            (32,),
            (64,),
            (128,),
            (32, 16),
            (64, 32),
            (128, 64),
            (64, 64, 32),
            (128, 64, 32),
        ],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
    },
    "MLPClassifier": {
        "hidden_layer_sizes": [
            (16,),
            (32,),
            (64,),
            (128,),
            (32, 16),
            (64, 32),
            (128, 64),
            (64, 64, 32),
            (128, 64, 32),
        ],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
    },
}


def tune_best_candidate(
    df: pd.DataFrame,
    problem: ProblemSpec,
    feature_engineering: FeatureEngineeringSpec,
    readiness: ModelReadiness,
    split: DataSplitPlan,
    *,
    family: ModelFamily,
    estimator_name: str,
    objective: str | None = None,
) -> DeepTuneResult:
    """Opt-in, deliberately separate from `run_expanded_search`'s fixed
    catalog: re-fits one *named* estimator from that catalog with
    `sklearn.model_selection.RandomizedSearchCV` over a wider hyperparameter
    neighborhood than the catalog's own fixed grid explores.

    Still fully reproducible — `RandomizedSearchCV` is given the same
    fixed `MODEL_TRAINING_RANDOM_SEED` this entire module always uses, so
    the same data and the same `(family, estimator_name)` always produce
    the same tuned result. This is the explicit tradeoff documented for
    this feature: wider search, same determinism guarantee, strictly
    slower (``_DEEP_TUNE_N_ITER`` fits, each cross-validated over
    ``_DEEP_TUNE_CV_FOLDS`` folds) than any single catalog candidate.

    `objective` is accepted and ignored, exactly like `run_expanded_search`
    — it has already shaped the upstream `ProblemSpec` this function reads;
    nothing here re-interprets it.
    """
    del objective
    try:
        prep = _prepare_supervised_run(
            df, problem, feature_engineering, readiness, split, require_catalog=False
        )
    except _PreparationError as exc:
        return DeepTuneResult(
            status=ModelingStatus.UNAVAILABLE, reason=exc.reason, task_type=exc.task_type
        )

    matching = [
        spec
        for spec in _expanded_catalog(prep.category)
        if spec.estimator_name == estimator_name and spec.family is family
    ]
    if not matching:
        return DeepTuneResult(
            status=ModelingStatus.UNAVAILABLE,
            reason=(
                f"'{estimator_name}' (family '{family.value}') is not in the Phase 7.7 "
                f"catalog for this task"
            ),
            task_type=prep.task.value,
            family=family,
            estimator_name=estimator_name,
        )

    param_distributions = _DEEP_TUNE_DISTRIBUTIONS.get(estimator_name)
    if not param_distributions:
        return DeepTuneResult(
            status=ModelingStatus.UNAVAILABLE,
            reason=(
                f"'{estimator_name}' has no deep-tune hyperparameter space defined — either "
                "it has no real hyperparameters to search, or it is not yet covered"
            ),
            task_type=prep.task.value,
            family=family,
            estimator_name=estimator_name,
        )

    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    start = _perf_counter()
    try:
        base_estimator = matching[0].build()
        pipeline = Pipeline([("preprocess", prep.preprocessor), ("model", base_estimator)])
        estimator: Any = pipeline
        if prep.target_log_transform:
            from sklearn.compose import TransformedTargetRegressor

            estimator = TransformedTargetRegressor(
                regressor=pipeline, func=np.log1p, inverse_func=np.expm1
            )
            prefixed = {f"regressor__model__{k}": v for k, v in param_distributions.items()}
        else:
            prefixed = {f"model__{k}": v for k, v in param_distributions.items()}

        scoring = _CV_SCORING.get(prep.selection_metric) if prep.selection_metric else None
        n_classes = len(np.unique(prep.y_train)) if prep.category == "classification" else 0
        cv_folds = _DEEP_TUNE_CV_FOLDS
        if prep.category == "classification":
            cv_folds = min(cv_folds, int(np.bincount(prep.y_train.astype(int)).min()))
        cv_folds = max(cv_folds, 2)

        search = RandomizedSearchCV(
            estimator,
            param_distributions=prefixed,
            n_iter=_DEEP_TUNE_N_ITER,
            cv=cv_folds,
            scoring=scoring,
            random_state=MODEL_TRAINING_RANDOM_SEED,
            n_jobs=1,
        )
        search.fit(prep.x_train, prep.y_train)

        y_pred = np.asarray(search.predict(prep.x_test))
        if prep.category == "regression":
            metrics = _regression_metrics(prep.y_test.astype(float), y_pred.astype(float))
        else:
            metrics = _classification_metrics(prep.y_test, y_pred, None, n_classes=n_classes)

        best_params = {
            k.split("__")[-1]: (list(v) if isinstance(v, tuple) else v)
            for k, v in search.best_params_.items()
        }

        return DeepTuneResult(
            status=ModelingStatus.COMPLETED,
            task_type=prep.task.value,
            family=family,
            estimator_name=estimator_name,
            best_hyperparameters=best_params,
            metrics=metrics,
            n_iterations=_DEEP_TUNE_N_ITER,
            cv_folds=cv_folds,
            fit_seconds=round(_perf_counter() - start, 4),
            notes=[
                f"RandomizedSearchCV over {_DEEP_TUNE_N_ITER} candidate hyperparameter "
                f"combinations, each scored with {cv_folds}-fold cross-validation — "
                f"{_DEEP_TUNE_N_ITER * cv_folds} total fits",
                f"random seed: {MODEL_TRAINING_RANDOM_SEED} (fixed) — same data and the same "
                "(family, estimator_name) always produce this same tuned result",
                "compare best_hyperparameters and metrics above against this same "
                "estimator's entry in the catalog search result to see what tuning changed",
            ],
        )
    except Exception as exc:  # noqa: BLE001 - deterministic, normalised failure record
        return DeepTuneResult(
            status=ModelingStatus.UNAVAILABLE,
            reason=f"{type(exc).__name__}: {_normalise_error(str(exc))}",
            task_type=prep.task.value,
            family=family,
            estimator_name=estimator_name,
            fit_seconds=round(_perf_counter() - start, 4),
        )
