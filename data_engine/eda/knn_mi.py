"""Deterministic k-NN / Kraskov mutual-information estimators (KSG estimator 1).

Two standalone estimators sharing the same KSG mathematics:

* :func:`estimate_mutual_information_knn` — a **continuous** MI estimate
  for two **numeric** columns, without any binning.
* :func:`estimate_mutual_information_datetime` — the same estimator after
  a **deterministic datetime → numeric** conversion, so a datetime column
  can participate in MI analysis (datetime ↔ numeric, datetime ↔
  datetime).

Both complement — they never replace — the discrete plug-in
``mutual_information`` in ``effects.py``.

Method (Kraskov, Stögbauer & Grassberger 2004, "estimator 1"):

    I(X; Y) = ψ(k) + ψ(N) - (1/N) Σ_i [ ψ(n_x(i) + 1) + ψ(n_y(i) + 1) ]

where, for each point ``i`` over the ``N`` paired finite observations:

* the **joint space** is the 2-D point ``(x_i, y_i)`` and distance is the
  **Chebyshev / L-infinity** norm;
* ``eps_i`` = the distance from ``i`` to its ``k``-th nearest neighbour in
  the joint space (self excluded);
* ``n_x(i)`` = number of other points whose ``|x - x_i|`` is **strictly
  less than** ``eps_i`` — a closed-ball count at radius
  ``np.nextafter(eps_i, 0)`` (the deterministic strict-``<`` convention
  used by scikit-learn); ``n_y(i)`` likewise on the Y marginal;
* ``ψ`` is the digamma function.

The estimate is in **nats**. It is not bounded below, so a small negative
value from floating-point error is clamped to ``0.0`` (recorded in
``notes``). The per-point mean is accumulated with :func:`math.fsum`, so
the result does not depend on DataFrame row order.

Read-only, deterministic, no randomness, no target inference, no model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma

from .knn_mi_models import (
    KNN_MI_DEFAULT_K,
    KNN_MI_MIN_OBSERVATIONS,
    KNN_MI_REPRESENTATION_DATETIME,
    KNN_MI_REPRESENTATION_RAW,
    KNN_MI_ROUND,
    KNNMutualInformationResult,
    KNNMutualInformationStatus,
)
from .models import EDAColumnKind
from .univariate import classify_column

# The Unix epoch in UTC — the fixed, dataset-independent reference for the
# datetime → seconds conversion. It is never the current time.
_EPOCH = pd.Timestamp("1970-01-01T00:00:00", tz="UTC")


def _unavailable(
    x_column: str, y_column: str, reason: str, *, representation: str | None = None
) -> KNNMutualInformationResult:
    return KNNMutualInformationResult(
        x_column=x_column,
        y_column=y_column,
        representation=representation,
        status=KNNMutualInformationStatus.UNAVAILABLE,
        reason=reason,
    )


def _invalid_k_reason(k: object) -> str | None:
    if isinstance(k, bool) or not isinstance(k, int):
        return f"k must be an integer (got {k!r})"
    if k < 1:
        return f"k must be >= 1 (got {k})"
    return None


def _kraskov_ksg1(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """KSG estimator 1 over already-cleaned, equal-length numeric arrays."""
    n = int(x.size)
    joint = np.column_stack([x, y])

    joint_tree = cKDTree(joint)
    # query returns the point itself first (distance 0); column k is the
    # distance to the k-th *other* neighbour.
    distances, _ = joint_tree.query(joint, k=k + 1, p=np.inf)
    eps = distances[:, k]
    radius = np.nextafter(eps, 0.0)  # strict "<" via the next float toward 0

    x_tree = cKDTree(x[:, None])
    y_tree = cKDTree(y[:, None])
    n_x = np.asarray(x_tree.query_ball_point(x[:, None], radius, p=np.inf, return_length=True)) - 1
    n_y = np.asarray(y_tree.query_ball_point(y[:, None], radius, p=np.inf, return_length=True)) - 1

    terms = digamma(n_x + 1) + digamma(n_y + 1)
    mean_terms = math.fsum(terms.tolist()) / n
    return float(digamma(k) + digamma(n) - mean_terms)


def _estimate(
    x_column: str,
    y_column: str,
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    representation: str,
    *,
    standardize: bool = False,
) -> KNNMutualInformationResult:
    """Shared: validate the already-cleaned arrays, run KSG-1, build the result.

    ``standardize`` — divide each marginal by its own mean / std before the
    joint-space distance. Used for the datetime path so the huge
    epoch-second magnitudes do not dominate the Chebyshev distance; it is
    an affine per-variable transform and does not change the population
    mutual information. The numeric-only estimator leaves it ``False`` so
    its behaviour is unchanged.
    """
    n = int(x.size)
    if n == 0:
        return _unavailable(
            x_column,
            y_column,
            "no paired finite observations remain",
            representation=representation,
        )
    minimum = max(KNN_MI_MIN_OBSERVATIONS, k + 1)
    if n < minimum:
        return _unavailable(
            x_column,
            y_column,
            f"only {n} paired finite observations; need at least {minimum} for k={k}",
            representation=representation,
        )
    if k >= n:
        return _unavailable(
            x_column,
            y_column,
            f"k={k} must be smaller than the observation count N={n}",
            representation=representation,
        )
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        constant = x_column if float(np.ptp(x)) == 0.0 else y_column
        return _unavailable(
            x_column,
            y_column,
            f"column '{constant}' is constant over the paired observations",
            representation=representation,
        )

    prep_notes: list[str] = []
    if standardize:
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        prep_notes.append(
            "each column standardized (zero mean, unit std) before the joint-space distance"
        )

    raw = _kraskov_ksg1(x, y, k)
    if not math.isfinite(raw):
        return _unavailable(
            x_column, y_column, "the Kraskov estimate is not finite", representation=representation
        )

    rounded = round(raw, KNN_MI_ROUND)
    notes: list[str] = list(prep_notes)
    if rounded < 0.0:
        notes.append(
            f"raw Kraskov estimate was {rounded}; clamped to 0.0 (KSG estimator 1 is not "
            "bounded below for near-independent variables)"
        )
        mutual_information = 0.0
    else:
        mutual_information = rounded

    return KNNMutualInformationResult(
        x_column=x_column,
        y_column=y_column,
        representation=representation,
        status=KNNMutualInformationStatus.COMPLETED,
        k=k,
        n_observations=n,
        mutual_information=mutual_information,
        notes=notes,
    )


def estimate_mutual_information_knn(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    k: int = KNN_MI_DEFAULT_K,
) -> KNNMutualInformationResult:
    """Continuous k-NN / Kraskov MI estimate for two numeric columns.

    ``x_column`` and ``y_column`` are **required and explicit** — no
    target is inferred. ``df`` is not modified, nothing is written, and no
    dataset version / lineage object is touched.

    Returns ``status = unavailable`` (with a ``reason``) when a column is
    absent, the two columns are the same, a column is not numeric, no
    paired finite observations remain, there are fewer than
    ``max(KNN_MI_MIN_OBSERVATIONS, k + 1)`` of them, ``k`` is invalid, a
    column is constant, or the estimator cannot produce a finite value.
    """
    columns = [str(c) for c in df.columns]
    if x_column not in columns:
        return _unavailable(x_column, y_column, f"column '{x_column}' is not in the DataFrame")
    if y_column not in columns:
        return _unavailable(x_column, y_column, f"column '{y_column}' is not in the DataFrame")
    if x_column == y_column:
        return _unavailable(x_column, y_column, "x_column and y_column are the same column")

    for column in (x_column, y_column):
        kind = classify_column(df[column])
        if kind is EDAColumnKind.DATETIME:
            return _unavailable(
                x_column,
                y_column,
                f"column '{column}' is a datetime column; use "
                "estimate_mutual_information_datetime for datetime involvement",
            )
        if kind is not EDAColumnKind.NUMERIC:
            return _unavailable(
                x_column,
                y_column,
                f"column '{column}' is not numeric ({kind.value}); the k-NN MI estimator "
                "supports numeric columns only",
            )

    invalid_k = _invalid_k_reason(k)
    if invalid_k is not None:
        return _unavailable(x_column, y_column, invalid_k)

    paired = df[[x_column, y_column]].dropna()
    x = paired[x_column].to_numpy(dtype=float)
    y = paired[y_column].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    return _estimate(x_column, y_column, x[finite], y[finite], k, KNN_MI_REPRESENTATION_RAW)


def _to_epoch_seconds(series: pd.Series) -> np.ndarray:
    """Deterministic datetime → float: elapsed seconds since the Unix epoch
    in UTC. Timezone-naive values are read as UTC; timezone-aware values are
    converted to UTC. ``NaT`` becomes ``nan``. No calendar features."""
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return (parsed - _EPOCH).dt.total_seconds().to_numpy(dtype=float)


def estimate_mutual_information_datetime(
    df: pd.DataFrame,
    datetime_column: str,
    other_column: str,
    *,
    k: int = KNN_MI_DEFAULT_K,
) -> KNNMutualInformationResult:
    """Continuous k-NN / Kraskov MI estimate involving a **datetime** column.

    ``datetime_column`` must be a datetime column; ``other_column`` may be
    numeric or datetime. Each datetime column is converted to **elapsed
    seconds since 1970-01-01T00:00:00Z** (UTC; naive timestamps are read as
    UTC) — a fixed, dataset-independent reference, never the current time,
    with no calendar-feature extraction. The converted values are then fed
    to the same KSG estimator 1 as :func:`estimate_mutual_information_knn`.

    ``x_column`` / ``y_column`` in the result are ``datetime_column`` /
    ``other_column``. Returns ``status = unavailable`` (with a ``reason``)
    for an absent column, the same column twice, a non-datetime
    ``datetime_column``, a categorical / unsupported ``other_column``, an
    invalid ``k``, no usable paired observations, too few of them, a
    constant column, non-finite converted values, or a non-finite result.
    ``df`` is not modified and nothing is written.
    """
    representation = KNN_MI_REPRESENTATION_DATETIME
    columns = [str(c) for c in df.columns]
    if datetime_column not in columns:
        return _unavailable(
            datetime_column,
            other_column,
            f"column '{datetime_column}' is not in the DataFrame",
            representation=representation,
        )
    if other_column not in columns:
        return _unavailable(
            datetime_column,
            other_column,
            f"column '{other_column}' is not in the DataFrame",
            representation=representation,
        )
    if datetime_column == other_column:
        return _unavailable(
            datetime_column,
            other_column,
            "datetime_column and other_column are the same column",
            representation=representation,
        )

    if classify_column(df[datetime_column]) is not EDAColumnKind.DATETIME:
        return _unavailable(
            datetime_column,
            other_column,
            f"column '{datetime_column}' is not a datetime column",
            representation=representation,
        )

    other_kind = classify_column(df[other_column])
    if other_kind not in (EDAColumnKind.NUMERIC, EDAColumnKind.DATETIME):
        return _unavailable(
            datetime_column,
            other_column,
            f"column '{other_column}' is {other_kind.value}; datetime MI supports a numeric "
            "or datetime partner only (categorical involvement uses the binned "
            "mutual_information)",
            representation=representation,
        )

    invalid_k = _invalid_k_reason(k)
    if invalid_k is not None:
        return _unavailable(datetime_column, other_column, invalid_k, representation=representation)

    x = _to_epoch_seconds(df[datetime_column])
    if other_kind is EDAColumnKind.DATETIME:
        y = _to_epoch_seconds(df[other_column])
    else:
        y = df[other_column].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    return _estimate(
        datetime_column,
        other_column,
        x[finite],
        y[finite],
        k,
        representation,
        standardize=True,
    )
