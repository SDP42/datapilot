"""Deterministic k-NN / Kraskov mutual-information estimator (KSG estimator 1).

:func:`estimate_mutual_information_knn` gives a **continuous** mutual-
information estimate for two **numeric** columns, without any binning —
the complement to the discrete plug-in estimate in ``effects.py``.

Method (Kraskov, Stögbauer & Grassberger 2004, "estimator 1"):

    I(X; Y) = ψ(k) + ψ(N) - (1/N) Σ_i [ ψ(n_x(i) + 1) + ψ(n_y(i) + 1) ]

where, for each point ``i`` over the ``N`` paired finite observations:

* the **joint space** is the 2-D point ``(x_i, y_i)`` and distance is the
  **Chebyshev / L-infinity** norm;
* ``eps_i`` = the distance from ``i`` to its ``k``-th nearest neighbour in
  the joint space (self excluded);
* ``n_x(i)`` = number of other points whose ``|x - x_i|`` is **strictly
  less than** ``eps_i`` — implemented as a closed-ball count at radius
  ``np.nextafter(eps_i, 0)`` (the largest float below ``eps_i``), which is
  the deterministic strict-``<`` convention used by e.g. scikit-learn;
* ``n_y(i)`` likewise on the Y marginal;
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
    KNN_MI_ROUND,
    KNNMutualInformationResult,
    KNNMutualInformationStatus,
)
from .models import EDAColumnKind
from .univariate import classify_column


def _unavailable(x_column: str, y_column: str, reason: str) -> KNNMutualInformationResult:
    return KNNMutualInformationResult(
        x_column=x_column,
        y_column=y_column,
        status=KNNMutualInformationStatus.UNAVAILABLE,
        reason=reason,
    )


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
                f"column '{column}' is a datetime column; the k-NN MI estimator supports "
                "numeric columns only in this increment",
            )
        if kind is not EDAColumnKind.NUMERIC:
            return _unavailable(
                x_column,
                y_column,
                f"column '{column}' is not numeric ({kind.value}); the k-NN MI estimator "
                "supports numeric columns only",
            )

    if isinstance(k, bool) or not isinstance(k, int):
        return _unavailable(x_column, y_column, f"k must be an integer (got {k!r})")
    if k < 1:
        return _unavailable(x_column, y_column, f"k must be >= 1 (got {k})")

    paired = df[[x_column, y_column]].dropna()
    x = paired[x_column].to_numpy(dtype=float)
    y = paired[y_column].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = int(x.size)

    if n == 0:
        return _unavailable(x_column, y_column, "no paired finite observations remain")
    minimum = max(KNN_MI_MIN_OBSERVATIONS, k + 1)
    if n < minimum:
        return _unavailable(
            x_column,
            y_column,
            f"only {n} paired finite observations; need at least {minimum} for k={k}",
        )
    if k >= n:
        return _unavailable(
            x_column, y_column, f"k={k} must be smaller than the observation count N={n}"
        )
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        constant = x_column if float(np.ptp(x)) == 0.0 else y_column
        return _unavailable(
            x_column, y_column, f"column '{constant}' is constant over the paired observations"
        )

    raw = _kraskov_ksg1(x, y, k)
    if not math.isfinite(raw):
        return _unavailable(x_column, y_column, "the Kraskov estimate is not finite")

    rounded = round(raw, KNN_MI_ROUND)
    notes: list[str] = []
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
        status=KNNMutualInformationStatus.COMPLETED,
        k=k,
        n_observations=n,
        mutual_information=mutual_information,
        notes=notes,
    )
