"""Result model for the k-NN / Kraskov mutual-information estimator.

Additive to the EDA layer. Pydantic v2, JSON round-trip safe. This model
holds only JSON primitives — no DataFrame, NumPy array, SciPy object,
KDTree, neighbour structure, or figure.

This estimator is **distinct** from the binning-based
``mutual_information`` in ``effects.py``:

* ``effects.mutual_information`` — a *discrete plug-in* estimate; any
  numeric column is quantile-binned first (``MI_NUMERIC_BINS``).
* this module — a *continuous* Kraskov / Kozachenko–Leonenko k-NN
  estimate (KSG estimator 1) for two numeric columns, with no binning.

The two are not expected to agree numerically.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

KNN_MI_ENGINE_VERSION = "1"

# Kraskov's recommended default; small k → lower bias, higher variance.
KNN_MI_DEFAULT_K = 3

# Below this many paired finite observations the estimate is not attempted.
KNN_MI_MIN_OBSERVATIONS = 5

# Fixed identifiers so a result is self-describing and never confused with
# the binning-based estimator.
KNN_MI_ESTIMATOR_NAME = "kraskov_knn"  # KSG estimator 1
KNN_MI_DISTANCE_METRIC = "chebyshev"  # L-infinity in the joint (X, Y) space

# Column-representation identifiers used by ``mutual_information`` estimators
# in this module.
KNN_MI_REPRESENTATION_RAW = "raw_numeric_values"
KNN_MI_REPRESENTATION_DATETIME = "elapsed_seconds_since_unix_epoch_utc"

# Estimates in nats are rounded to this many places for cross-platform
# deterministic representation (mirrors the rest of the EDA layer).
KNN_MI_ROUND = 10


class KNNMutualInformationStatus(str, Enum):
    COMPLETED = "completed"  # a finite, non-negative estimate was produced
    UNAVAILABLE = "unavailable"  # the estimate could not be computed (see `reason`)


class KNNMutualInformationResult(BaseModel):
    """One k-NN / Kraskov mutual-information estimate for a numeric pair."""

    knn_mi_engine_version: str = KNN_MI_ENGINE_VERSION
    estimator: str = Field(
        default=KNN_MI_ESTIMATOR_NAME,
        description="Estimator family — 'kraskov_knn' (KSG estimator 1).",
    )
    distance_metric: str = Field(
        default=KNN_MI_DISTANCE_METRIC,
        description="Joint-space distance — Chebyshev / L-infinity.",
    )

    x_column: str
    y_column: str
    representation: str | None = Field(
        default=None,
        description=(
            "How each column was turned into a real value before estimation — "
            "'raw_numeric_values' for numeric columns, "
            "'elapsed_seconds_since_unix_epoch_utc' for datetime columns. None on results "
            "serialised before this field existed."
        ),
    )

    status: KNNMutualInformationStatus
    reason: str | None = Field(
        default=None,
        description="Explicit reason the estimate is unavailable; None when completed.",
    )

    k: int | None = Field(default=None, description="Number of nearest neighbours actually used.")
    n_observations: int | None = Field(
        default=None,
        description="Paired observations used = rows where both values are finite.",
    )
    mutual_information: float | None = Field(
        default=None,
        description=(
            "Estimated mutual information in nats (natural log), >= 0 after the documented "
            "small-negative clamp. None when unavailable. A genuine 0.0 is a completed result."
        ),
    )

    finite_pair_filtering: bool = Field(
        default=True,
        description="Rows with a missing or non-finite value in either column were excluded.",
    )
    tie_handling: str = Field(
        default=(
            "marginal neighbour counts use radius np.nextafter(eps, 0) so points at exactly "
            "the k-th joint distance are excluded (strict '<'); no random jitter"
        ),
        description="Deterministic tie / epsilon policy.",
    )
    notes: list[str] = Field(default_factory=list)
