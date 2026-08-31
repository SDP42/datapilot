"""Deterministic paired / one-sided non-parametric tests.

Three related-samples tests, complementing the independent-sample
non-parametric foundation (``analyze_nonparametric``):

* :func:`wilcoxon_signed_rank` — ``(x, y, *, alternative="two-sided")``
* :func:`sign_test` — ``(x, y, *, alternative="two-sided")``
* :func:`friedman_test` — ``(*samples)``

All inputs are **positionally paired** array-likes (list / ``numpy``
array / ``pandas`` Series) — pairing is never inferred and the caller
supplies it. Observations are **not** sorted, **not** reordered, and
**not** imputed. Missing / non-finite pairs are dropped listwise.

Invalid API arguments (length mismatch, an unknown ``alternative``,
fewer than three Friedman groups) raise ``ValueError``. Ordinary data
degeneracy (too few usable observations, all-zero differences, a
non-finite SciPy result) returns ``status = unavailable`` + a ``reason``
— never a fabricated ``0`` / ``1`` / ``False``.

Deterministic: no randomness, no sampling, no seed, no timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from .paired_nonparametric_models import (
    FRIEDMAN_MIN_GROUPS,
    PAIRED_MIN_OBSERVATIONS,
    PAIRED_NONPARAMETRIC_ROUND,
    VALID_ALTERNATIVES,
    PairedNonParametricResult,
    PairedNonParametricStatus,
    PairedNonParametricTestName,
)
from .statistical_models import DEFAULT_ALPHA
from .univariate import _clean_float

_ROUND = PAIRED_NONPARAMETRIC_ROUND

ArrayLike = Sequence[float] | np.ndarray


def _as_float_array(values: ArrayLike, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    return array


def _require_alternative(alternative: str) -> None:
    if not isinstance(alternative, str) or alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES!r} (got {alternative!r})")


def _unavailable(
    test_name: PairedNonParametricTestName,
    reason: str,
    *,
    alternative: str | None,
    alpha: float,
    n_groups: int | None = None,
    n_zero: int | None = None,
    n_positive: int | None = None,
    n_negative: int | None = None,
) -> PairedNonParametricResult:
    return PairedNonParametricResult(
        test_name=test_name,
        alternative=alternative,
        status=PairedNonParametricStatus.UNAVAILABLE,
        reason=reason,
        alpha=alpha,
        n_groups=n_groups,
        n_zero=n_zero,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def _completed(
    test_name: PairedNonParametricTestName,
    *,
    statistic: float,
    p_value: float,
    alternative: str | None,
    alpha: float,
    n_observations: int,
    n_groups: int | None = None,
    n_positive: int | None = None,
    n_negative: int | None = None,
    n_zero: int | None = None,
    notes: list[str] | None = None,
) -> PairedNonParametricResult:
    clean_p = _clean_float(p_value)
    return PairedNonParametricResult(
        test_name=test_name,
        alternative=alternative,
        status=PairedNonParametricStatus.COMPLETED,
        statistic=round(statistic, _ROUND),
        p_value=clean_p,
        n_observations=n_observations,
        n_groups=n_groups,
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        alpha=alpha,
        significant=bool(clean_p < alpha) if clean_p is not None else None,
        notes=notes or [],
    )


def _paired_differences(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, int]:
    """Return the finite paired differences ``x - y`` and the count of
    dropped (missing / non-finite) pairs. Raises on a length mismatch."""
    xa = _as_float_array(x, "x")
    ya = _as_float_array(y, "y")
    if xa.size != ya.size:
        raise ValueError(f"x and y must have the same length ({xa.size} != {ya.size})")
    diff = xa - ya
    finite = np.isfinite(diff)
    return diff[finite], int(diff.size - int(finite.sum()))


# --- Wilcoxon signed-rank -------------------------------------------------


def wilcoxon_signed_rank(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    alpha: float = DEFAULT_ALPHA,
) -> PairedNonParametricResult:
    """Wilcoxon signed-rank test on the positionally-paired samples ``x``,
    ``y`` (``d = x - y``).

    * H0: the paired differences are symmetric about zero.
    * ``alternative="greater"`` → H1: ``x`` tends to exceed ``y``;
      ``"less"`` → H1: ``x`` tends to be below ``y`` (exact SciPy
      ``alternative`` semantics).
    * Zero differences are dropped (``zero_method="wilcox"``). SciPy
      chooses the exact vs. normal-approximation method automatically
      (``method="auto"``).
    * Needs at least ``PAIRED_MIN_OBSERVATIONS`` non-zero differences.
    """
    _require_alternative(alternative)
    name = PairedNonParametricTestName.WILCOXON_SIGNED_RANK
    diff, dropped = _paired_differences(x, y)
    notes = [f"{dropped} non-finite pair(s) dropped"] if dropped else []

    n_zero = int(np.count_nonzero(diff == 0.0))
    nonzero = diff[diff != 0.0]
    if nonzero.size < PAIRED_MIN_OBSERVATIONS:
        return _unavailable(
            name,
            f"only {nonzero.size} non-zero paired difference(s); need at least "
            f"{PAIRED_MIN_OBSERVATIONS}",
            alternative=alternative,
            alpha=alpha,
            n_zero=n_zero,
        )

    result = stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            name,
            "SciPy returned a non-finite statistic or p-value",
            alternative=alternative,
            alpha=alpha,
            n_zero=n_zero,
        )
    return _completed(
        name,
        statistic=statistic,
        p_value=p_value,
        alternative=alternative,
        alpha=alpha,
        n_observations=int(nonzero.size),
        n_zero=n_zero,
        notes=notes,
    )


# --- sign test ---------------------------------------------------------------


def sign_test(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    alpha: float = DEFAULT_ALPHA,
) -> PairedNonParametricResult:
    """Binomial sign test on the signs of the non-zero paired differences
    ``d = x - y``.

    * H0: ``P(d > 0) = 0.5`` among the non-zero differences.
    * ``alternative="greater"`` → H1: ``P(d > 0) > 0.5`` (``x`` above
      ``y``); ``"less"`` → H1: ``P(d > 0) < 0.5``.
    * Zero differences are excluded from both counts. The test is
      ``scipy.stats.binomtest(n_positive, n_nonzero, 0.5, alternative)``.
    * ``statistic`` = the number of positive differences (the binomial
      success count) — a count, not a continuous effect size.
    * Needs at least ``PAIRED_MIN_OBSERVATIONS`` non-zero differences.
    """
    _require_alternative(alternative)
    name = PairedNonParametricTestName.SIGN_TEST
    diff, dropped = _paired_differences(x, y)
    notes = [f"{dropped} non-finite pair(s) dropped"] if dropped else []

    n_positive = int(np.count_nonzero(diff > 0.0))
    n_negative = int(np.count_nonzero(diff < 0.0))
    n_zero = int(np.count_nonzero(diff == 0.0))
    n_nonzero = n_positive + n_negative
    if n_nonzero < PAIRED_MIN_OBSERVATIONS:
        return _unavailable(
            name,
            f"only {n_nonzero} non-zero paired difference(s); need at least "
            f"{PAIRED_MIN_OBSERVATIONS}",
            alternative=alternative,
            alpha=alpha,
            n_zero=n_zero,
            n_positive=n_positive,
            n_negative=n_negative,
        )

    result = stats.binomtest(n_positive, n_nonzero, 0.5, alternative=alternative)
    p_value = float(result.pvalue)
    if not np.isfinite(p_value):
        return _unavailable(
            name,
            "SciPy returned a non-finite p-value",
            alternative=alternative,
            alpha=alpha,
            n_zero=n_zero,
            n_positive=n_positive,
            n_negative=n_negative,
        )
    return _completed(
        name,
        statistic=float(n_positive),
        p_value=p_value,
        alternative=alternative,
        alpha=alpha,
        n_observations=n_nonzero,
        n_positive=n_positive,
        n_negative=n_negative,
        n_zero=n_zero,
        notes=[*notes, "statistic is the number of positive differences"],
    )


# --- Friedman --------------------------------------------------------------


def friedman_test(
    *samples: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
) -> PairedNonParametricResult:
    """Friedman test for three or more **related** (repeated-measures)
    samples, supplied in a deterministic caller-chosen order.

    * H0: the related groups have the same distribution / location.
    * All samples must have the same length (one row = one block).
      A row with a missing / non-finite value in **any** sample is
      dropped listwise.
    * Fewer than ``FRIEDMAN_MIN_GROUPS`` samples, or unequal lengths,
      raise ``ValueError``. Fewer than ``PAIRED_MIN_OBSERVATIONS``
      complete blocks, or a non-finite SciPy result, → unavailable.
    * ``scipy.stats.friedmanchisquare`` — this is **not** ANOVA and
      **not** Kruskal-Wallis.
    """
    name = PairedNonParametricTestName.FRIEDMAN
    if len(samples) < FRIEDMAN_MIN_GROUPS:
        raise ValueError(
            f"the Friedman test needs at least {FRIEDMAN_MIN_GROUPS} related samples "
            f"(got {len(samples)})"
        )
    arrays = [_as_float_array(sample, f"sample {i}") for i, sample in enumerate(samples)]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"all Friedman samples must have the same length (got {sorted(lengths)})")

    matrix = np.column_stack(arrays)
    complete = np.all(np.isfinite(matrix), axis=1)
    dropped = int(matrix.shape[0] - int(complete.sum()))
    matrix = matrix[complete]
    notes = [f"{dropped} incomplete block(s) dropped"] if dropped else []

    n_blocks = int(matrix.shape[0])
    n_groups = len(arrays)
    if n_blocks < PAIRED_MIN_OBSERVATIONS:
        return _unavailable(
            name,
            f"only {n_blocks} complete block(s); need at least {PAIRED_MIN_OBSERVATIONS}",
            alternative=None,
            alpha=alpha,
            n_groups=n_groups,
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        # identical groups make the Friedman denominator zero; the non-finite
        # check below turns that into an explicit `unavailable` result.
        result = stats.friedmanchisquare(*[matrix[:, j] for j in range(n_groups)])
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        return _unavailable(
            name,
            "SciPy returned a non-finite statistic or p-value (the groups may be identical)",
            alternative=None,
            alpha=alpha,
            n_groups=n_groups,
        )
    return _completed(
        name,
        statistic=statistic,
        p_value=p_value,
        alternative=None,
        alpha=alpha,
        n_observations=n_blocks,
        n_groups=n_groups,
        notes=notes,
    )
