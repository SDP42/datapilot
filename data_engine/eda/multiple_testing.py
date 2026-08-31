"""Deterministic multiple-testing correction for a family of p-values.

:func:`correct_multiple_testing` takes **already-computed** p-values and
returns corrected p-values plus rejection decisions, without recomputing
anything and without touching any existing statistical-test output.

Methods (all implemented directly on top of NumPy — SciPy has no
Bonferroni/Holm helper, so all three are done here for consistency and
version-independence):

* **Bonferroni** — ``p_i* = min(1, m · p_i)``; controls the family-wise
  error rate (FWER).
* **Holm** — step-down FWER. Sort ascending; ``p_(j)* = max_{i ≤ j}
  min(1, (m − i) · p_(i))`` (enforced monotone non-decreasing); reject
  every hypothesis up to the first ``j`` with ``p_(j) > α / (m − j)``.
* **Benjamini-Hochberg** — FDR. Sort ascending; ``p_(j)* = min_{i ≥ j}
  min(1, (m / (i + 1)) · p_(i))``; reject every hypothesis up to the
  largest ``j`` with ``p_(j) ≤ ((j + 1) / m) · α``.

The **output preserves the input order** — internal sorting is by index
and mapped back. Corrected p-values stay in ``[0, 1]``. ``0.0`` and
``1.0`` are valid inputs. NaN / ±inf / out-of-range p-values are
**rejected** (``status = unavailable``), never clipped. Invalid API
arguments (unknown method, bad alpha type, label-length mismatch) raise
``TypeError`` / ``ValueError``.

Deterministic: no randomness, no sampling, no seed, no timestamps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .multiple_testing_models import (
    _METHOD_CONTROLS,
    CORRECTION_METHOD_ALIASES,
    DEFAULT_CORRECTION_ALPHA,
    DEFAULT_CORRECTION_METHOD,
    MULTIPLE_TESTING_ROUND,
    CorrectionMethod,
    MultipleTestingCorrectionResult,
    MultipleTestingStatus,
)

_ROUND = MULTIPLE_TESTING_ROUND


def _resolve_method(method: str) -> CorrectionMethod:
    if not isinstance(method, str):
        raise TypeError(f"method must be a string (got {type(method).__name__})")
    resolved = CORRECTION_METHOD_ALIASES.get(method.strip().lower())
    if resolved is None:
        raise ValueError(
            f"unknown correction method {method!r}; supported: "
            f"{sorted(set(CORRECTION_METHOD_ALIASES))}"
        )
    return resolved


def _resolve_alpha(alpha: object) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a real number in (0, 1), not {type(alpha).__name__}")
    value = float(alpha)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"alpha must be in the open interval (0, 1) (got {value})")
    return value


def _bonferroni(p: np.ndarray, m: int) -> np.ndarray:
    return np.minimum(1.0, p * m)


def _holm(p: np.ndarray, m: int) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted = np.minimum(1.0, ranked * (m - np.arange(m)))
    adjusted = np.maximum.accumulate(adjusted)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out


def _benjamini_hochberg(p: np.ndarray, m: int) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    factors = m / (np.arange(m) + 1.0)
    adjusted = np.minimum(1.0, ranked * factors)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out


_CORRECTORS = {
    CorrectionMethod.BONFERRONI: _bonferroni,
    CorrectionMethod.HOLM: _holm,
    CorrectionMethod.BENJAMINI_HOCHBERG: _benjamini_hochberg,
}


def correct_multiple_testing(
    p_values: Sequence[float],
    *,
    method: str = DEFAULT_CORRECTION_METHOD,
    alpha: float = DEFAULT_CORRECTION_ALPHA,
    labels: Sequence[str] | None = None,
) -> MultipleTestingCorrectionResult:
    """Correct a family of p-values.

    ``method`` — ``"bonferroni"`` / ``"holm"`` / ``"benjamini_hochberg"``
    (aliases ``"bh"``, ``"fdr_bh"``, ``"holm-bonferroni"`` accepted).
    ``alpha`` — significance level in ``(0, 1)``. ``labels`` — optional,
    same length as ``p_values``, preserved in input order.

    Raises ``TypeError`` / ``ValueError`` for invalid API arguments.
    Returns ``status = unavailable`` (with a ``reason``) when a p-value is
    NaN, infinite, or outside ``[0, 1]``, or when the input is empty.
    """
    resolved_method = _resolve_method(method)
    resolved_alpha = _resolve_alpha(alpha)

    raw = list(p_values)
    label_list: list[str] | None = None
    if labels is not None:
        label_list = [str(label) for label in labels]
        if len(label_list) != len(raw):
            raise ValueError(
                f"labels and p_values must have the same length ({len(label_list)} != {len(raw)})"
            )

    def _unavailable(reason: str) -> MultipleTestingCorrectionResult:
        return MultipleTestingCorrectionResult(
            method=resolved_method,
            controls=_METHOD_CONTROLS[resolved_method],
            alpha=resolved_alpha,
            n_hypotheses=len(raw),
            labels=label_list,
            status=MultipleTestingStatus.UNAVAILABLE,
            reason=reason,
        )

    if not raw:
        return _unavailable("no p-values supplied")

    numeric: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise TypeError(f"p-value at index {index} is not a real number ({value!r})")
        number = float(value)
        if not math.isfinite(number) or not (0.0 <= number <= 1.0):
            return _unavailable(
                f"p-value at index {index} is not a finite value in [0, 1] (got {number})"
            )
        numeric.append(number)

    p = np.asarray(numeric, dtype=float)
    m = int(p.size)
    corrected = _CORRECTORS[resolved_method](p, m)
    corrected = np.round(np.clip(corrected, 0.0, 1.0), _ROUND)
    rejected = corrected <= resolved_alpha

    return MultipleTestingCorrectionResult(
        method=resolved_method,
        controls=_METHOD_CONTROLS[resolved_method],
        alpha=resolved_alpha,
        n_hypotheses=m,
        labels=label_list,
        p_values=[float(value) for value in numeric],
        corrected_p_values=[float(value) for value in corrected],
        rejected=[bool(flag) for flag in rejected],
        n_rejected=int(rejected.sum()),
        status=MultipleTestingStatus.COMPLETED,
    )
