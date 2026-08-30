"""Pure functions that map an observed statistic to a :class:`Severity`.

Isolated here so the rules are easy to read, test, and adjust without
touching detection logic.
"""

from __future__ import annotations

from .models import Severity
from .thresholds import (
    DUPLICATE_LOW_MAX_PCT,
    DUPLICATE_MEDIUM_MAX_PCT,
    MISSING_HIGH_MAX_PCT,
    MISSING_LOW_MAX_PCT,
    MISSING_MEDIUM_MAX_PCT,
    OUTLIER_LOW_MAX_PCT,
    SKEW_SEVERE_ABS,
)


def severity_from_missing_pct(pct: float) -> Severity:
    if pct <= MISSING_LOW_MAX_PCT:
        return Severity.LOW
    if pct <= MISSING_MEDIUM_MAX_PCT:
        return Severity.MEDIUM
    if pct <= MISSING_HIGH_MAX_PCT:
        return Severity.HIGH
    return Severity.CRITICAL


def severity_from_duplicate_pct(pct: float) -> Severity:
    if pct <= DUPLICATE_LOW_MAX_PCT:
        return Severity.LOW
    if pct <= DUPLICATE_MEDIUM_MAX_PCT:
        return Severity.MEDIUM
    return Severity.HIGH


def severity_from_outlier_pct(pct: float) -> Severity:
    return Severity.LOW if pct <= OUTLIER_LOW_MAX_PCT else Severity.MEDIUM


def severity_from_skew(skew_abs: float) -> Severity:
    return Severity.MEDIUM if skew_abs >= SKEW_SEVERE_ABS else Severity.LOW
