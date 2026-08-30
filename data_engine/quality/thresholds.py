"""Every tunable threshold the quality checks use, in one place.

Each constant is documented where it is defined and again in
``docs/data-quality.md``. Keeping them together makes the engine's
behaviour auditable and easy to revisit.
"""

from __future__ import annotations

from .models import Severity

# --- Missing values -------------------------------------------------------
# Severity is driven by the *proportion* missing, not the raw count.
MISSING_LOW_MAX_PCT = 5.0  # (0, 5]     -> LOW
MISSING_MEDIUM_MAX_PCT = 20.0  # (5, 20]    -> MEDIUM
MISSING_HIGH_MAX_PCT = 50.0  # (20, 50]   -> HIGH
# > 50 -> CRITICAL

# --- Duplicate rows ------------------------------------------------------
DUPLICATE_LOW_MAX_PCT = 1.0
DUPLICATE_MEDIUM_MAX_PCT = 10.0

# --- Potential type mismatch ------------------------------------------
# A text column is "really numeric/datetime" if at least this fraction of
# its non-null values parse cleanly.
TYPE_MISMATCH_MIN_PARSE_RATIO = 0.90

# --- Categorical inconsistency ------------------------------------------
# Skip columns that look like free text / identifiers rather than categories.
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_DISTINCT_RATIO = 0.5
CATEGORICAL_INCONSISTENCY_CONFIDENCE = 0.7

# --- Outliers (IQR rule) ----------------------------------------------
# Tukey's fences: values outside [Q1 - k*IQR, Q3 + k*IQR] are "potential"
# outliers. k = 1.5 is the standard "outlier" fence; 3.0 is "far out".
IQR_FENCE_MULTIPLIER = 1.5
OUTLIER_LOW_MAX_PCT = 5.0  # <=5% flagged -> LOW, else MEDIUM
OUTLIER_MIN_NON_NULL = 8  # need enough points for quartiles to mean anything

# --- Skewness -----------------------------------------------------------
# Bulmer's rule of thumb: |skew| > 1 is "highly skewed".
SKEW_HIGH_ABS = 1.0
SKEW_SEVERE_ABS = 2.0  # >= this -> MEDIUM, else LOW
SKEW_MIN_NON_NULL = 8

# --- Class imbalance --------------------------------------------------
# Only runs when the caller passes an explicit target column.
IMBALANCE_MAX_CLASSES = 20  # more distinct values -> treat as non-categorical, skip
IMBALANCE_LOW_RATIO = 1.5  # majority/minority count ratio
IMBALANCE_MEDIUM_RATIO = 4.0  # ~ minority below 20%
IMBALANCE_HIGH_RATIO = 10.0

# --- Score weighting --------------------------------------------------
SEVERITY_PENALTY: dict[Severity, float] = {
    Severity.LOW: 2.0,
    Severity.MEDIUM: 6.0,
    Severity.HIGH: 15.0,
    Severity.CRITICAL: 35.0,
}
