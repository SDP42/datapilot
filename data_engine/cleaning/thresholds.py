"""Tunable thresholds for the deterministic cleaning planner, in one place."""

from __future__ import annotations

from .models import OperationStatus

# --- Missing values ---------------------------------------------------
# At or below this %, a standard imputation is "relatively safe" and the
# operation is RECOMMENDED. Above it, imputation is REVIEW_REQUIRED
# because it materially reshapes the column.
MISSING_SAFE_IMPUTE_MAX_PCT = 5.0

# At or above this %, the planner *additionally* surfaces a
# "drop this column" proposal — always NOT_SAFE_TO_AUTOMATE.
MISSING_HIGH_PCT = 50.0

# Order used when sorting operations in a plan (safest first).
STATUS_ORDER: dict[OperationStatus, int] = {
    OperationStatus.RECOMMENDED: 0,
    OperationStatus.REVIEW_REQUIRED: 1,
    OperationStatus.NOT_SAFE_TO_AUTOMATE: 2,
}
