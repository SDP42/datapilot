"""Check: class imbalance in an *explicitly supplied* target column.

This check does nothing unless the caller passes ``target_column``. The
engine never guesses the target — picking a target is a modelling
decision that belongs to a later phase.

Rule: let ``ratio = count(majority class) / count(minority class)``.
``ratio >= 1.5`` -> LOW, ``>= 4`` (minority below ~20%) -> MEDIUM,
``>= 10`` -> HIGH. Columns with more than ``IMBALANCE_MAX_CLASSES``
distinct values are treated as non-categorical and skipped.
"""

from __future__ import annotations

from ..context import CheckContext
from ..models import FindingType, QualityFinding, Severity, SuggestedAction
from ..thresholds import (
    IMBALANCE_HIGH_RATIO,
    IMBALANCE_LOW_RATIO,
    IMBALANCE_MAX_CLASSES,
    IMBALANCE_MEDIUM_RATIO,
)

CHECK_NAME = "class_imbalance"


def _severity(ratio: float) -> Severity | None:
    if ratio >= IMBALANCE_HIGH_RATIO:
        return Severity.HIGH
    if ratio >= IMBALANCE_MEDIUM_RATIO:
        return Severity.MEDIUM
    if ratio >= IMBALANCE_LOW_RATIO:
        return Severity.LOW
    return None


def check(ctx: CheckContext) -> list[QualityFinding]:
    target = ctx.target_column
    if target is None:
        return []
    if target not in ctx.df.columns:
        raise ValueError(f"target_column {target!r} is not a column in the dataset")

    counts = ctx.df[target].value_counts(dropna=True)
    if counts.empty or len(counts) < 2 or len(counts) > IMBALANCE_MAX_CLASSES:
        return []

    majority, minority = int(counts.iloc[0]), int(counts.iloc[-1])
    ratio = majority / minority
    severity = _severity(ratio)
    if severity is None:
        return []

    total = int(counts.sum())
    distribution = {str(k): int(v) for k, v in counts.items()}
    minority_pct = round(100.0 * minority / total, 4)
    return [
        QualityFinding(
            finding_id=f"{FindingType.CLASS_IMBALANCE.value}:{target}",
            finding_type=FindingType.CLASS_IMBALANCE,
            severity=severity,
            columns=[target],
            observed={
                "class_distribution": distribution,
                "n_classes": len(counts),
                "majority_class": str(counts.index[0]),
                "minority_class": str(counts.index[-1]),
                "imbalance_ratio": round(ratio, 4),
                "minority_percentage": minority_pct,
            },
            description=(
                f"Target '{target}' is imbalanced: the minority class "
                f"'{counts.index[-1]}' is {minority_pct:.2f}% of labelled rows "
                f"(majority/minority ratio {ratio:.1f})."
            ),
            recommended_action=SuggestedAction.ADDRESS_CLASS_IMBALANCE,
            confidence=None,
        )
    ]
