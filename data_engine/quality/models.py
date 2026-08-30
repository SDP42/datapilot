"""Structured, machine-readable models for data-quality analysis.

Mirrors the profiling architecture: everything is Pydantic v2 and
JSON-serialisable so the future cleaning engine and AI engine can consume
a report with ``report.model_dump(mode="json")`` — never by parsing prose.

This module defines *what a finding is*. It contains no detection logic
and no cleaning logic.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

QUALITY_ENGINE_VERSION = "1"


class Severity(str, Enum):
    """How much a finding should worry a downstream consumer.

    Severity reflects *impact and prevalence*, not certainty that
    something is broken (see ``confidence`` on a finding for that).
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Ordering helper for summaries / sorting (low -> critical).
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class FindingType(str, Enum):
    """The category of data-quality issue detected."""

    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    POTENTIAL_TYPE_MISMATCH = "potential_type_mismatch"
    INCONSISTENT_CATEGORIES = "inconsistent_categories"
    POTENTIAL_OUTLIERS = "potential_outliers"
    HIGH_SKEW = "high_skew"
    CLASS_IMBALANCE = "class_imbalance"


class SuggestedAction(str, Enum):
    """A *pointer* to what a later, explicit cleaning step might do.

    These are suggestions for humans / the AI planner. The quality engine
    never performs them.
    """

    REVIEW_AND_DECIDE = "review_and_decide"
    HANDLE_MISSING_VALUES = "handle_missing_values"
    REMOVE_DUPLICATE_ROWS = "remove_duplicate_rows"
    CONVERT_COLUMN_TYPE = "convert_column_type"
    STANDARDIZE_CATEGORY_VALUES = "standardize_category_values"
    INVESTIGATE_OUTLIERS = "investigate_outliers"
    CONSIDER_DISTRIBUTION_TRANSFORM = "consider_distribution_transform"
    ADDRESS_CLASS_IMBALANCE = "address_class_imbalance"


class QualityFinding(BaseModel):
    """One detected data-quality issue."""

    finding_id: str = Field(
        description="Stable id for this finding within a report, e.g. 'missing_values:age'."
    )
    finding_type: FindingType
    severity: Severity
    columns: list[str] = Field(
        default_factory=list,
        description="Affected column(s). Empty for dataset-level findings (e.g. duplicate rows).",
    )
    affected_rows: int | None = Field(
        default=None, description="Number of rows involved, where that is meaningful."
    )
    affected_percentage: float | None = Field(
        default=None, description="Percentage of rows involved (0-100), where meaningful."
    )
    observed: dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-readable statistics backing the finding (JSON primitives only).",
    )
    description: str = Field(description="Human-readable explanation of what was observed.")
    recommended_action: SuggestedAction
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="For heuristic checks: how sure the engine is (0-1). None for exact checks.",
    )


class QualitySummary(BaseModel):
    """Dataset-level roll-up of all findings."""

    n_rows: int
    n_columns: int
    total_findings: int
    findings_by_severity: dict[Severity, int]
    findings_by_type: dict[FindingType, int]
    columns_with_findings: list[str]
    has_critical: bool
    score: float = Field(
        description=(
            "Heuristic 0-100 quality score (100 = no findings). A weighted penalty per "
            "finding severity, floored at 0. Not a statistical measure — a triage aid."
        )
    )


class QualityReport(BaseModel):
    """The complete result of running the quality engine on one dataset."""

    dataset_id: str
    quality_engine_version: str = QUALITY_ENGINE_VERSION
    generated_at: _dt.datetime
    target_column: str | None = None
    summary: QualitySummary
    findings: list[QualityFinding]
