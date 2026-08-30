"""Data-quality analysis engine — **detection only**.

Examines an ingested dataset and its :class:`DatasetProfile` and produces
a structured, machine-readable :class:`QualityReport` of findings
(missing values, duplicates, type mismatches, inconsistent categories,
outliers, skew, class imbalance).

It never modifies the DataFrame or the raw file and never fixes anything.
Cleaning is a separate, later phase that consumes these findings.

Typical use::

    from data_engine.ingestion import ingest_dataset
    from data_engine.quality import analyze_quality

    ref = ingest_dataset("customers.csv")
    report = analyze_quality(ref, target_column="churned")
    payload = report.model_dump(mode="json")
"""

from __future__ import annotations

from .analyzer import (
    CHECKS,
    analyze_dataframe,
    analyze_profile,
    analyze_quality,
    available_checks,
)
from .models import (
    FindingType,
    QualityFinding,
    QualityReport,
    QualitySummary,
    Severity,
    SuggestedAction,
)

__all__ = [
    "CHECKS",
    "FindingType",
    "QualityFinding",
    "QualityReport",
    "QualitySummary",
    "Severity",
    "SuggestedAction",
    "analyze_dataframe",
    "analyze_profile",
    "analyze_quality",
    "available_checks",
]
