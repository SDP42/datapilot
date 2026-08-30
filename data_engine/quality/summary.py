"""Build the dataset-level :class:`QualitySummary` from a list of findings."""

from __future__ import annotations

from .models import (
    FindingType,
    QualityFinding,
    QualitySummary,
    Severity,
)
from .thresholds import SEVERITY_PENALTY


def build_summary(findings: list[QualityFinding], *, n_rows: int, n_columns: int) -> QualitySummary:
    by_severity: dict[Severity, int] = {s: 0 for s in Severity}
    by_type: dict[FindingType, int] = {t: 0 for t in FindingType}
    columns_with_findings: list[str] = []

    for f in findings:
        by_severity[f.severity] += 1
        by_type[f.finding_type] += 1
        for col in f.columns:
            if col not in columns_with_findings:
                columns_with_findings.append(col)

    penalty = sum(SEVERITY_PENALTY[f.severity] for f in findings)
    score = round(max(0.0, 100.0 - penalty), 2)

    return QualitySummary(
        n_rows=n_rows,
        n_columns=n_columns,
        total_findings=len(findings),
        findings_by_severity=by_severity,
        findings_by_type=by_type,
        columns_with_findings=columns_with_findings,
        has_critical=by_severity[Severity.CRITICAL] > 0,
        score=score,
    )
