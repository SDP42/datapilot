"""Before/after quality comparison for the execution report.

Runs the existing quality engine on the original and the processed
dataset and summarises what changed — including regressions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data_engine.quality.models import FindingType, QualityReport

from .statistics import total_missing_cells


def _snapshot(report: QualityReport, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_findings": report.summary.total_findings,
        "score": report.summary.score,
        "has_critical": report.summary.has_critical,
        "n_rows": len(df),
        "n_columns": int(df.shape[1]),
        "total_missing_cells": total_missing_cells(df),
        "findings_by_type": {k.value: v for k, v in report.summary.findings_by_type.items() if v},
    }


def compare_quality(
    before_report: QualityReport,
    after_report: QualityReport,
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    target_column: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], list[str]]:
    before = _snapshot(before_report, before_df)
    after = _snapshot(after_report, after_df)

    improvements: list[str] = []
    regressions: list[str] = []
    notes: list[str] = []

    for finding_type in FindingType:
        b = before_report.summary.findings_by_type.get(finding_type, 0)
        a = after_report.summary.findings_by_type.get(finding_type, 0)
        if a < b:
            improvements.append(f"{finding_type.value}: {b} -> {a}")
        elif a > b:
            regressions.append(f"{finding_type.value}: {b} -> {a} (new issue introduced)")

    if after["total_missing_cells"] > before["total_missing_cells"]:
        regressions.append(
            f"missing cells increased {before['total_missing_cells']} -> "
            f"{after['total_missing_cells']}"
        )
    if target_column and target_column not in after_df.columns:
        regressions.append(f"target column '{target_column}' is missing from the processed dataset")

    lost_columns = sorted(set(before_df.columns) - set(after_df.columns))
    if lost_columns:
        notes.append(f"columns removed by the plan: {lost_columns}")
    if before["n_rows"] != after["n_rows"]:
        notes.append(f"row count {before['n_rows']} -> {after['n_rows']} (duplicate removal)")
    if after_report.summary.score >= before_report.summary.score:
        notes.append(f"quality score {before_report.summary.score} -> {after_report.summary.score}")
    else:
        regressions.append(
            f"quality score dropped {before_report.summary.score} -> {after_report.summary.score}"
        )

    return before, after, improvements, regressions, notes
