"""The quality-analysis pipeline.

    DatasetReference
        -> DatasetProfile        (data_engine.profiling)
        -> CheckContext          (df + profile + optional target)
        -> individual checks     (modular, read-only)
        -> QualityReport

Nothing here mutates the DataFrame or the raw file.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable

import pandas as pd

from data_engine.profiling import load_dataframe, profile_dataframe
from data_engine.profiling.models import DatasetProfile
from datapilot.contracts import DatasetReference

from .checks import (
    categorical_consistency,
    class_imbalance,
    data_types,
    duplicate_rows,
    missing_values,
    outliers,
    skewness,
)
from .context import CheckContext
from .models import SEVERITY_ORDER, QualityFinding, QualityReport
from .summary import build_summary

Check = Callable[[CheckContext], list[QualityFinding]]

# Registry — order here is the order findings are grouped in before sorting.
CHECKS: dict[str, Check] = {
    missing_values.CHECK_NAME: missing_values.check,
    duplicate_rows.CHECK_NAME: duplicate_rows.check,
    data_types.CHECK_NAME: data_types.check,
    categorical_consistency.CHECK_NAME: categorical_consistency.check,
    outliers.CHECK_NAME: outliers.check,
    skewness.CHECK_NAME: skewness.check,
    class_imbalance.CHECK_NAME: class_imbalance.check,
}


def available_checks() -> tuple[str, ...]:
    return tuple(CHECKS)


def _run(ctx: CheckContext, checks: Iterable[Check]) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for check in checks:
        findings.extend(check(ctx))
    # Most severe first; stable within a severity.
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity], reverse=True)
    return findings


def analyze_profile(
    df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    target_column: str | None = None,
    checks: Iterable[str] | None = None,
) -> QualityReport:
    """Run the quality checks against an already-loaded df + its profile.

    ``df`` is read only. Pass ``checks`` to run a subset (names from
    :func:`available_checks`).
    """
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} is not a column in the dataset")

    selected = list(CHECKS.values()) if checks is None else [CHECKS[name] for name in checks]
    ctx = CheckContext(df=df, profile=profile, target_column=target_column)
    findings = _run(ctx, selected)

    return QualityReport(
        dataset_id=profile.dataset_id,
        generated_at=dt.datetime.now(dt.UTC),
        target_column=target_column,
        summary=build_summary(findings, n_rows=profile.n_rows, n_columns=profile.n_columns),
        findings=findings,
    )


def analyze_dataframe(
    df: pd.DataFrame,
    *,
    dataset_id: str = "adhoc",
    target_column: str | None = None,
    checks: Iterable[str] | None = None,
) -> QualityReport:
    """Profile an in-memory DataFrame, then analyse its quality. ``df`` is not modified."""
    profile = profile_dataframe(df, dataset_id=dataset_id)
    return analyze_profile(df, profile, target_column=target_column, checks=checks)


def analyze_quality(
    reference: DatasetReference,
    *,
    target_column: str | None = None,
    profile: DatasetProfile | None = None,
    checks: Iterable[str] | None = None,
) -> QualityReport:
    """Contract-level entrypoint: load the referenced raw dataset (read-only),
    profile it if no profile is supplied, and produce a :class:`QualityReport`.
    """
    df = load_dataframe(reference)
    if profile is None:
        profile = profile_dataframe(df, dataset_id=reference.dataset_id)
    return analyze_profile(df, profile, target_column=target_column, checks=checks)
