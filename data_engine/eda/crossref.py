"""Deterministic EDA <-> data-quality cross-reference.

:func:`cross_reference_eda_quality` walks the findings already present in
a :class:`~data_engine.quality.models.QualityReport` and, for each
finding whose subject column also has a matching observation in the
:class:`~data_engine.eda.models.EDAReport`, emits one structured
correspondence entry.

Guarantees:

* No detector runs here. Quality detection is not touched, no quality
  finding is invented, no target is inferred (class-imbalance entries are
  emitted only when the ``QualityReport`` already carries a
  ``target_column``).
* Neither input object is mutated.
* Wording is fixed template text — never LLM-generated.
* If nothing lines up, the result is empty.
"""

from __future__ import annotations

from data_engine.quality.models import FindingType, QualityReport

from .crossref_models import (
    EDAQualityCrossReference,
    EDAQualityCrossReferenceEntry,
    EDASignalKind,
)
from .distribution_models import DistributionStatus, NumericDistribution
from .models import EDAReport


def _distribution_index(eda_result: EDAReport) -> dict[str, NumericDistribution]:
    return {d.column: d for d in eda_result.distribution.columns}


def _quantile_map(distribution: NumericDistribution) -> dict[float, float | None]:
    return {q.quantile: q.value for q in distribution.quantiles}


def cross_reference_eda_quality(
    eda_result: EDAReport, quality_report: QualityReport
) -> EDAQualityCrossReference:
    """Correlate an existing ``EDAReport`` with an existing ``QualityReport``.

    Read-only for both arguments. Returns an
    :class:`EDAQualityCrossReference` whose ``entries`` are sorted
    deterministically by ``(column, eda_signal, quality_finding_id)``.
    """
    categorical_uni = {c.column: c for c in eda_result.univariate.categorical}
    missingness = {c.column: c for c in eda_result.univariate.missingness.columns}
    distribution = _distribution_index(eda_result)
    column_kinds = eda_result.column_kinds

    entries: list[EDAQualityCrossReferenceEntry] = []
    notes: list[str] = []

    def add(
        column: str | None,
        signal: EDASignalKind,
        finding_id: str,
        finding_type: FindingType,
        severity: str,
        relationship: str,
        evidence: dict[str, float | int | str | bool | None],
    ) -> None:
        entries.append(
            EDAQualityCrossReferenceEntry(
                column=column,
                eda_signal=signal,
                quality_finding_id=finding_id,
                quality_finding_type=finding_type,
                quality_severity=severity,
                relationship=relationship,
                eda_evidence=evidence,
            )
        )

    for finding in quality_report.findings:
        ft = finding.finding_type
        sev = finding.severity.value
        fid = finding.finding_id
        col = finding.columns[0] if finding.columns else None

        if ft is FindingType.MISSING_VALUES and col is not None:
            m = missingness.get(col)
            if m is not None and m.missing_count > 0:
                add(
                    col,
                    EDASignalKind.MISSINGNESS,
                    fid,
                    ft,
                    sev,
                    f"EDA missingness reports {m.missing_count} missing value(s) "
                    f"({m.missing_percentage}%) in '{col}', matching the quality "
                    f"finding '{ft.value}'.",
                    {
                        "eda_missing_count": m.missing_count,
                        "eda_missing_percentage": m.missing_percentage,
                    },
                )
            else:
                notes.append(f"no EDA missingness signal for '{col}' (finding {fid})")

        elif ft is FindingType.HIGH_SKEW and col is not None:
            d = distribution.get(col)
            if d is not None and d.skewness is not None:
                add(
                    col,
                    EDASignalKind.SKEWNESS,
                    fid,
                    ft,
                    sev,
                    f"EDA distribution analysis computes skewness={d.skewness} for "
                    f"'{col}' (adjusted Fisher-Pearson; 0 = symmetric), consistent "
                    f"with the quality finding '{ft.value}'.",
                    {
                        "eda_skewness": d.skewness,
                        "eda_kurtosis": d.kurtosis,
                        "eda_mean": d.mean,
                        "eda_median": d.median,
                    },
                )
            else:
                notes.append(f"no EDA skewness signal for '{col}' (finding {fid})")

        elif ft is FindingType.POTENTIAL_OUTLIERS and col is not None:
            d = distribution.get(col)
            if (
                d is not None
                and d.status is DistributionStatus.COMPLETED
                and d.minimum is not None
                and d.maximum is not None
            ):
                q = _quantile_map(d)
                add(
                    col,
                    EDASignalKind.DISPERSION,
                    fid,
                    ft,
                    sev,
                    f"EDA distribution analysis describes the spread of '{col}' "
                    f"(min={d.minimum}, Q1={q.get(0.25)}, median={d.median}, "
                    f"Q3={q.get(0.75)}, max={d.maximum}); the quality finding "
                    f"'{ft.value}' flags extreme values within that range.",
                    {
                        "eda_minimum": d.minimum,
                        "eda_q25": q.get(0.25),
                        "eda_median": d.median,
                        "eda_q75": q.get(0.75),
                        "eda_maximum": d.maximum,
                        "eda_std": d.std,
                    },
                )
            else:
                notes.append(f"no EDA dispersion signal for '{col}' (finding {fid})")

        elif ft is FindingType.POTENTIAL_TYPE_MISMATCH and col is not None:
            kind = column_kinds.get(col)
            if kind is not None:
                looks_like = finding.observed.get("looks_like")
                if looks_like is None:
                    looks_like = finding.observed.get("inferred_type")
                add(
                    col,
                    EDASignalKind.COLUMN_TYPE,
                    fid,
                    ft,
                    sev,
                    f"EDA classifies '{col}' as {kind.value} (by stored pandas dtype); "
                    f"the quality finding '{ft.value}' reports the values look like a "
                    f"different type. EDA never converts stored types.",
                    {
                        "eda_column_kind": kind.value,
                        "quality_looks_like": None if looks_like is None else str(looks_like),
                    },
                )
            else:
                notes.append(f"no EDA classification for '{col}' (finding {fid})")

        elif ft is FindingType.INCONSISTENT_CATEGORIES and col is not None:
            c = categorical_uni.get(col)
            if c is not None:
                top = ", ".join(f"'{t.value}'" for t in c.top_values[:3])
                add(
                    col,
                    EDASignalKind.CATEGORY_CARDINALITY,
                    fid,
                    ft,
                    sev,
                    f"EDA reports {c.unique_count} distinct value(s) in '{col}' "
                    f"(most frequent: {top or 'n/a'}); the quality finding "
                    f"'{ft.value}' flags some of these as variants of one category.",
                    {
                        "eda_unique_count": c.unique_count,
                        "eda_cardinality_ratio": c.cardinality_ratio,
                    },
                )
            else:
                notes.append(f"no EDA categorical summary for '{col}' (finding {fid})")

        elif ft is FindingType.CLASS_IMBALANCE:
            target = quality_report.target_column
            if target is None:
                notes.append(
                    f"class_imbalance finding {fid} not cross-referenced: "
                    "the quality report carries no target_column"
                )
                continue
            c = categorical_uni.get(target)
            evidence: dict[str, float | int | str | bool | None] = {"quality_target_column": target}
            if c is not None:
                evidence["eda_unique_count"] = c.unique_count
                if c.top_values:
                    evidence["eda_most_frequent_value"] = c.top_values[0].value
                    evidence["eda_most_frequent_frequency"] = c.top_values[0].frequency
            add(
                target,
                EDASignalKind.CLASS_BALANCE,
                fid,
                ft,
                sev,
                f"'{target}' is the analysis target; EDA's categorical summary and "
                f"the quality finding '{ft.value}' both describe an uneven class "
                f"distribution.",
                evidence,
            )

        elif ft is FindingType.DUPLICATE_ROWS:
            notes.append(
                f"quality finding '{ft.value}' ({fid}) has no EDA counterpart: "
                "EDA does not analyse row duplication"
            )
        else:
            notes.append(f"finding {fid} ('{ft.value}') not cross-referenced")

    entries.sort(key=lambda e: (e.column or "", e.eda_signal.value, e.quality_finding_id))
    return EDAQualityCrossReference(entries=entries, notes=notes)
