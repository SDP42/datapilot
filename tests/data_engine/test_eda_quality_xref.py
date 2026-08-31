"""Phase 4 — deterministic EDA <-> data-quality cross-reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine import quality
from data_engine.eda import (
    EDAQualityCrossReference,
    analyze_dataframe,
    cross_reference_eda_quality,
)
from data_engine.eda.crossref_models import EDASignalKind
from data_engine.quality.models import FindingType


@pytest.fixture
def messy() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "amount": np.concatenate([rng.normal(10, 2, n - 4), [500.0, 600.0, 700.0, 800.0]]),
            "age": [float(i % 40) if i % 5 else None for i in range(n)],
            "city": (["London"] * 60) + (["london "] * 30) + (["Paris"] * 30),
            "target": (["yes"] * 112) + (["no"] * 8),
            "count_text": [str(i) for i in range(n)],
        }
    )


@pytest.fixture
def eda(messy):
    return analyze_dataframe(messy)


@pytest.fixture
def report(messy):
    return quality.analyze_dataframe(messy, dataset_id="t", target_column="target")


def test_returns_cross_reference_model(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    assert isinstance(xref, EDAQualityCrossReference)


def test_does_not_mutate_inputs(eda, report):
    eda_before = eda.model_dump_json()
    report_before = report.model_dump_json()
    cross_reference_eda_quality(eda, report)
    assert eda.model_dump_json() == eda_before
    assert report.model_dump_json() == report_before


def test_json_round_trip(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    dumped = xref.model_dump_json()
    assert EDAQualityCrossReference.model_validate_json(dumped).model_dump() == xref.model_dump()


def test_deterministic_and_sorted(eda, report):
    a = cross_reference_eda_quality(eda, report)
    b = cross_reference_eda_quality(eda, report)
    assert a.model_dump() == b.model_dump()
    keys = [(e.column or "", e.eda_signal.value, e.quality_finding_id) for e in a.entries]
    assert keys == sorted(keys)


def test_missing_values_entry(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    e = next(e for e in xref.entries if e.eda_signal is EDASignalKind.MISSINGNESS)
    assert e.column == "age"
    assert e.quality_finding_type is FindingType.MISSING_VALUES
    assert e.eda_evidence["eda_missing_count"] > 0
    assert "age" in e.relationship


def test_outlier_entry_uses_distribution_evidence(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    outliers = [e for e in xref.entries if e.eda_signal is EDASignalKind.DISPERSION]
    if any(f.finding_type is FindingType.POTENTIAL_OUTLIERS for f in report.findings):
        e = outliers[0]
        assert e.column == "amount"
        assert e.eda_evidence["eda_maximum"] >= e.eda_evidence["eda_q75"]


def test_class_imbalance_only_with_target(eda, messy):
    with_target = quality.analyze_dataframe(messy, target_column="target")
    xref = cross_reference_eda_quality(eda, with_target)
    imbalance = [e for e in xref.entries if e.eda_signal is EDASignalKind.CLASS_BALANCE]
    if any(f.finding_type is FindingType.CLASS_IMBALANCE for f in with_target.findings):
        assert imbalance and imbalance[0].column == "target"

    no_target = quality.analyze_dataframe(messy)
    xref2 = cross_reference_eda_quality(eda, no_target)
    assert not [e for e in xref2.entries if e.eda_signal is EDASignalKind.CLASS_BALANCE]


def test_no_matching_quality_info_returns_empty():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": ["x", "y", "x", "y", "x"]})
    eda = analyze_dataframe(df)
    report = quality.analyze_dataframe(df)
    xref = cross_reference_eda_quality(eda, report)
    assert xref.entries == []


def test_every_entry_references_a_real_finding(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    finding_ids = {f.finding_id for f in report.findings}
    for e in xref.entries:
        assert e.quality_finding_id in finding_ids


def test_relationship_text_is_templated_not_empty(eda, report):
    xref = cross_reference_eda_quality(eda, report)
    for e in xref.entries:
        assert e.relationship
        assert e.quality_finding_type.value in e.relationship


def test_analyze_dataframe_leaves_cross_reference_empty(messy):
    report = analyze_dataframe(messy)
    assert report.quality_cross_reference.entries == []


def test_duplicate_rows_finding_is_noted_not_entried():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "b": ["x", "x", "y", "z"]})
    eda = analyze_dataframe(df)
    report = quality.analyze_dataframe(df)
    xref = cross_reference_eda_quality(eda, report)
    if any(f.finding_type is FindingType.DUPLICATE_ROWS for f in report.findings):
        assert not any(e.quality_finding_type is FindingType.DUPLICATE_ROWS for e in xref.entries)
        assert any("duplicate" in n for n in xref.notes)
