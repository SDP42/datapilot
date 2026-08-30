"""Data-quality engine: one focused test per check + report guarantees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_engine.quality import (
    FindingType,
    QualityReport,
    Severity,
    analyze_dataframe,
    available_checks,
)


def test_missing_value_detection_and_severity():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "few_missing": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],  # 10% -> MEDIUM
            "mostly_missing": [1] + [None] * 9,  # 90% -> CRITICAL
        }
    )
    report = analyze_dataframe(df)
    missing = {
        f.columns[0]: f for f in report.findings if f.finding_type is FindingType.MISSING_VALUES
    }

    assert "few_missing" in missing and "mostly_missing" in missing
    assert missing["few_missing"].severity is Severity.MEDIUM
    assert missing["few_missing"].affected_rows == 1
    assert missing["mostly_missing"].severity is Severity.CRITICAL
    assert report.summary.has_critical is True


def test_duplicate_row_detection():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    report = analyze_dataframe(df)
    dupes = [f for f in report.findings if f.finding_type is FindingType.DUPLICATE_ROWS]
    assert len(dupes) == 1
    assert dupes[0].affected_rows == 1
    assert dupes[0].columns == []


def test_numeric_stored_as_text_detection():
    df = pd.DataFrame({"amount": [str(x) for x in range(1, 31)], "name": list("abcdefghij") * 3})
    report = analyze_dataframe(df)
    type_findings = [
        f for f in report.findings if f.finding_type is FindingType.POTENTIAL_TYPE_MISMATCH
    ]
    assert [f.columns[0] for f in type_findings] == ["amount"]
    assert type_findings[0].observed["looks_like"] == "numeric"
    assert type_findings[0].confidence == pytest.approx(1.0)


def test_datetime_stored_as_text_detection():
    df = pd.DataFrame({"when": pd.Series([f"2021-01-{d:02d}" for d in range(1, 21)])})
    report = analyze_dataframe(df)
    type_findings = [
        f for f in report.findings if f.finding_type is FindingType.POTENTIAL_TYPE_MISMATCH
    ]
    assert type_findings and type_findings[0].observed["looks_like"] == "datetime"


def test_inconsistent_categorical_values_detection():
    df = pd.DataFrame({"gender": (["Male", "male", "MALE", " male "] * 5) + ["Female"] * 5})
    report = analyze_dataframe(df)
    cat = [f for f in report.findings if f.finding_type is FindingType.INCONSISTENT_CATEGORIES]
    assert cat and cat[0].columns == ["gender"]
    variants = cat[0].observed["variant_groups"]
    assert any(len(v) >= 3 for v in variants.values())
    # analysis only — the column is untouched
    assert set(df["gender"].unique()) >= {"Male", "male", "MALE", " male "}


def test_outlier_detection_iqr():
    values = list(range(1, 101)) + [100_000]  # one extreme high value
    df = pd.DataFrame({"income": values})
    report = analyze_dataframe(df)
    out = [f for f in report.findings if f.finding_type is FindingType.POTENTIAL_OUTLIERS]
    assert out and out[0].columns == ["income"]
    assert out[0].observed["method"] == "iqr"
    assert out[0].affected_rows >= 1
    assert 100_000 in df["income"].values  # not removed


def test_skewness_detection():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"skewed": np.concatenate([rng.exponential(1.0, 500), [50, 60, 70]])})
    report = analyze_dataframe(df)
    skew = [f for f in report.findings if f.finding_type is FindingType.HIGH_SKEW]
    assert skew and skew[0].columns == ["skewed"]
    assert abs(skew[0].observed["skewness"]) >= 1.0


def test_class_imbalance_when_target_provided():
    df = pd.DataFrame({"x": range(100), "label": [0] * 95 + [1] * 5})
    report = analyze_dataframe(df, target_column="label")
    imb = [f for f in report.findings if f.finding_type is FindingType.CLASS_IMBALANCE]
    assert imb and imb[0].columns == ["label"]
    assert imb[0].severity in (Severity.MEDIUM, Severity.HIGH)
    assert report.target_column == "label"


def test_no_class_imbalance_check_without_target():
    df = pd.DataFrame({"x": range(100), "label": [0] * 95 + [1] * 5})
    report = analyze_dataframe(df)
    assert not any(f.finding_type is FindingType.CLASS_IMBALANCE for f in report.findings)
    assert report.target_column is None


def test_unknown_target_column_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not a column"):
        analyze_dataframe(df, target_column="nope")


def test_report_is_json_serialisable_and_round_trips():
    df = pd.DataFrame({"a": [1, None, 3, 3], "g": ["x", "X", "y", "y"]})
    report = analyze_dataframe(df, dataset_id="ds-json")
    payload = report.model_dump_json()
    restored = QualityReport.model_validate_json(payload)
    assert restored.dataset_id == "ds-json"
    assert restored.summary.total_findings == report.summary.total_findings
    assert 0.0 <= restored.summary.score <= 100.0


def test_findings_sorted_most_severe_first():
    df = pd.DataFrame(
        {
            "mostly_missing": [1] + [None] * 9,
            "few_missing": [1, None] + list(range(3, 11)),
        }
    )
    report = analyze_dataframe(df)
    severities = [f.severity for f in report.findings]
    order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
    assert severities == sorted(severities, key=lambda s: order[s], reverse=True)


def test_analysis_does_not_mutate_input_dataframe():
    df = pd.DataFrame(
        {
            "num_text": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "gender": ["M", "m", "M", "m", "M", "m", "M", "m"],
            "val": [1, 2, 3, 4, 5, 6, 7, 900],
        }
    )
    before = df.copy(deep=True)
    analyze_dataframe(df, target_column="gender")
    pd.testing.assert_frame_equal(df, before)


def test_can_run_a_single_check():
    df = pd.DataFrame({"a": [1, None, 3]})
    report = analyze_dataframe(df, checks=["missing_values"])
    assert {f.finding_type for f in report.findings} == {FindingType.MISSING_VALUES}
    assert "missing_values" in available_checks()
