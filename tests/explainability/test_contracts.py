"""Phase 10.1 — the Explainability foundation contract."""

from __future__ import annotations

from explainability.contracts import (
    ExplainabilityStatus,
    ExplanationReport,
    ExplanationRequest,
    understand_explanation,
)


def test_foundation_report_is_all_not_yet_inferred():
    request = ExplanationRequest(dataset_id="ds1", objective="predict y")
    report = understand_explanation(request)
    assert report.status is ExplainabilityStatus.NOT_YET_INFERRED
    assert report.feature_importance.status is ExplainabilityStatus.NOT_YET_INFERRED
    assert report.shap_importance.status is ExplainabilityStatus.NOT_YET_INFERRED
    assert report.partial_dependence == []


def test_foundation_echoes_request_fields():
    request = ExplanationRequest(dataset_id="ds1", dataset_version_id="ds1:raw", objective="x")
    report = understand_explanation(request)
    assert report.dataset_id == "ds1"
    assert report.dataset_version_id == "ds1:raw"
    assert report.objective == "x"


def test_repeated_calls_are_byte_identical():
    request = ExplanationRequest(dataset_id="ds1")
    a = understand_explanation(request)
    b = understand_explanation(request)
    assert a.model_dump_json() == b.model_dump_json()


def test_report_is_json_roundtrippable():
    report = understand_explanation(ExplanationRequest(dataset_id="ds1"))
    restored = ExplanationReport.model_validate_json(report.model_dump_json())
    assert restored == report
