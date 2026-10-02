"""Phase 12.2 — the deterministic critic (`ai_engine.critic`)."""

from __future__ import annotations

from data_engine.modeling import TrainingRunStatus
from ai_engine.critic import StepVerdict, evaluate_step
from ai_engine.execution import ExecutionResult


def test_completed_result_continues():
    result = ExecutionResult(status=TrainingRunStatus.COMPLETED, tool="analyze_quality", output={})
    verdict, reason = evaluate_step(result)
    assert verdict is StepVerdict.CONTINUE
    assert "analyze_quality" in reason


def test_unavailable_result_stops():
    result = ExecutionResult(
        status=TrainingRunStatus.UNAVAILABLE, tool="x", reason="missing context"
    )
    verdict, reason = evaluate_step(result)
    assert verdict is StepVerdict.STOP
    assert "missing context" in reason


def test_failed_result_stops():
    result = ExecutionResult(status=TrainingRunStatus.FAILED, tool="x", reason="boom")
    verdict, reason = evaluate_step(result)
    assert verdict is StepVerdict.STOP
    assert "boom" in reason


def test_deterministic_for_same_input():
    result = ExecutionResult(status=TrainingRunStatus.COMPLETED, tool="x", output={})
    assert evaluate_step(result) == evaluate_step(result)
