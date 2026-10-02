"""Phase 12.3/12.4 — the planner -> executor -> critic loop under budget
(`ai_engine.agent.run_autonomous_experimentation`).

Fully testable with scripted fake `LLMProvider`s — no network, no API
key, no cost.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from data_engine.modeling import TrainingRunStatus
from data_engine.ingestion import RawDataStore, ingest_dataset

from ai_engine.agent import run_autonomous_experimentation
from ai_engine.context import build_analysis_context
from ai_engine.execution import ExecutionContext
from ai_engine.providers.base import LLMMessage, LLMProvider, LLMResponse


class ScriptedProvider(LLMProvider):
    """Returns one scripted response per call, in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        text = self.responses[self.call_count]
        self.call_count += 1
        return LLMResponse(text=text)


class RaisingProvider(LLMProvider):
    def complete(self, messages, **kwargs):
        raise RuntimeError("network down")


def _recommend(tool: str, parameters: dict | None = None) -> str:
    return json.dumps(
        {"recommendations": [{"tool": tool, "rationale": "r", "parameters": parameters or {}}]}
    )


@pytest.fixture
def dataset(tmp_path):
    csv_path = tmp_path / "d.csv"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=60), "x2": rng.normal(size=60)})
    df["y"] = df["x1"] * 2 + df["x2"] + rng.normal(size=60) * 0.1
    df.to_csv(csv_path, index=False)
    ref = ingest_dataset(csv_path, raw_store=RawDataStore(tmp_path / "raw"))
    return df, ref


def test_stops_naturally_when_no_more_recommendations(dataset):
    df, ref = dataset
    context = build_analysis_context(ref.dataset_id, objective="predict y")
    exec_ctx = ExecutionContext(df=df, reference=ref)

    provider = ScriptedProvider(
        [
            _recommend("analyze_quality"),
            _recommend("run_modeling_pipeline", {"objective": "predict y"}),
            json.dumps({"recommendations": []}),
        ]
    )
    trace = run_autonomous_experimentation(provider, context, exec_ctx, max_steps=5)

    assert trace.status is TrainingRunStatus.COMPLETED
    assert len(trace.steps) == 2
    assert "no further recommendations" in trace.stop_reason
    assert trace.steps[0].tool == "analyze_quality"
    assert trace.steps[1].tool == "run_modeling_pipeline"
    assert all(s.verdict.value == "continue" for s in trace.steps)


def test_stops_on_unavailable_step():
    context = build_analysis_context("ds1", objective="x")
    exec_ctx = ExecutionContext()  # no reference -> analyze_quality is unavailable
    provider = ScriptedProvider([_recommend("analyze_quality")])

    trace = run_autonomous_experimentation(provider, context, exec_ctx, max_steps=5)

    assert trace.status is TrainingRunStatus.COMPLETED  # unavailable, not a failure
    assert len(trace.steps) == 1
    assert trace.steps[0].verdict.value == "stop"
    assert "unavailable" in trace.stop_reason


def test_stops_on_failed_step():
    context = build_analysis_context("ds1", objective="x")
    # compare_experiments with an unknown id -> FAILED (not just unavailable)
    from experimentation import ExperimentStore

    exec_ctx = ExecutionContext(experiment_store=ExperimentStore("/tmp/does-not-exist-for-test"))
    provider = ScriptedProvider(
        [_recommend("compare_experiments", {"experiment_ids": ["missing-1", "missing-2"]})]
    )

    trace = run_autonomous_experimentation(provider, context, exec_ctx, max_steps=5)

    assert trace.status is TrainingRunStatus.FAILED
    assert len(trace.steps) == 1


def test_respects_max_steps_budget(dataset):
    df, ref = dataset
    context = build_analysis_context(ref.dataset_id, objective="predict y")
    exec_ctx = ExecutionContext(df=df, reference=ref)

    # analyze_quality always succeeds -> would run forever without a budget
    provider = ScriptedProvider([_recommend("analyze_quality") for _ in range(10)])
    trace = run_autonomous_experimentation(provider, context, exec_ctx, max_steps=3)

    assert trace.status is TrainingRunStatus.COMPLETED
    assert len(trace.steps) == 3
    assert "max_steps budget (3)" in trace.stop_reason


def test_recommendation_call_failure_returns_failed_trace():
    context = build_analysis_context("ds1", objective="x")
    trace = run_autonomous_experimentation(
        RaisingProvider(), context, ExecutionContext(), max_steps=3
    )
    assert trace.status is TrainingRunStatus.FAILED
    assert trace.steps == []
    assert "network down" in trace.stop_reason


def test_each_step_result_is_folded_into_next_context(dataset):
    df, ref = dataset
    context = build_analysis_context(ref.dataset_id, objective="predict y")
    exec_ctx = ExecutionContext(df=df, reference=ref)
    provider = ScriptedProvider(
        [_recommend("analyze_quality"), json.dumps({"recommendations": []})]
    )
    run_autonomous_experimentation(provider, context, exec_ctx, max_steps=5)

    # the second recommend_next_steps call should see step 1's result in its context
    second_call_user_message = provider.calls[1][-1].content
    assert "step_1_analyze_quality" in second_call_user_message


def test_trace_has_unique_run_id_per_call(dataset):
    df, ref = dataset
    context = build_analysis_context(ref.dataset_id, objective="predict y")
    exec_ctx = ExecutionContext(df=df, reference=ref)
    provider_a = ScriptedProvider([json.dumps({"recommendations": []})])
    provider_b = ScriptedProvider([json.dumps({"recommendations": []})])
    trace_a = run_autonomous_experimentation(provider_a, context, exec_ctx, max_steps=1)
    trace_b = run_autonomous_experimentation(provider_b, context, exec_ctx, max_steps=1)
    assert trace_a.run_id != trace_b.run_id


def test_trace_is_json_roundtrippable(dataset):
    from ai_engine.agent import AutonomousRunTrace

    df, ref = dataset
    context = build_analysis_context(ref.dataset_id, objective="predict y")
    exec_ctx = ExecutionContext(df=df, reference=ref)
    provider = ScriptedProvider(
        [_recommend("analyze_quality"), json.dumps({"recommendations": []})]
    )
    trace = run_autonomous_experimentation(provider, context, exec_ctx, max_steps=5)
    restored = AutonomousRunTrace.model_validate_json(trace.model_dump_json())
    assert restored == trace
