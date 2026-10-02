"""Phase 11.4 — LLM-backed interpretation and recommendations
(`ai_engine.analysis`).

Fully testable with an in-memory fake `LLMProvider` — no network, no API
key, no cost.
"""

from __future__ import annotations

import json

from data_engine.modeling import ModelingSpec, TrainingRunStatus
from ai_engine.analysis import interpret_results, recommend_next_steps
from ai_engine.context import build_analysis_context
from ai_engine.providers.base import LLMMessage, LLMProvider, LLMResponse
from ai_engine.tools import ToolSchema


class FakeProvider(LLMProvider):
    def __init__(self, text: str | None = None, *, raises: Exception | None = None):
        self.text = text
        self.raises = raises
        self.last_messages: list[LLMMessage] | None = None

    def complete(self, messages, **kwargs):
        self.last_messages = messages
        if self.raises is not None:
            raise self.raises
        return LLMResponse(text=self.text or "")


def _context():
    spec = ModelingSpec(dataset_id="ds1", objective_provided=False)
    return build_analysis_context("ds1", objective="predict y", modeling=spec)


# --- interpret_results -------------------------------------------------


def test_interpret_results_returns_completed_with_text():
    provider = FakeProvider("The model performs well.")
    result = interpret_results(provider, _context())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.analysis_text == "The model performs well."


def test_interpret_results_sends_system_and_user_messages():
    provider = FakeProvider("ok")
    interpret_results(provider, _context())
    roles = [m.role for m in provider.last_messages]
    assert roles == ["system", "user"]


def test_interpret_results_provider_failure_returns_failed():
    provider = FakeProvider(raises=RuntimeError("network error"))
    result = interpret_results(provider, _context())
    assert result.status is TrainingRunStatus.FAILED
    assert "network error" in (result.reason or "")
    assert result.analysis_text is None


def test_interpret_results_custom_system_prompt_is_used():
    provider = FakeProvider("ok")
    interpret_results(provider, _context(), system_prompt="Be terse.")
    assert provider.last_messages[0].content == "Be terse."


# --- recommend_next_steps -----------------------------------------------


def test_recommend_next_steps_valid_response():
    response_json = json.dumps(
        {
            "recommendations": [
                {
                    "tool": "analyze_quality",
                    "rationale": "check for issues",
                    "parameters": {"dataset_version_id": "ds1:raw"},
                }
            ]
        }
    )
    provider = FakeProvider(response_json)
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.COMPLETED
    assert len(result.recommendations) == 1
    assert result.recommendations[0].tool == "analyze_quality"
    assert result.dropped == []


def test_recommend_next_steps_drops_unknown_tool():
    response_json = json.dumps(
        {"recommendations": [{"tool": "fabricated_tool", "rationale": "x", "parameters": {}}]}
    )
    provider = FakeProvider(response_json)
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.recommendations == []
    assert len(result.dropped) == 1
    assert "fabricated_tool" in result.dropped[0]


def test_recommend_next_steps_drops_malformed_entry():
    response_json = json.dumps({"recommendations": ["not a dict"]})
    provider = FakeProvider(response_json)
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.recommendations == []
    assert len(result.dropped) == 1


def test_recommend_next_steps_unparseable_json_returns_failed():
    provider = FakeProvider("not json at all")
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.FAILED
    assert result.raw_response == "not json at all"
    assert result.recommendations == []


def test_recommend_next_steps_wrong_top_level_shape_returns_failed():
    provider = FakeProvider(json.dumps({"something_else": []}))
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.FAILED


def test_recommend_next_steps_non_list_recommendations_returns_failed():
    provider = FakeProvider(json.dumps({"recommendations": "not a list"}))
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.FAILED


def test_recommend_next_steps_provider_failure_returns_failed():
    provider = FakeProvider(raises=RuntimeError("api error"))
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.FAILED
    assert "api error" in (result.reason or "")


def test_recommend_next_steps_restricts_to_supplied_tools():
    custom_tool = ToolSchema(name="custom_tool", description="d", parameters={"type": "object"})
    response_json = json.dumps(
        {"recommendations": [{"tool": "analyze_quality", "rationale": "x", "parameters": {}}]}
    )
    provider = FakeProvider(response_json)
    result = recommend_next_steps(provider, _context(), tools=[custom_tool])
    # analyze_quality isn't in the restricted tool list -> dropped
    assert result.recommendations == []
    assert len(result.dropped) == 1


def test_recommend_next_steps_zero_recommendations_is_still_completed():
    provider = FakeProvider(json.dumps({"recommendations": []}))
    result = recommend_next_steps(provider, _context())
    assert result.status is TrainingRunStatus.COMPLETED
    assert result.recommendations == []
    assert result.dropped == []


def test_result_is_json_roundtrippable():
    from ai_engine.analysis import RecommendationResult

    response_json = json.dumps(
        {"recommendations": [{"tool": "analyze_quality", "rationale": "x", "parameters": {}}]}
    )
    provider = FakeProvider(response_json)
    result = recommend_next_steps(provider, _context())
    restored = RecommendationResult.model_validate_json(result.model_dump_json())
    assert restored == result
