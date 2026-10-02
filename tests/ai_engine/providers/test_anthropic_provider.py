"""Phase 11.3 — the Anthropic concrete `LLMProvider`
(`ai_engine.providers.anthropic_provider.AnthropicProvider`).

Environment-independent tests throughout: construction failure uses the
injectable `_import` seam (no real package needed), and message
translation is verified against a hand-built fake `anthropic` module (no
real network call, no API key, no cost) rather than `pytest.importorskip`
+ a real request — unlike `torch` / `shap`, calling the real Anthropic
API is not free, so this is the one optional-dependency boundary in this
codebase that is never exercised end-to-end in tests, by design.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_engine.providers.anthropic_provider import AnthropicProvider
from ai_engine.providers.base import LLMMessage


def _raise_import_error(name: str):
    raise ImportError(f"No module named '{name}'")


def test_construction_raises_runtime_error_when_sdk_missing():
    with pytest.raises(RuntimeError, match="anthropic"):
        AnthropicProvider(_import=_raise_import_error)


class _FakeMessages:
    def __init__(self):
        self.captured: dict[str, object] = {}

    def create(self, **kwargs):
        self.captured = kwargs
        block = SimpleNamespace(type="text", text="fake response text")
        return SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.messages = _FakeMessages()


def _fake_anthropic_module():
    module = SimpleNamespace(__version__="0.99.0", Anthropic=_FakeClient)
    return module


def _fake_import(name: str):
    if name == "anthropic":
        return _fake_anthropic_module()
    raise ImportError(name)


def test_complete_extracts_system_message_separately():
    provider = AnthropicProvider(api_key="test-key", _import=_fake_import)
    messages = [
        LLMMessage(role="system", content="Be helpful."),
        LLMMessage(role="user", content="Hi"),
    ]
    response = provider.complete(messages)
    assert response.text == "fake response text"
    captured = provider._client.messages.captured
    assert captured["system"] == "Be helpful."
    assert captured["messages"] == [{"role": "user", "content": "Hi"}]


def test_complete_without_system_message_omits_system_kwarg():
    provider = AnthropicProvider(_import=_fake_import)
    provider.complete([LLMMessage(role="user", content="Hi")])
    captured = provider._client.messages.captured
    assert "system" not in captured


def test_complete_uses_default_model_and_max_tokens():
    from ai_engine.providers.anthropic_provider import DEFAULT_ANTHROPIC_MODEL, DEFAULT_MAX_TOKENS

    provider = AnthropicProvider(_import=_fake_import)
    provider.complete([LLMMessage(role="user", content="Hi")])
    captured = provider._client.messages.captured
    assert captured["model"] == DEFAULT_ANTHROPIC_MODEL
    assert captured["max_tokens"] == DEFAULT_MAX_TOKENS


def test_complete_allows_per_call_model_and_max_tokens_override():
    provider = AnthropicProvider(_import=_fake_import)
    provider.complete([LLMMessage(role="user", content="Hi")], model="claude-other", max_tokens=50)
    captured = provider._client.messages.captured
    assert captured["model"] == "claude-other"
    assert captured["max_tokens"] == 50


def test_response_raw_holds_the_sdk_response_object():
    provider = AnthropicProvider(_import=_fake_import)
    response = provider.complete([LLMMessage(role="user", content="Hi")])
    assert response.raw is not None
    assert response.raw.content[0].text == "fake response text"
