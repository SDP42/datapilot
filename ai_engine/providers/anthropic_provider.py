"""Phase 11.3 — the Anthropic concrete `LLMProvider`.

:class:`AnthropicProvider` is the first concrete implementation of the
Phase-0 :class:`~ai_engine.providers.base.LLMProvider` interface (Phase
0's decision 0004: "define the abstraction now; implement no concrete
provider — Phase 11 adds concrete providers behind this interface").

The `anthropic` SDK is imported lazily (via
:mod:`ai_engine.providers.availability`), exactly like every other
optional-dependency boundary in this codebase (`torch`, `mlflow`,
`shap`) — constructing an `AnthropicProvider` without the package
installed raises `RuntimeError` naming the missing extra, never an
opaque `ImportError` from deep inside the SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from .availability import anthropic_availability
from .base import LLMMessage, LLMProvider, LLMResponse

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    """An `LLMProvider` backed by the Anthropic Messages API.

    Translates the provider-agnostic `LLMMessage` list (a `system` role
    entry, if present, becomes the API's top-level `system` parameter —
    Anthropic's Messages API does not accept a `system`-role message
    inside the `messages` array itself; every other message is passed
    through as `{"role": ..., "content": ...}`) and wraps the response's
    text content blocks into one `LLMResponse.text`. `LLMResponse.raw`
    holds the SDK's own response object, for a caller that needs more
    than the plain text (e.g. token usage) — never parsed or
    re-interpreted by this class itself.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        _import: Callable[[str], ModuleType] = import_module,
    ) -> None:
        availability = anthropic_availability(_import=_import)
        if not availability.available:
            raise RuntimeError(availability.reason)

        anthropic = _import("anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        """Send `messages` to the Anthropic Messages API and return the text response.

        `kwargs` accepts `max_tokens` (default `DEFAULT_MAX_TOKENS`) and
        `model` (overrides the instance default for this one call) —
        never silently coerces an unrecognised keyword; anything else is
        passed straight through to the SDK's own `messages.create`, so an
        invalid keyword surfaces as the SDK's own error rather than being
        swallowed here.
        """
        system = next((m.content for m in messages if m.role == "system"), None)
        conversation = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]

        call_kwargs: dict[str, object] = dict(kwargs)
        model = call_kwargs.pop("model", self._model)
        max_tokens = call_kwargs.pop("max_tokens", DEFAULT_MAX_TOKENS)
        if system is not None:
            call_kwargs["system"] = system

        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=conversation,
            **call_kwargs,
        )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return LLMResponse(text=text, raw=response)


__all__ = ["DEFAULT_ANTHROPIC_MODEL", "DEFAULT_MAX_TOKENS", "AnthropicProvider"]
