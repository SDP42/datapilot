"""LLM provider abstraction.

DataPilot is not built around a single LLM vendor. Every part of the
system that needs language-model reasoning depends on this interface,
never on a concrete SDK. Concrete providers (Anthropic, OpenAI, local
models) are implemented in Phase 11.

The interface is deliberately narrow: the AI engine sends structured
context and receives text / structured recommendations. It never gives
the model direct access to datasets or execution state — see
docs/architecture-principles.md.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: object | None = None


class LLMProvider(abc.ABC):
    """Minimal contract every concrete provider must satisfy."""

    @abc.abstractmethod
    def complete(self, messages: list[LLMMessage], **kwargs: object) -> LLMResponse:
        """Return a completion for the given message list."""
        raise NotImplementedError
