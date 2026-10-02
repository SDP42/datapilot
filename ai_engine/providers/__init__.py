"""LLM provider abstraction layer.

:mod:`.base` defines the vendor-agnostic :class:`LLMProvider` interface
(Phase 0). :class:`~.anthropic_provider.AnthropicProvider` (Phase 11.3)
is the first concrete implementation; the ``anthropic`` SDK it wraps is
an optional dependency, detected lazily by :mod:`.availability` — never
imported at this package's own import time.
"""

from __future__ import annotations

from .anthropic_provider import DEFAULT_ANTHROPIC_MODEL, DEFAULT_MAX_TOKENS, AnthropicProvider
from .availability import AnthropicAvailability, anthropic_availability, is_anthropic_available
from .base import LLMMessage, LLMProvider, LLMResponse

__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_MAX_TOKENS",
    "AnthropicAvailability",
    "AnthropicProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "anthropic_availability",
    "is_anthropic_available",
]
