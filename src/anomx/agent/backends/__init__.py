"""AI backend registry."""

from __future__ import annotations

from collections.abc import Callable

from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend
from anomx.agent.base.backends import BaseBackend

BackendFactory = Callable[[object], BaseBackend]

BACKENDS: dict[str, BackendFactory] = {
    OpenAIBackend.provider_key: OpenAIBackend,
    AnthropicBackend.provider_key: AnthropicBackend,
    DesyAssistantBackend.provider_key: DesyAssistantBackend,
    OllamaBackend.provider_key: OllamaBackend,
}


def backend_for_provider(provider_key: str, runtime: object) -> BaseBackend | None:
    """Return a runtime-bound backend for a provider key."""

    factory = BACKENDS.get(provider_key)
    return None if factory is None else factory(runtime)


__all__ = [
    "AnthropicBackend",
    "BACKENDS",
    "BaseBackend",
    "DesyAssistantBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "backend_for_provider",
]
