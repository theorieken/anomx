"""AI backend registry."""

from __future__ import annotations

from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.backends.base import AIBackend
from anomx.agent.backends.desy_assistant import DesyAssistantBackend
from anomx.agent.backends.ollama import OllamaBackend
from anomx.agent.backends.openai import OpenAIBackend

BACKENDS: dict[str, AIBackend] = {
    OpenAIBackend.provider_key: OpenAIBackend(),
    AnthropicBackend.provider_key: AnthropicBackend(),
    DesyAssistantBackend.provider_key: DesyAssistantBackend(),
    OllamaBackend.provider_key: OllamaBackend(),
}


def backend_for_provider(provider_key: str) -> AIBackend | None:
    """Return the backend registered for a provider key."""

    return BACKENDS.get(provider_key)


__all__ = [
    "AIBackend",
    "AnthropicBackend",
    "BACKENDS",
    "DesyAssistantBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "backend_for_provider",
]
