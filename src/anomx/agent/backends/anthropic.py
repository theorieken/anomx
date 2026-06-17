"""Anthropic backend."""

from __future__ import annotations

from pathlib import Path


class AnthropicBackend:
    """Anthropic Messages API backend."""

    provider_key = "anthropic"

    def generate(
        self,
        runtime: object,
        session_path: Path,
        model: str,
        callbacks: object,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        return runtime._anthropic_response(
            session_path,
            model,
            callbacks,
            thinking_intensity=thinking_intensity,
        )
