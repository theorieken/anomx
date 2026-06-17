"""Ollama backend."""

from __future__ import annotations

from pathlib import Path


class OllamaBackend:
    """Local Ollama chat backend."""

    provider_key = "ollama"

    def generate(
        self,
        runtime: object,
        session_path: Path,
        model: str,
        callbacks: object,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        del thinking_intensity
        return runtime._ollama_response(session_path, model, callbacks)
