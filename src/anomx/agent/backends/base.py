"""AI backend abstractions."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class AIBackend(Protocol):
    """Model backend capable of generating one agent response."""

    provider_key: str

    def generate(
        self,
        runtime: object,
        session_path: Path,
        model: str,
        callbacks: object,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a model response through this backend."""
