"""DESY Assistant backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

from anomx.agent.backends.anthropic import AnthropicCompatibleBackend
from anomx.agent.base.backends import (
    AnthropicStreamResponse,
    BackendCallbacks,
    BackendTextCallback,
)

DESY_MESSAGES_ENDPOINT = "https://assistant.desy.de/api/v1/messages"


class DesyAssistantBackend(AnthropicCompatibleBackend):
    """DESY Assistant Messages API backend."""

    provider_key = "desy"
    provider_label = "DESY Assistant"
    env_var = "DESY_ASSISTANT_API_KEY"

    def generate(
        self,
        session_path: Path,
        model: str,
        callbacks: BackendCallbacks,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        del thinking_intensity
        return self._messages_api_response(
            session_path,
            model,
            callbacks,
            include_thinking=False,
        )

    def _stream_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: BackendTextCallback | None,
        status_callback: BackendTextCallback | None,
    ) -> AnthropicStreamResponse | str:
        return self._stream_anthropic_compatible_response(
            DESY_MESSAGES_ENDPOINT,
            {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            api_key,
            payload,
            delta_callback,
            status_callback,
        )

    def suggest_session_title(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            DESY_MESSAGES_ENDPOINT,
            data=json.dumps(
                {
                    "model": model,
                    "system": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "messages": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_tokens": 24,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_anthropic_text(data))

    def suggest_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            DESY_MESSAGES_ENDPOINT,
            data=json.dumps(
                {
                    "model": model,
                    "system": self._project_name_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_project_name(self.extract_anthropic_text(data))

    def suggest_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        api_key = self._api_key(self.provider_key, self.env_var)
        if api_key is None:
            return None

        request = urllib.request.Request(
            DESY_MESSAGES_ENDPOINT,
            data=json.dumps(
                {
                    "model": model,
                    "system": self._continuation_system_prompt(),
                    "messages": [{"role": "user", "content": self._title_prompt(messages)}],
                    "max_tokens": 48,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_anthropic_text(data))
