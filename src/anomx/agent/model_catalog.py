"""Model discovery for supported Anomx agent providers."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from typing import Any

MODEL_DISCOVERY_TIMEOUT_SECONDS = 8
OPENAI_MODELS_ENDPOINT = "https://api.openai.com/v1/models"
ANTHROPIC_MODELS_ENDPOINT = "https://api.anthropic.com/v1/models?limit=100"
DESY_MODELS_ENDPOINT = "https://assistant.desy.de/api/models"
BLABLADOR_MODELS_ENDPOINT = "https://api.blablador.fz-juelich.de/v1/models"
KIMI_MODELS_ENDPOINT = "https://api.moonshot.ai/v1/models"
OLLAMA_MODELS_ENDPOINT = "http://127.0.0.1:11434/api/tags"


def discover_provider_models(
    provider_key: str,
    api_key: str | None = None,
) -> tuple[str, ...] | None:
    """Return model identifiers currently advertised by a provider.

    None means discovery failed; an empty tuple means the provider advertised no
    models. Only failed discovery should fall back to the curated defaults.
    """

    normalized_provider = provider_key.strip().lower()
    request = _model_request(normalized_provider, api_key)
    if request is None:
        return None
    models: list[str] = []
    cursors: set[str] = set()
    try:
        while True:
            with urllib.request.urlopen(
                request, timeout=MODEL_DISCOVERY_TIMEOUT_SECONDS
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            entry_key = "models" if normalized_provider == "ollama" else "data"
            if not isinstance(payload, dict) or not isinstance(payload.get(entry_key), list):
                return None
            models.extend(_model_ids(normalized_provider, payload))
            if normalized_provider != "anthropic" or not payload.get("has_more"):
                return tuple(dict.fromkeys(models))
            cursor = str(payload.get("last_id") or "")
            if not cursor or cursor in cursors:
                return None
            cursors.add(cursor)
            request = urllib.request.Request(
                ANTHROPIC_MODELS_ENDPOINT + "&" + urllib.parse.urlencode({"after_id": cursor}),
                headers=dict(request.header_items()),
                method="GET",
            )
    except (
        OSError,
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
        UnicodeError,
    ):
        return None


def available_curated_models(
    curated_models: Iterable[str],
    discovered_models: Iterable[str] | None,
) -> tuple[str, ...]:
    """Keep curated ordering, excluding IDs absent from a successful discovery."""

    available = set(discovered_models) if discovered_models is not None else None
    return tuple(
        dict.fromkeys(model for model in curated_models if available is None or model in available)
    )


def _model_request(provider_key: str, api_key: str | None) -> urllib.request.Request | None:
    if provider_key == "ollama":
        return urllib.request.Request(OLLAMA_MODELS_ENDPOINT, method="GET")

    token = str(api_key or "").strip()
    if not token:
        return None
    if provider_key == "openai":
        return urllib.request.Request(
            OPENAI_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
    if provider_key == "anthropic":
        return urllib.request.Request(
            ANTHROPIC_MODELS_ENDPOINT,
            headers={
                "anthropic-version": "2023-06-01",
                "x-api-key": token,
            },
            method="GET",
        )
    if provider_key in {"blablador", "kimi", "desy"}:
        endpoint = {
            "blablador": BLABLADOR_MODELS_ENDPOINT,
            "kimi": KIMI_MODELS_ENDPOINT,
            "desy": DESY_MODELS_ENDPOINT,
        }[provider_key]
        return urllib.request.Request(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
    return None


def _model_ids(provider_key: str, payload: dict[str, Any]) -> tuple[str, ...]:
    entries = payload.get("models") if provider_key == "ollama" else payload.get("data")
    if not isinstance(entries, list):
        return ()

    models: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("name") if provider_key == "ollama" else entry.get("id")
        model = str(candidate or "").strip()
        if model and model not in models:
            models.append(model)
    return tuple(models)
