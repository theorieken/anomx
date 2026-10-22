from __future__ import annotations

import io
import json
import queue
from urllib.error import URLError

import pytest

from anomx.agent import model_catalog, startup
from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AI_PROVIDERS, MODEL_MENU_OPTIONS, AnomxHome, provider_by_key


def test_curated_provider_defaults_match_both_model_menus():
    for provider in AI_PROVIDERS:
        if provider.key != "ollama":
            assert set(provider.models) == {
                item.model for item in MODEL_MENU_OPTIONS if item.provider_key == provider.key
            }
    assert provider_by_key("anthropic").models == (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    )


@pytest.mark.parametrize(
    "discovered, expected",
    [
        (None, ("current", "retired")),
        ((), ()),
        (("current", "uncurated"), ("current",)),
    ],
)
def test_availability_distinguishes_failure_from_empty_catalog(discovered, expected):
    assert model_catalog.available_curated_models(("current", "retired"), discovered) == expected


def test_discovery_handles_anthropic_pagination(monkeypatch):
    pages = iter(
        [
            {"data": [{"id": "claude-opus-5-5"}], "has_more": True, "last_id": "first/page"},
            {"data": [{"id": "claude-haiku-5-5"}], "has_more": False},
        ]
    )
    requests = []

    def fetch(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(next(pages)).encode())

    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", fetch)
    assert model_catalog.discover_provider_models("anthropic", "key") == (
        "claude-opus-5-5",
        "claude-haiku-5-5",
    )
    assert requests[1].full_url.endswith("limit=100&after_id=first%2Fpage")
    assert requests[1].get_header("X-api-key") == "key"


@pytest.mark.parametrize(
    "payload, expected", [({"data": []}, ()), ({"error": "unavailable"}, None)]
)
def test_desy_discovery_and_invalid_catalog(monkeypatch, payload, expected):
    def fetch(request, timeout):
        assert request.full_url == "https://assistant.desy.de/api/models"
        assert request.get_header("Authorization") == "Bearer key"
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", fetch)
    assert model_catalog.discover_provider_models("desy", "key") == expected


def test_discovery_failure_retains_defaults(monkeypatch):
    def fetch(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(model_catalog.urllib.request, "urlopen", fetch)
    assert model_catalog.discover_provider_models("blablador", "key") is None


def test_successful_discovery_filters_picker_without_exposing_uncurated_models(
    tmp_path, monkeypatch
):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("blablador", "key")
    app = AnomxCliApp(home=home, isolate_runtime=True)
    monkeypatch.setattr(
        "anomx.agent.app.discover_provider_models", lambda *args: ("alias-muse-spark", "embedding")
    )
    enriched = app._provider_with_discovered_models(provider_by_key("blablador"))
    assert enriched.models == ("alias-muse-spark",)
    assert [choice.value for choice in app._connected_model_menu_choices()] == [
        "blablador::alias-muse-spark"
    ]
    app._discovered_provider_models["blablador"] = ()
    assert app._connected_model_menu_choices() == []
    app.runtime.close()


def test_startup_worker_checks_version_and_each_connected_catalog(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("desy", "desy-key")
    home.set_api_key("blablador", "jsc-key")
    seen = {}

    def discover(provider, key):
        seen[provider] = key
        return ("current",)

    monkeypatch.setattr(model_catalog, "discover_provider_models", discover)
    monkeypatch.setattr(startup, "check_latest_version", lambda version: "9.9.9")
    results = queue.Queue()
    startup._run_startup_tasks(str(home.root), results, "0.2.86")
    assert seen == {"desy": "desy-key", "blablador": "jsc-key"}
    worker = startup.CliStartupProcess(home.root)
    worker._results = results
    worker.poll()
    assert worker.version_ready and worker.latest_version == "9.9.9"
    assert worker.provider_models == {"desy": ("current",), "blablador": ("current",)}
