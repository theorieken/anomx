from __future__ import annotations

from anomx.agent.app import AnomxCliApp
from anomx.agent.backends.anthropic import AnthropicBackend
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome, model_detail, thinking_intensity_options


def test_model_menu_uses_only_curated_models_with_specific_labels(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    for provider_key in ("openai", "anthropic", "desy", "blablador", "kimi"):
        home.set_api_key(provider_key, "test-key")
    home.set_backend_connected("ollama", True)
    app = AnomxCliApp(home=home, isolate_runtime=True)

    monkeypatch.setattr(
        app,
        "_provider_with_discovered_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the /model menu must not use provider discovery")
        ),
    )

    choices = app._connected_model_menu_choices()

    assert [(choice.label, choice.value) for choice in choices] == [
        ("5.6 Sol", "openai::gpt-5.6-sol"),
        ("5.6 Terra", "openai::gpt-5.6-terra"),
        ("5.6 Luna", "openai::gpt-5.6-luna"),
        ("5.5", "openai::gpt-5.5"),
        ("5.4", "openai::gpt-5.4"),
        ("5.4 Mini", "openai::gpt-5.4-mini"),
        ("Fable 5.1", "anthropic::claude-fable-5-1"),
        ("Opus 5", "anthropic::claude-opus-5"),
        ("Sonnet 5", "anthropic::claude-sonnet-5"),
        ("Haiku 4.5", "anthropic::claude-haiku-4-5-20251001"),
        ("Coding", "desy::coding"),
        ("Assistant", "desy::desy-assistant"),
        ("Reasoning", "desy::reasoning"),
        ("GPT OSS (120B)", "blablador::alias-fast"),
        ("Qwen 3.5 (122B)", "blablador::alias-large"),
        ("Qwen 3.5 (35B)", "blablador::alias-code"),
        ("MiniMax M2.5", "blablador::alias-huge"),
    ]
    assert all(not choice.label.startswith("Custom ") for choice in choices)


def test_direct_model_override_remains_available_outside_curated_menu(tmp_path):
    home = AnomxHome(tmp_path / "home")
    app = AnomxCliApp(
        home=home,
        startup_provider="openai",
        startup_model="gpt-experimental",
        isolate_runtime=True,
    )

    app.prepare_startup_config()

    config = home.load_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-experimental"


def test_curated_claude_5_models_use_adaptive_thinking_and_all_effort_levels(tmp_path):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    backend = AnthropicBackend(runtime)

    for model in ("claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"):
        assert backend._anthropic_thinking_config(model) == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert [
            option.value for option in thinking_intensity_options("anthropic", model)
        ] == ["auto", "low", "medium", "high", "xhigh", "max"]


def test_curated_models_have_context_and_output_metadata():
    for model in (
        "coding",
        "desy-assistant",
        "reasoning",
        "alias-fast",
        "alias-large",
        "alias-code",
        "alias-huge",
    ):
        detail = model_detail(model)
        assert "context" in detail
        assert "max output" in detail
        assert "custom context" not in detail


def test_manage_settings_is_last_config_item_with_background_work_defaults(tmp_path):
    app = AnomxCliApp(home=AnomxHome(tmp_path / "home"), isolate_runtime=True)

    assert app._config_menu_choices()[-1].label == "Manage Settings"
    assert [(choice.label, choice.detail) for choice in app._background_work_setting_choices()] == [
        ("Hard Work: Selection", "e.g. computation of recommended next steps"),
        ("Medium Work: Selection", "e.g. risk assessment of agent commands"),
        ("Easy Work: Selection", "e.g. automatic naming of chats"),
    ]


def test_manage_settings_can_select_model_and_restore_current_selection(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    app = AnomxCliApp(home=home, isolate_runtime=True)
    selections = iter(
        [
            "background_hard_work_model",
            "openai::gpt-5.4",
            "background_easy_work_model",
            "current",
            None,
        ]
    )
    seen_choices: list[list[tuple[str, str]]] = []

    def select_menu(_stdscr, _title, _subtitle, choices, **_kwargs):
        seen_choices.append([(choice.label, choice.detail) for choice in choices])
        return next(selections)

    monkeypatch.setattr(app, "_menu", select_menu)

    app._run_manage_settings_panel(None)

    config = home.load_config()
    assert config["background_hard_work_model"] == "openai::gpt-5.4"
    assert config["background_easy_work_model"] == "current"
    assert seen_choices[1][0] == (
        "Current Model",
        "Use the currently selected model for background work",
    )
    assert app._background_work_setting_choices()[2].label == "Easy Work: Selection"


def test_background_work_runtime_uses_current_or_configured_model(
    tmp_path,
    monkeypatch,
):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    runtime = AgentRuntime(home, tmp_path)
    backend = object()
    monkeypatch.setattr(
        "anomx.agent.runtime.backend_for_provider",
        lambda provider, _runtime: backend if provider == "openai" else None,
    )

    assert runtime._background_work_backend("background_easy_work_model") == (
        backend,
        "gpt-5.5",
    )

    config = home.load_config()
    config["background_easy_work_model"] = "openai::gpt-5.6-luna"
    home.save_config(config)

    assert runtime._background_work_backend("background_easy_work_model") == (
        backend,
        "gpt-5.6-luna",
    )
