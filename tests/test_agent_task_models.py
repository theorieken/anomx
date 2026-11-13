from anomx.agent.app import AnomxCliApp
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome, SPECIFIC_TASK_MODEL_SETTINGS


def test_task_model_and_effort_persist_and_menu_can_restore_standard(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    app = AnomxCliApp(home=home, isolate_runtime=True)
    selections = iter(("openai::gpt-6.1-sol", "high"))
    monkeypatch.setattr(app, "_menu", lambda *_a, **_kw: next(selections))
    app._change_agent_setting(None, "task_context_compression_model")
    config = home.load_config()
    assert config["task_context_compression_model"] == "openai::gpt-6.1-sol"
    assert config["task_context_compression_effort"] == "high"
    assert all(setting.config_key in config for setting in SPECIFIC_TASK_MODEL_SETTINGS)
    selections = iter(("standard",))
    app._change_agent_setting(None, "task_context_compression_model")
    config = home.load_config()
    assert config["task_context_compression_model"] == "standard"
    assert config["task_context_compression_effort"] == ""


def test_task_model_override_system_default_and_standard_fallback(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("openai", "test-key")
    runtime = AgentRuntime(home, tmp_path)
    backend = object()
    monkeypatch.setattr("anomx.agent.runtime.backend_for_provider", lambda *_: backend)
    config = home.load_config()
    config["background_easy_work_model"] = "openai::easy-model"
    config["background_medium_work_model"] = "openai::analysis-model"
    home.save_config(config)
    assert runtime._background_work_backend("task_context_compression_model") == (backend, "easy-model")
    assert runtime._background_work_backend("task_channel_analysis_model") == (backend, "analysis-model")
    config["task_system_tasks_model"] = "openai::system-model"
    home.save_config(config)
    assert runtime._background_work_backend("task_session_naming_model") == (backend, "system-model")
    config["task_session_naming_model"] = "openai::title-model"
    home.save_config(config)
    assert runtime._background_work_backend("task_session_naming_model") == (backend, "title-model")
    assert runtime._background_work_backend("task_context_compression_model") == (backend, "system-model")


def test_session_naming_uses_task_resolver(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="openai", model="chat-model")
    selections = []
    class Backend:
        def suggest_session_title(self, messages, model):
            assert model == "title-model"
            return "Task title"
    runtime = AgentRuntime(home, tmp_path, background_backend_resolver=lambda key, _runtime: selections.append(key) or (Backend(), "title-model"))
    monkeypatch.setattr(runtime, "conversation_messages", lambda _: [{"role": "user", "content": "Hello"}])
    assert runtime.suggest_session_title(session.path) == "Task title"
    assert selections == ["task_session_naming_model"]
