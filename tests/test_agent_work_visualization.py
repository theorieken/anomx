import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.base.tools import statement_property
from anomx.agent.store import AnomxHome
from anomx.agent.ui.models import MessageLine


@pytest.fixture
def chat(tmp_path):
    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="openai", model="gpt-5.5")
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    home.append_session_event(session.path, "user_message", {"message": "Inspect channels"})
    home.append_session_event(session.path, "work_message", {
        "message": "Reading channels", "role": "tool", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {
        "message": "I found the channel list.", "intermediate": True, "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "work_message", {
        "message": "Counting channels", "role": "tool", "turn_id": "turn-1",
    })
    return home, session, app


def test_work_visualization_defaults_persists_and_normalizes(tmp_path):
    home = AnomxHome(tmp_path / "home")
    assert home.load_config()["work_visualization"] == "default"
    home.save_config({"work_visualization": "extended"})
    assert AnomxHome(home.root).load_config()["work_visualization"] == "extended"
    assert AnomxCliApp(home=home, cwd=tmp_path).work_visualization == "extended"
    home.save_config({"work_visualization": "invalid"})
    assert home.load_config()["work_visualization"] == "default"
    home.config_path.write_text('work_visualization = "invalid"\n')
    assert home.load_config()["work_visualization"] == "default"


def test_default_shows_latest_activity_and_expands_all_updates(chat):
    home, session, app = chat
    messages = app._read_message_lines(session.path)
    assert messages == [
        MessageLine("user", "Inspect channels"),
        MessageLine("work_active", "Counting channels · expand", "turn-1"),
    ]
    assert app._messages_with_working_status(messages, "Thinking") == messages
    app._toggle_work_turn("turn-1")
    expanded = app._read_message_lines(session.path)
    assert [line.text for line in expanded] == [
        "Inspect channels", "Reading channels", "I found the channel list.",
        "Counting channels", "Counting channels · collapse",
    ]
    home.append_session_event(session.path, "work_message", {
        "message": "Inspecting metadata", "role": "tool", "turn_id": "turn-1",
    })
    app._toggle_work_turn("turn-1")
    assert app._read_message_lines(session.path)[-1].text == "Inspecting metadata · expand"


def test_extended_invalidates_cache_and_keeps_intermediate_order(chat):
    _, session, app = chat
    app._read_message_lines(session.path)
    app.work_visualization = "extended"
    messages = app._read_message_lines(session.path)
    assert [line.role for line in messages] == ["user", "tool", "agent_intermediate", "tool"]
    assert [line.text for line in messages][1:] == [
        "Reading channels", "I found the channel list.", "Counting channels",
    ]


def test_new_activity_stays_after_the_user_message_that_triggered_it(chat):
    home, session, app = chat
    home.append_session_event(session.path, "user_message", {"message": "Also inspect metadata"})
    home.append_session_event(session.path, "work_message", {
        "message": "Reading metadata", "role": "tool", "turn_id": "turn-1",
    })
    messages = app._read_message_lines(session.path)
    assert [line.text for line in messages] == [
        "Inspect channels", "Counting channels · expand",
        "Also inspect metadata", "Reading metadata · expand",
    ]


def test_completed_summary_keeps_final_response_in_both_modes(chat):
    home, session, app = chat
    home.append_session_event(session.path, "work_summary", {
        "message": "Worked for 00:03", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {"message": "19 channels"})
    assert app._read_message_lines(session.path) == [
        MessageLine("user", "Inspect channels"),
        MessageLine("work_summary", "Worked for 00:03 · expand", "turn-1"),
        MessageLine("agent", "19 channels"),
    ]
    app.work_visualization = "extended"
    messages = app._read_message_lines(session.path)
    assert len(messages) == 5
    assert messages[-1] == MessageLine("agent", "19 channels")


def test_default_keeps_errors_and_final_output_visible_without_summary(chat):
    home, session, app = chat
    home.append_session_event(session.path, "system_message", {
        "message": "Connection failed", "role": "system", "turn_id": "turn-1",
    })
    home.append_session_event(session.path, "agent_message", {
        "message": "The source is unavailable.", "turn_id": "turn-1",
    })
    messages = app._read_message_lines(session.path)
    assert MessageLine("system", "Connection failed", "turn-1") in messages
    assert MessageLine("agent", "The source is unavailable.", "turn-1") in messages


def test_manage_settings_saves_global_work_visualization(chat, monkeypatch):
    home, _, app = chat
    choices = iter(["work_visualization", "extended", None])
    menus = []

    def select(_window, title, _description, options):
        menus.append((title, options))
        return next(choices)

    monkeypatch.setattr(app, "_menu", select)
    app._run_manage_settings_panel(object())
    assert menus[1][0] == "Work Visualization"
    assert [option.value for option in menus[1][1]] == ["default", "extended"]
    assert home.load_config()["work_visualization"] == "extended"
    assert app.work_visualization == "extended"
    assert any(choice.label == "Work Visualization: Extended"
               for choice in app._manage_settings_choices())


def test_working_fallback_and_long_activity_stay_on_one_line(chat):
    _, _, app = chat
    assert app._messages_with_working_status([], "Thinking")[0].text == "Thinking"
    lines = app._render_messages([
        MessageLine("work_active", "Inspecting " + "channel " * 100, "turn-1"),
    ], 40)
    assert len(lines) == 1
    assert len(lines[0].text) <= 40
    assert lines[0].meta == "turn-1"


def test_statement_schema_requests_short_action_labels():
    description = statement_property("Describe this tool call.")["description"]
    assert "3–7 words" in description
    assert "60 characters" in description
