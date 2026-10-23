"""Background approval automation keeps its read-only scope and CLI defaults persist."""

import json

import pytest

from anomx.agent.app import AnomxCliApp
from anomx.agent.base.backends import BaseBackend
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import (
    ApprovalChoice,
    CommandRiskEvaluation,
    approval_choice_for_mode,
)
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def background_runtime(tmp_path, **kwargs):
    return AgentRuntime(
        AnomxHome(tmp_path / "home"),
        tmp_path,
        mode=AgentMode.BACKGROUND,
        **kwargs,
    )


@pytest.mark.parametrize(
    "decision,read_only,allowed",
    [
        ("allow", True, True),
        ("deny", True, False),
        ("allow", False, False),
    ],
)
def test_background_evaluator_decides_reads_without_human_review(
    tmp_path,
    decision,
    read_only,
    allowed,
):
    runtime = background_runtime(tmp_path)
    reviews = []

    def evaluate(request):
        reviews.append(request)
        return approval_choice_for_mode(
            AgentMode.BACKGROUND,
            CommandRiskEvaluation("low", "Read Python version", decision, read_only=read_only),
        )

    result = json.loads(
        runtime._execute_tool(
            "run_command",
            {"command": "python3 -V", "statement": "Read interpreter version"},
            RuntimeCallbacks(approval=evaluate),
        )
    )
    assert result["approved"] is allowed
    assert len(reviews) == 1
    assert "approval_pending" not in result
    if allowed:
        assert "Python" in result["output"]


@pytest.mark.parametrize("evaluation", [None, CommandRiskEvaluation("low", "Legacy", "allow")])
def test_background_fails_closed_without_read_only_assessment(evaluation):
    assert approval_choice_for_mode(AgentMode.BACKGROUND, evaluation) == ApprovalChoice.REJECT


def test_background_saved_interactive_allowance_cannot_authorize_writes(tmp_path):
    runtime = background_runtime(tmp_path, session_allowed_commands={"cmd:touch"})
    reviewed = []
    result = json.loads(
        runtime._execute_tool(
            "run_command",
            {"command": "touch changed.txt", "statement": "Create file"},
            RuntimeCallbacks(
                approval=lambda request: (
                    reviewed.append(request)
                    or approval_choice_for_mode(
                        AgentMode.BACKGROUND,
                        CommandRiskEvaluation(
                            "low",
                            "Requested file creation",
                            "allow",
                            read_only=False,
                        ),
                    )
                )
            ),
        )
    )
    assert reviewed and not result["approved"]
    assert not (tmp_path / "changed.txt").exists()


@pytest.mark.parametrize("command", ["cat ../outside.txt", "touch denied.txt"])
def test_background_hard_boundaries_and_explicit_denials_do_not_reach_evaluator(tmp_path, command):
    runtime = background_runtime(tmp_path, session_rejected_commands={"cmd:touch"})
    result = json.loads(
        runtime._execute_tool(
            "run_command",
            {"command": command, "statement": "Inspect workspace"},
            RuntimeCallbacks(approval=lambda _: pytest.fail("Hard policy must remain enforced")),
        )
    )
    assert not result["approved"]
    assert not (tmp_path / "denied.txt").exists()


def test_background_evaluator_read_only_flag_is_strict_boolean(tmp_path):
    backend = BaseBackend(background_runtime(tmp_path))
    payload = {"risk": "low", "description": "Inspect data", "decision": "allow"}
    for flag in (None, "true", 1, False):
        result = backend._sanitize_command_evaluation(json.dumps({**payload, "read_only": flag}))
        assert not result.read_only
        assert approval_choice_for_mode(AgentMode.BACKGROUND, result) == ApprovalChoice.REJECT
    result = backend._sanitize_command_evaluation(json.dumps({**payload, "read_only": True}))
    assert result.read_only
    assert approval_choice_for_mode(AgentMode.BACKGROUND, result) == ApprovalChoice.ALLOW


@pytest.mark.parametrize(
    "stored,expected",
    [
        (None, AgentMode.AUTOMATIC),
        ("invalid", AgentMode.AUTOMATIC),
        ("standard", AgentMode.STANDARD),
        ("autonomous", AgentMode.AUTONOMOUS),
    ],
)
def test_cli_defaults_to_auto_approve_and_preserves_explicit_saved_choice(
    tmp_path, stored, expected
):
    home = AnomxHome(tmp_path / "home")
    if stored is not None:
        home.save_config({"agent_mode": stored})
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    assert app.agent_mode == expected
    assert app.runtime.tool_manager.mode == expected
    assert app._create_session().mode == expected


@pytest.mark.parametrize("mode", [AgentMode.STANDARD, AgentMode.AUTONOMOUS])
def test_cli_opening_session_restores_its_explicit_mode(tmp_path, monkeypatch, mode):
    class Window:
        keys = iter(("\x03", "\x03"))

        def get_wch(self):
            return next(self.keys)

    home = AnomxHome(tmp_path / "home")
    session = home.create_session(tmp_path, provider="openai", model="example", mode=mode)
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    assert app.agent_mode == AgentMode.AUTOMATIC
    monkeypatch.setattr(app, "_draw_session", lambda *args, **kwargs: None)
    assert app._run_session(Window(), session) == 0
    assert app.agent_mode == mode
    assert app.runtime.tool_manager.mode == mode
    assert home.load_config()["agent_mode"] == "automatic"


def test_return_to_project_uses_persisted_default_selection(tmp_path, monkeypatch):
    home = AnomxHome(tmp_path / "home")
    home.save_config({"agent_mode": "standard"})
    session = home.create_session(tmp_path, provider="openai", model="example", mode="autonomous")
    app = AnomxCliApp(home=home, cwd=tmp_path, use_color=False)
    app._activate_agent_mode(AgentMode.AUTONOMOUS)
    monkeypatch.setattr(app, "_prepare_session_for_opening", lambda value: value)
    monkeypatch.setattr(app, "_run_session", lambda *args: "project")
    assert app._open_project_session(object(), session) is None
    assert app.agent_mode == AgentMode.STANDARD
    assert home.load_config()["agent_mode"] == "standard"
