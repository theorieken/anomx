import json
import queue

import pytest

from anomx import cli
from anomx.agent import command_mode
from anomx.agent.exceptions import BackendFailure
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandRiskEvaluation
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import CLI_HISTORY_LIMIT, AnomxHome
from anomx.agent.ui.activity import ActivityPresentation


def connected_home(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("blablador", "test-key")
    config = home.load_config()
    config.update(onboarding_complete=True, provider="blablador", model="coding")
    home.save_config(config)
    return home


def fake_turn(monkeypatch, *responses):
    """Replace the model turn; record each turn's transcript, prompt and selection."""

    turns = []
    answers = iter(responses)

    def backend_response(self, session_path, callbacks=None, *, provider=None, model=None, **_):
        turns.append(
            {
                "messages": self.conversation_messages(session_path),
                "instructions": self._instructions(session_path),
                "provider": provider,
                "model": model,
                "callbacks": callbacks,
            }
        )
        return next(answers)

    monkeypatch.setattr(AgentRuntime, "backend_response", backend_response)
    return turns


@pytest.mark.parametrize(
    ("response", "kind", "text"),
    [
        ('{"type": "command", "command": "ls -d */"}', "command", "ls -d */"),
        ('```json\n{"type": "command", "command": "```sh\\n$ ls\\n```"}\n```', "command", "ls"),
        ('<think>easy</think>{"type": "answer", "answer": "Paris."}', "answer", "Paris."),
        ('{"type": "answer", "answer": "Intro.\n\n- a\n- b"}', "answer", "Intro.\n\n- a\n- b"),
        ('{"type": "answer", "answer": "Named "Anomx" here."}', "answer", 'Named "Anomx" here.'),
        ('```json\n{"type": "command", "command": "echo "hi""}\n```', "command", 'echo "hi"'),
        ("Just prose without JSON.", "answer", "Just prose without JSON."),
        ('{"type": "command", "command": ""}', "answer", '{"type": "command", "command": ""}'),
    ],
)
def test_parse_quick_response_distinguishes_commands_and_answers(response, kind, text):
    assert command_mode.parse_quick_response(response) == command_mode.QuickResponse(kind, text)


def test_cli_request_runs_agent_turn_with_selected_model(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    turns = fake_turn(monkeypatch, '{"type": "command", "command": "ls -d */"}')
    monkeypatch.chdir(tmp_path)

    assert cli.main(["--home", str(home.root), "list", "folders"]) == 0

    assert capsys.readouterr().out == "ls -d */\n"
    turn = turns[0]
    assert (turn["provider"], turn["model"]) == ("blablador", "coding")
    assert turn["messages"] == [
        {"role": "user", "content": f"Directory: {tmp_path.resolve()}\nRequest: list folders"}
    ]
    assert "You are Anomx in quick mode." in turn["instructions"]
    assert '{"type": "command", "command": "<shell command>"}' in turn["instructions"]
    assert not (home.root / "cli_runs").exists() or not any((home.root / "cli_runs").iterdir())


def test_answers_are_printed_and_both_outcomes_join_the_history(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    command_file = tmp_path / "command"
    monkeypatch.setenv(command_mode.COMMAND_FILE_ENV, str(command_file))
    turns = fake_turn(
        monkeypatch,
        '{"type": "answer", "answer": "Three folders."}',
        '{"type": "command", "command": "ls -d */ | wc -l"}',
    )

    assert command_mode.run_command_mode(home, "how many folders?", cwd=tmp_path) == 0
    assert command_file.exists() is False
    assert capsys.readouterr().out == "Three folders.\n"

    assert command_mode.run_command_mode(home, "as a command", cwd=tmp_path / "sub") == 0
    assert command_file.read_text() == "ls -d */ | wc -l"
    assert capsys.readouterr().out == ""

    assert turns[1]["messages"] == [
        {"role": "user", "content": f"Directory: {tmp_path}\nRequest: how many folders?"},
        {"role": "assistant", "content": '{"type": "answer", "answer": "Three folders."}'},
        {"role": "user", "content": f"Directory: {tmp_path / 'sub'}\nRequest: as a command"},
    ]
    assert [(entry["request"], entry["type"], entry["content"], entry["cwd"])
            for entry in home.load_cli_history()] == [
        ("how many folders?", "answer", "Three folders.", str(tmp_path)),
        ("as a command", "command", "ls -d */ | wc -l", str(tmp_path / "sub")),
    ]


def test_cli_history_keeps_only_the_latest_exchanges(tmp_path):
    home = AnomxHome(tmp_path / "home")

    for index in range(CLI_HISTORY_LIMIT + 5):
        home.append_cli_history({"request": f"r{index}", "type": "answer", "content": "ok"})

    lines = home.cli_history_path.read_text().splitlines()
    assert len(lines) == CLI_HISTORY_LIMIT
    assert json.loads(lines[0])["request"] == "r5"
    assert home.load_cli_history(2)[-1]["request"] == f"r{CLI_HISTORY_LIMIT + 4}"


def test_backend_failures_are_reported_and_not_remembered(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    fake_turn(monkeypatch, BackendFailure("Blablador request failed (500)"), "")

    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 1
    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "anomx: Blablador request failed (500)" in captured.err
    assert "anomx: The model returned no response." in captured.err
    assert home.load_cli_history() == []


def test_command_mode_requires_a_selected_model(tmp_path, capsys):
    home = AnomxHome(tmp_path / "home")

    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 1

    assert "Run `anomx` once to choose a backend and model." in capsys.readouterr().err


@pytest.mark.parametrize(
    ("evaluation", "choice"),
    [
        (
            CommandRiskEvaluation("low", "lists files", "allow", read_only=True),
            ApprovalChoice.ALLOW,
        ),
        (CommandRiskEvaluation("low", "edits a file", "allow"), ApprovalChoice.REJECT),
        (None, ApprovalChoice.REJECT),
    ],
)
def test_quick_agent_only_runs_commands_judged_read_only(
    tmp_path, monkeypatch, capsys, evaluation, choice
):
    home = connected_home(tmp_path)
    turns = fake_turn(monkeypatch, '{"type": "answer", "answer": "ok"}')
    monkeypatch.setattr(AgentRuntime, "evaluate_command_request", lambda *args: evaluation)

    assert command_mode.run_command_mode(home, "edit the first file", cwd=tmp_path) == 0

    assert turns[0]["callbacks"].approval(object()) == choice


def test_live_label_shows_what_the_agent_is_doing():
    activity = ActivityPresentation()
    events = queue.SimpleQueue()

    assert command_mode._drain_activity(activity, events) == "Thinking"

    events.put(command_mode._ActivityEvent(
        "tool", payload={"id": "1", "status": "running", "statement": "Reading\nREADME.md"}
    ))
    assert command_mode._drain_activity(activity, events) == "Reading README.md"


@pytest.mark.parametrize(("shell", "prefill"), [("zsh", "print -z"), ("bash", "read -r -e -i")])
def test_shell_init_prints_prompt_prefill_wrapper(capsys, shell, prefill):
    assert cli.main(["--shell-init", shell]) == 0

    script = capsys.readouterr().out
    assert script.startswith("anomx() {")
    assert command_mode.COMMAND_FILE_ENV in script
    assert prefill in script


def test_multiline_commands_are_never_queued_as_terminal_input(monkeypatch):
    import fcntl

    queued = []
    monkeypatch.setattr(fcntl, "ioctl", lambda *args: queued.append(args))

    assert command_mode._insert_into_prompt("cd src\nls") is False
    assert queued == []
