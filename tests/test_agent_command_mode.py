import pytest

from anomx import cli
from anomx.agent import command_mode
from anomx.agent.backends.openai_chat import OpenAICompatibleChatBackend
from anomx.agent.exceptions import AgentBackendError
from anomx.agent.runtime import AgentRuntime
from anomx.agent.store import AnomxHome


def connected_home(tmp_path):
    home = AnomxHome(tmp_path / "home")
    home.set_api_key("blablador", "test-key")
    config = home.load_config()
    config.update(onboarding_complete=True, provider="blablador", model="coding")
    home.save_config(config)
    return home


@pytest.mark.parametrize(
    ("answer", "command"),
    [
        ("ls -d */", "ls -d */"),
        ("```bash\n$ ls -d */\n```", "ls -d */"),
        ("<think>folders only</think>`ls -d */`", "ls -d */"),
        ("No response.", None),
    ],
)
def test_sanitize_shell_command_returns_bare_command(tmp_path, answer, command):
    backend = OpenAICompatibleChatBackend(AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path))

    assert backend._sanitize_shell_command(answer) == command


def test_cli_request_prints_command_from_selected_model(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    calls = []

    def complete_text(self, system, prompt, model, **kwargs):
        calls.append((self.provider_key, model, prompt))
        return "```sh\nls -d */\n```"

    monkeypatch.setattr(OpenAICompatibleChatBackend, "complete_text", complete_text)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["--home", str(home.root), "list", "folders"]) == 0

    assert capsys.readouterr().out == "ls -d */\n"
    provider, model, prompt = calls[0]
    assert (provider, model) == ("blablador", "coding")
    assert prompt.endswith("Request:\nlist folders")
    assert f"Working directory: {tmp_path.resolve()}" in prompt


def test_command_mode_reports_refusals_and_backend_errors(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    answers = iter(["# Not a shell task.", AgentBackendError("Blablador request failed (500)")])

    def complete_text(self, system, prompt, model, **kwargs):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(OpenAICompatibleChatBackend, "complete_text", complete_text)

    assert command_mode.run_command_mode(home, "capital of France", cwd=tmp_path) == 1
    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "anomx: Not a shell task." in captured.err
    assert "anomx: Blablador request failed (500)" in captured.err


def test_command_mode_requires_a_selected_model(tmp_path, capsys):
    home = AnomxHome(tmp_path / "home")

    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 1

    assert "Run `anomx` once to choose a backend and model." in capsys.readouterr().err


def test_command_mode_retries_empty_answers(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    answers = iter(["", "No response.", "du -sh ."])
    monkeypatch.setattr(
        OpenAICompatibleChatBackend,
        "complete_text",
        lambda self, system, prompt, model, **kwargs: next(answers),
    )

    assert command_mode.run_command_mode(home, "folder size", cwd=tmp_path) == 0

    assert capsys.readouterr().out == "du -sh .\n"


def test_command_mode_hands_command_to_shell_integration_file(tmp_path, monkeypatch, capsys):
    home = connected_home(tmp_path)
    command_file = tmp_path / "command"
    monkeypatch.setenv(command_mode.COMMAND_FILE_ENV, str(command_file))
    monkeypatch.setattr(
        OpenAICompatibleChatBackend,
        "complete_text",
        lambda self, system, prompt, model, **kwargs: "ls -d */",
    )

    assert command_mode.run_command_mode(home, "list folders", cwd=tmp_path) == 0

    assert command_file.read_text() == "ls -d */"
    assert capsys.readouterr().out == ""


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
