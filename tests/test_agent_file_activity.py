"""File activity labels follow operations and paths, not model token streaming."""

import json
import shlex

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.file_activity import file_activity_statement
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.tool_manager import CliToolManager
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("printf hello > settings.json", "Writing settings.json"),
        ("echo more >> 'notes with spaces.md'", "Writing notes with spaces.md"),
        (
            "cat <<'EOF' > notes.md\nPath('/anomx/brain/fake').write_text('no')\nEOF",
            "Writing notes.md",
        ),
        ("printf hello | tee src/config.ts", "Writing src/config.ts"),
        ("sed -i 's/old/new/' src/main.py", "Writing src/main.py"),
        ("sed -i '' 's/old/new/' src/main.py", "Writing src/main.py"),
        ("sed --in-place=.bak -f edits.sed src/main.py", "Writing src/main.py"),
        ("cp template.txt result.txt", "Writing result.txt"),
        ("cd src && printf hi > main.py", "Writing src/main.py"),
        (
            'python -c \'from pathlib import Path; Path("settings.json").write_text("ok")\'',
            "Writing settings.json",
        ),
        (
            "python3 - <<'PY'\nfrom pathlib import Path\n"
            "p = Path('src') / 'main.py'\np.write_text('ok')\nPY",
            "Writing src/main.py",
        ),
        ('python -c \'with open("log.txt", "a") as f: f.write("ok")\'', "Writing log.txt"),
        (
            "apply_patch <<'PATCH'\n*** Begin Patch\n*** Update File: src/main.py\n"
            "@@\n-old\n+new\n*** End Patch\nPATCH",
            "Writing src/main.py",
        ),
        ("cat /anomx/brain/preferences.md", "Retrieving Memories"),
        ("rg 'preference' /anomx/brain", "Retrieving Memories"),
        ("ls /anomx/brain", "Retrieving Memories"),
        ("cat < /anomx/brain/preferences.md", "Retrieving Memories"),
        ("printf hello > /anomx/brain/preferences.md", "Updating Memories"),
        (
            'python -c \'from pathlib import Path; Path("/anomx/brain/note.md").write_text("ok")\'',
            "Updating Memories",
        ),
        ("python -c 'print(open(\"/anomx/brain/note.md\").read())'", "Retrieving Memories"),
        (
            "python -c 'from pathlib import Path; "
            "p = Path(\"/anomx/brain/note.md\"); print(p.read_text())'",
            "Retrieving Memories",
        ),
        (
            "echo hi > output.txt; echo note > /anomx/brain/note.md",
            "Writing output.txt · Updating Memories",
        ),
        ("echo hi > one.txt; echo hi > two.txt", "Writing one.txt, two.txt"),
        ("echo '/anomx/brain'", None),
        ("echo 'write > file.txt'", None),
        ("printf hi > /dev/null 2>&1", None),
        ("printf hi > $UNKNOWN_FILE", None),
        ("cat /anomx/brainstorm.md", None),
        ("cat /anomx/brain/../other.txt", None),
        ("rg /anomx/brain .", None),
        ("python -c 'print(\"write > file.txt\")'", None),
        ('python -c \'def later(): open("file.txt", "w")\'', None),
        ("python -c 'from pathlib import Path; Path(unknown).write_text(\"ok\")'", None),
        ("python -c 'broken syntax'", None),
        ("echo 'unclosed", None),
    ],
)
def test_command_file_labels(tmp_path, command, expected):
    manager = CliToolManager(tmp_path)
    assert (
        file_activity_statement(
            "run_command",
            {"command": command},
            manager,
            tmp_path / "brain",
        )
        == expected
    )


@pytest.mark.parametrize("tool", ["read", "grep", "glob", "list"])
def test_memory_tools_use_configured_brain_and_resolved_paths(tmp_path, tool):
    brain = tmp_path / "personal-notes"
    brain.mkdir()
    (tmp_path / "notes-link").symlink_to(brain, target_is_directory=True)
    manager = CliToolManager(tmp_path)
    for path in (str(brain / "profile.md"), "notes-link/profile.md"):
        assert (
            file_activity_statement(tool, {"path": path}, manager, brain) == "Retrieving Memories"
        )
    assert file_activity_statement(tool, {"path": "README.md"}, manager, brain) is None


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("memorize", {"statement": "Saving user preference"}, "Updating Memories"),
        (
            "read",
            {"path": "/anomx/brain/profile.md", "statement": "Reading file"},
            "Retrieving Memories",
        ),
        (
            "run_command",
            {"command": "echo hi > hello.txt", "statement": "Writing"},
            "Writing hello.txt",
        ),
        (
            "run_command",
            {"command": "git status", "statement": "Checking changes"},
            "Checking changes",
        ),
        (
            "run_command",
            {"command": "echo hello", "statement": "Writing response"},
            "Running command",
        ),
    ],
)
def test_runtime_live_activity_matches_completed_tool_statement(
    tmp_path, monkeypatch, tool, arguments, expected
):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    monkeypatch.setattr(runtime, "init_sandbox", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "refresh_mode", lambda: None)
    activity, history = [], []
    callbacks = RuntimeCallbacks(
        tool_activity=activity.append,
        command=lambda statement, *_: history.append(statement),
    )

    def execute(selected_tool, args, cb, session_path):
        ToolExecutionContext(runtime, cb).emit_operator_statement(selected_tool.name, args)
        return "{}"

    monkeypatch.setattr(runtime, "_execute_with_approval", execute)
    runtime._execute_tool(tool, arguments, callbacks)
    assert [item["statement"] for item in activity] == [expected, expected]
    assert history == [expected]


def test_real_file_write_publishes_filename_without_changing_command(tmp_path, monkeypatch):
    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    runtime.set_mode(AgentMode.AUTONOMOUS)
    monkeypatch.setattr(runtime, "init_sandbox", lambda **kwargs: None)
    monkeypatch.setattr(runtime, "refresh_mode", lambda: None)
    activities, commands = [], []
    target = tmp_path / "notes with spaces.txt"
    command = f"printf hello > {shlex.quote(str(target))}"
    result = runtime._execute_tool(
        "run_command",
        {"command": command, "statement": "Writing"},
        RuntimeCallbacks(
            tool_activity=activities.append,
            command=lambda statement, source, _: commands.append((statement, source)),
        ),
    )
    assert json.loads(result)["approved"]
    assert target.read_text() == "hello"
    assert activities[0]["statement"] == "Writing notes with spaces.txt"
    assert commands[0][0] == activities[0]["statement"]
    assert activities[0]["command"] == command
