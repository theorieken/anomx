"""Chat isolation, lossless layout migration, and container mount contracts."""
from __future__ import annotations

import json
from pathlib import Path

from anomx.agent.helpers.sandbox import SandboxConfig, SandboxSession
from anomx.agent.helpers.tool_manager import CliToolManager
from anomx.agent.helpers.utils import session_id_from_path
from anomx.agent.store import AnomxHome


def test_chat_directories_isolate_buffers_and_share_memory(tmp_path):
    home = AnomxHome(tmp_path / "home")
    first = home.create_session(tmp_path, "openai", "example")
    second = home.create_session(tmp_path, "openai", "example")
    one, two = home.for_session(first.path), home.for_session(second.path)
    assert first.path == home.sessions_dir / first.session_id / "session.jsonl"
    assert session_id_from_path(first.path) == first.session_id
    assert one.tmp_dir != two.tmp_dir
    assert one.sub_agents_dir != two.sub_agents_dir
    assert one.brain_dir == two.brain_dir == home.brain_dir
    assert one.skills_dir == two.skills_dir == home.skills_dir
    assert one.responses_dir.is_dir()
    assert one.sub_agents_dir.is_dir()
    assert home.tmp_dir != one.tmp_dir
    assert not (home.root / "responses").exists()
    assert not (home.root / "search").exists()
    assert not (home.root / "instructions").exists()


def test_explicit_runtime_mounts_survive_chat_and_child_scoping(tmp_path):
    home = AnomxHome(
        tmp_path / "config", brain_root=tmp_path / "brain", skills_root=tmp_path / "skills",
        tmp_root=tmp_path / "tmp", sub_agents_root=tmp_path / "sub_agents",
    )
    session = home.create_session(tmp_path / "task", "openai", "example")
    parent = home.for_session(session.path)
    child_path = home.subagent_session_path(
        parent_session_path=session.path, subagent_name="Research", subagent_id="child1"
    )
    child = parent.for_session(child_path)
    assert child.tmp_dir == tmp_path / "tmp"
    assert child.sub_agents_dir == tmp_path / "sub_agents"
    assert child.skills_dir == tmp_path / "skills"
    assert child.brain_dir == tmp_path / "brain"
    assert child_path.parent.parent == tmp_path / "sub_agents"


def test_migrate_legacy_chat_without_losing_files_or_parent_references(tmp_path):
    root = tmp_path / "home"
    old = root / "sessions/2026/10/21/rollout-time-abc123.jsonl"
    old.parent.mkdir(parents=True)
    events = [{"type": "session_meta", "payload": {"id": "abc123", "cwd": "/project"}}]
    old.write_text(json.dumps(events[0]) + "\n")
    child = old.parent / "subagent-Research-child1"
    child.mkdir()
    (child / "session.jsonl").write_text(json.dumps({
        "type": "subagent_session_meta", "payload": {
            "parent_session_id": "abc123", "parent_session_path": str(old)
        }
    }) + "\n")
    (child / "notes.md").write_text("Keep this research")
    (root / "responses").mkdir()
    (root / "responses/result.json").write_text('{"value": 42}')
    (root / "instructions").mkdir()
    (root / "instructions/instruction.md").write_text("Keep instructions")
    (root / "usage.json").write_text('{"20261021": 3}')
    (root / "session_index.jsonl").write_text(json.dumps({"path": str(old)}) + "\n")
    (root / "unknown.txt").write_text("User content")
    home = AnomxHome(root)
    home.ensure()
    home.migrate_legacy_layout()  # Idempotent after restart.
    new = root / "sessions/abc123/session.jsonl"
    assert new.exists() and not old.exists()
    assert home.list_sessions()[0].session_id == "abc123"
    moved_child = new.parent / "sub_agents" / child.name
    assert (moved_child / "notes.md").read_text() == "Keep this research"
    assert json.loads((moved_child / "session.jsonl").read_text())["payload"][
        "parent_session_path"
    ] == str(new)
    assert str(new) in home.session_index_path.read_text()
    assert (root / "sessions/_legacy/tmp/responses/result.json").read_text() == '{"value": 42}'
    assert (home.instructions_dir / "instruction.md").read_text() == "Keep instructions"
    assert home.usage_path.read_text() == '{"20261021": 3}'
    assert (root / "unknown.txt").read_text() == "User content"
    assert not (root / "sessions/2026").exists()


def test_delete_chat_removes_only_its_scoped_work(tmp_path):
    home = AnomxHome(tmp_path / "home")
    one = home.create_session(tmp_path, "openai", "example")
    two = home.create_session(tmp_path, "openai", "example")
    (home.for_session(one.path).tmp_dir / "data.txt").write_text("temporary")
    (home.brain_dir / "memory.md").write_text("persistent")
    assert home.delete_session(one.path)
    assert not one.path.parent.exists()
    assert two.path.exists()
    assert (home.brain_dir / "memory.md").exists()


def test_container_mounts_only_selected_data_and_separates_chats(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    folders = {name: tmp_path / name for name in ("brain", "skills", "tmp", "sub_agents")}
    session = SandboxSession(
        SandboxConfig(), project, runtime_directories=folders, session_key="one"
    )
    other = SandboxSession(SandboxConfig(), project, runtime_directories=folders, session_key="two")
    command = session.build_run_command("example", ["true"], interactive=False)
    mounts = [command[index + 1] for index, arg in enumerate(command) if arg == "-v"]
    assert f"{project}:/anomx/task" in mounts
    assert all(f"{source}:/anomx/{name}" in mounts for name, source in folders.items())
    assert not any("auth.json" in mount or "config.toml" in mount for mount in mounts)
    assert session.container_name != other.container_name
    assert session.container_path(folders["sub_agents"] / "research") == Path(
        "/anomx/sub_agents/research"
    )
    manager = CliToolManager(project, trusted_roots=tuple(folders.values()))
    manager.path_aliases = session.path_aliases
    assert manager.resolve_trusted_path("/anomx/brain/memory.md") == folders["brain"] / "memory.md"
    escaped = manager.resolve_trusted_path("/anomx/brain/../../outside")
    assert not manager.path_inside_workspace(escaped)


def test_stopped_container_never_falls_back_to_host(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from anomx.agent.base.tools import ToolExecutionContext
    from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
    from anomx.agent.tools.cli_command import CliCommandTool

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path)
    runtime._sandbox_session = SimpleNamespace(is_running=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host execution must never be used for a stopped container")

    monkeypatch.setattr(runtime.tool_manager, "run_command", forbidden)
    output = CliCommandTool(statement_description="Working").execute(
        {"command": "pwd", "statement": "Check task location"},
        ToolExecutionContext(runtime=runtime, callbacks=RuntimeCallbacks(), session_path=None),
    )
    assert json.loads(output)["approved"] is False
    assert "runtime is stopped" in json.loads(output)["output"].lower()


def test_container_commands_use_subagent_directory_and_only_platform_environment(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    folders = {name: tmp_path / name for name in ("brain", "skills", "tmp", "sub_agents")}
    sandbox = SandboxSession(SandboxConfig(), project, runtime_directories=folders)
    sandbox._container_id = "example"
    calls = []
    monkeypatch.setattr(sandbox, "_run_captured", lambda args, **kwargs: calls.append(args) or "ok")
    assert sandbox.exec_command(
        "pwd", current_dir=folders["sub_agents"] / "research",
        env={
            "ANOMX_RESPONSES_DIR": str(folders["tmp"] / "responses"),
            "UNRELATED_SECRET": "private",
        },
    ) == "ok"
    args = calls[0]
    assert args[args.index("-w") + 1] == "/anomx/sub_agents/research"
    assert any(value.startswith("ANOMX_RESPONSES_DIR=") for value in args)
    assert not any("UNRELATED_SECRET" in value for value in args)
