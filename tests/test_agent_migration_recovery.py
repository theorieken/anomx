"""The automatic upgrade survives interruptions and simultaneous new clients."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome


def legacy_home(root):
    old = root / "sessions/2026/10/21/rollout-old-chat.jsonl"
    old.parent.mkdir(parents=True)
    response = root / "responses/result.json"
    response.parent.mkdir()
    response.write_text(json.dumps({"original_path": str(response), "value": 42}))
    events = [
        {"type": "session_meta", "payload": {"id": "chat", "cwd": str(root.parent)}},
        {
            "type": "event_msg",
            "payload": {
                "type": "tool_execution",
                "tool": "example",
                "result": json.dumps({"result_path": str(response)}),
            },
        },
        {"type": "event_msg", "payload": {"type": "user_message", "message": str(response)}},
    ]
    old.write_text("".join(json.dumps(event) + "\n" for event in events))
    (root / "session_index.jsonl").write_text(json.dumps({"path": str(old)}) + "\n")
    (root / "config.toml").write_text("onboarding_complete = true\n")
    (root / "auth.json").write_text('{"keep": "credentials"}\n')
    return old, response


@pytest.mark.parametrize("stage", ["first_move", "transcript_move", "before_complete"])
def test_interrupted_migration_replays_journal_and_repairs_references(tmp_path, monkeypatch, stage):
    root = tmp_path / "home"
    old, response = legacy_home(root)
    home = AnomxHome(root)
    original_move = home._move_legacy_path
    original_journal = home._write_layout_journal
    moved = 0

    def fail_after_move(source, destination):
        nonlocal moved
        original_move(source, destination)
        moved += 1
        if (stage == "first_move" and moved == 1) or (stage == "transcript_move" and source == old):
            raise RuntimeError("simulated interrupted upgrade")

    def fail_before_complete(journal):
        if stage == "before_complete" and journal["status"] == "complete":
            raise RuntimeError("simulated interrupted upgrade")
        original_journal(journal)

    monkeypatch.setattr(home, "_move_legacy_path", fail_after_move)
    monkeypatch.setattr(home, "_write_layout_journal", fail_before_complete)
    with pytest.raises(RuntimeError, match="interrupted upgrade"):
        home.ensure()
    assert json.loads(home._layout_journal_path.read_text())["status"] == "pending"

    resumed = AnomxHome(root)
    resumed.ensure()
    new = root / "sessions/chat/session.jsonl"
    archived = root / "sessions/_legacy/tmp/responses/result.json"
    events = resumed.read_session_events(new)
    assert json.loads(events[1]["payload"]["result"])["result_path"] == str(archived)
    assert events[2]["payload"]["message"] == str(response)  # User prose stays verbatim.
    assert json.loads(archived.read_text())["original_path"] == str(response)  # Raw data too.
    assert str(new) in resumed.session_index_path.read_text()
    assert resumed.resolve_legacy_path(old) == new
    assert resumed.resolve_legacy_path(response) == archived
    assert json.loads(resumed._layout_journal_path.read_text())["status"] == "complete"
    assert (root / "config.toml").read_text() == "onboarding_complete = true\n"
    assert (root / "auth.json").read_text() == '{"keep": "credentials"}\n'


def migrate_in_process(root):
    AnomxHome(Path(root)).ensure()


def test_simultaneous_processes_migrate_once_without_losing_data(tmp_path):
    root = tmp_path / "home"
    legacy_home(root)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=migrate_in_process, args=(str(root),)) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("Concurrent migration did not release its process lock.")
        assert process.exitcode == 0
    home = AnomxHome(root)
    assert [record.session_id for record in home.list_sessions()] == ["chat"]
    assert len(home.read_session_events(root / "sessions/chat/session.jsonl")) == 3
    assert json.loads(home._layout_journal_path.read_text())["status"] == "complete"


def test_completed_migration_does_not_scan_history_again(tmp_path, monkeypatch):
    root = tmp_path / "home"
    legacy_home(root)
    AnomxHome(root).ensure()
    fresh = AnomxHome(root)

    def should_not_run():
        pytest.fail("A completed migration must not rescan history on startup.")

    monkeypatch.setattr(fresh, "_plan_legacy_layout", should_not_run)
    fresh.ensure()


def test_collisions_and_unknown_content_are_preserved(tmp_path):
    root = tmp_path / "home"
    old, _ = legacy_home(root)
    destination = root / "sessions/chat/session.jsonl"
    destination.parent.mkdir()
    destination.write_text("existing transcript\n")
    old_content = old.read_bytes()
    for directory in (root / "instructions", root / "brain/instructions"):
        directory.mkdir(parents=True)
    (root / "instructions/note.md").write_text("old instructions")
    (root / "brain/instructions/note.md").write_text("new instructions")
    unknown = root / "my-notes.md"
    unknown.write_text("Personal notes")
    AnomxHome(root).ensure()
    assert old.read_bytes() == old_content
    assert destination.read_text() == "existing transcript\n"
    assert (root / "instructions/note.md").read_text() == "old instructions"
    assert (root / "brain/instructions/note.md").read_text() == "new instructions"
    assert unknown.read_text() == "Personal notes"


def test_file_tools_can_read_old_response_paths_without_trusting_auth(tmp_path):
    root = tmp_path / "home"
    _, response = legacy_home(root)
    task = tmp_path / "task"
    task.mkdir()
    runtime = AgentRuntime(AnomxHome(root), task, local_sandbox_enabled=True)
    result = json.loads(
        runtime._execute_tool(
            "read",
            {"statement": "Read prior evidence", "path": str(response)},
            RuntimeCallbacks(),
        )
    )
    assert json.loads(result["content"])["value"] == 42
    blocked = json.loads(
        runtime._execute_tool(
            "read",
            {"statement": "Read config", "path": str(root / "auth.json")},
            RuntimeCallbacks(),
        )
    )
    assert "error" in blocked
    assert response not in runtime.tool_manager.path_aliases
    assert not runtime.tool_manager.path_inside_workspace(
        runtime.tool_manager.resolve_trusted_path(str(response))
    )


@pytest.mark.parametrize("attack", ["move_auth", "repair_auth", "repair_cache"])
def test_invalid_journal_cannot_move_or_rewrite_unrelated_files(tmp_path, attack):
    root = tmp_path / "home"
    old, response = legacy_home(root)
    home = AnomxHome(root)
    moves = [
        {
            "source": str(response.parent),
            "destination": str(root / "sessions/_legacy/tmp/responses"),
        }
    ]
    transcripts = []
    if attack == "move_auth":
        moves = [
            {
                "source": str(home.auth_path),
                "destination": str(root / "sessions/_legacy/tmp/auth.json"),
            }
        ]
    elif attack == "repair_auth":
        transcripts = [str(home.auth_path)]
    else:
        transcripts = [str(response)]
    home._layout_journal_path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "pending",
                "moves": moves,
                "transcripts": transcripts,
            }
        )
    )
    with pytest.raises(ValueError):
        home.ensure()
    assert home.auth_path.read_text() == '{"keep": "credentials"}\n'
    assert old.exists() and response.exists()


@pytest.mark.parametrize("name", [".layout-v1.json", ".layout.lock"])
def test_migration_rejects_symlink_control_files(tmp_path, name):
    root = tmp_path / "home"
    legacy_home(root)
    home = AnomxHome(root)
    (home.sessions_dir / name).symlink_to(home.auth_path)
    with pytest.raises(ValueError, match="symbolic link"):
        home.ensure()
    assert home.auth_path.read_text() == '{"keep": "credentials"}\n'


def test_old_fixed_temp_symlinks_cannot_overwrite_credentials(tmp_path):
    root = tmp_path / "home"
    legacy_home(root)
    home = AnomxHome(root)
    (home.sessions_dir / ".layout-v1.pending").symlink_to(home.auth_path)
    destination = home.sessions_dir / "chat"
    destination.mkdir()
    (destination / "session.layout-v1.tmp").symlink_to(home.auth_path)
    home.ensure()
    assert home.auth_path.read_text() == '{"keep": "credentials"}\n'


def test_explicit_artifact_alias_trust_does_not_cover_sibling_files(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    own = archive / "own.json"
    own.write_text('{"value": 42}')
    sibling = archive / "other-user.json"
    sibling.write_text('{"private": true}')
    old = tmp_path / "old-result.json"
    runtime = AgentRuntime(
        AnomxHome(tmp_path / "home"),
        task,
        additional_trusted_roots=(own,),
        path_aliases={old: own},
        local_sandbox_enabled=True,
    )
    result = json.loads(
        runtime._execute_tool(
            "read",
            {"statement": "Read own result", "path": str(old)},
            RuntimeCallbacks(),
        )
    )
    assert json.loads(result["content"])["value"] == 42
    blocked = json.loads(
        runtime._execute_tool(
            "read",
            {"statement": "Read sibling", "path": str(sibling)},
            RuntimeCallbacks(),
        )
    )
    assert "error" in blocked


@pytest.mark.parametrize("link_parent", [False, True])
def test_symlink_archive_does_not_broaden_runtime_trust(tmp_path, link_parent):
    home = AnomxHome(tmp_path / "home")
    home.ensure()
    private = tmp_path / "private"
    private.mkdir()
    (private / "credentials.json").write_text("private credentials")
    if link_parent:
        (private / "tmp").mkdir()
        (home.sessions_dir / "_legacy").symlink_to(private)
    else:
        (home.sessions_dir / "_legacy").mkdir()
        (home.sessions_dir / "_legacy/tmp").symlink_to(private)
    task = tmp_path / "task"
    task.mkdir()
    runtime = AgentRuntime(home, task, local_sandbox_enabled=True)
    assert private not in runtime.trusted_roots
    blocked = json.loads(
        runtime._execute_tool(
            "read",
            {"statement": "Read credentials", "path": str(private / "credentials.json")},
            RuntimeCallbacks(),
        )
    )
    assert "error" in blocked
