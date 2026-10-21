import base64

import pytest

from anomx.agent.brain import merge_brain_files, synchronize_brain


def server():
    files = {}

    def exchange(payload):
        nonlocal files
        result = merge_brain_files(files, payload["changes"])
        files = result["files"]
        return result

    return exchange


def sync(root, exchange):
    return synchronize_brain(root / "brain", root / "sync.json", exchange, scope="account-a")


def write(root, name, content):
    path = root / "brain" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_two_clients_create_edit_delete_and_preserve_binary_files(tmp_path):
    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    write(a, "notes/task.md", b"Train a transformer")
    write(a, "assets/example.bin", b"\x00\xff\x80")
    sync(a, exchange)
    sync(b, exchange)
    assert (b / "brain/notes/task.md").read_bytes() == b"Train a transformer"
    assert (b / "brain/assets/example.bin").read_bytes() == b"\x00\xff\x80"
    write(b, "notes/task.md", b"Training verified")
    sync(b, exchange)
    sync(a, exchange)
    assert (a / "brain/notes/task.md").read_bytes() == b"Training verified"
    (a / "brain/notes/task.md").unlink()
    sync(a, exchange)
    sync(b, exchange)
    assert not (b / "brain/notes/task.md").exists()
    sync(a, exchange)
    assert not (a / "brain/notes/task.md").exists()


def test_concurrent_edits_are_preserved_and_eventually_converge(tmp_path):
    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    write(a, "task.md", b"Original")
    sync(a, exchange)
    sync(b, exchange)
    write(a, "task.md", b"Local edit")
    write(b, "task.md", b"Cloud edit")
    sync(a, exchange)
    assert sync(b, exchange)["conflicts"] == ["task.md"]
    assert (b / "brain/task.md").read_bytes() == b"Local edit"
    assert (b / "brain/task_01.md").read_bytes() == b"Cloud edit"
    sync(b, exchange)
    sync(a, exchange)
    assert (a / "brain/task_01.md").read_bytes() == b"Cloud edit"


def test_edit_during_exchange_is_not_lost(tmp_path):
    exchange = server()
    write(tmp_path, "task.md", b"Before")

    def racing(payload):
        result = exchange(payload)
        write(tmp_path, "task.md", b"During sync")
        return result

    assert sync(tmp_path, racing)["conflicts"] == ["task.md"]
    assert (tmp_path / "brain/task_01.md").read_bytes() == b"During sync"


def test_failed_network_and_account_switch_preserve_local_notes(tmp_path):
    write(tmp_path, "task.md", b"Important")
    with pytest.raises(OSError):
        sync(tmp_path, lambda _: (_ for _ in ()).throw(OSError("offline")))
    assert (tmp_path / "brain/task.md").read_bytes() == b"Important"
    exchange = server()
    sync(tmp_path, exchange)
    with pytest.raises(ValueError, match="another platform"):
        synchronize_brain(tmp_path / "brain", tmp_path / "sync.json", exchange, scope="account-b")


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "a/../b", "a\\b", "a//b"])
def test_server_rejects_traversal(path):
    with pytest.raises(ValueError):
        merge_brain_files({}, [{"path": path, "content": "", "base_revision": 0}])


def test_symlink_is_never_uploaded(tmp_path):
    (tmp_path / "brain").mkdir()
    (tmp_path / "secret").write_text("secret")
    (tmp_path / "brain/link").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError, match="symlink"):
        sync(tmp_path, server())


def test_server_conflict_and_retry_are_idempotent():
    change = {"path": "task.md", "content": base64.b64encode(b"one").decode(), "base_revision": 0}
    first = merge_brain_files({}, [change])
    assert merge_brain_files(first["files"], [change]) == first
    conflict = merge_brain_files(
        first["files"], [{**change, "content": base64.b64encode(b"two").decode()}]
    )
    assert base64.b64decode(conflict["files"]["task_01.md"]["content"]) == b"two"
    assert (
        merge_brain_files(
            conflict["files"], [{**change, "content": base64.b64encode(b"two").decode()}]
        )["files"]
        == conflict["files"]
    )
    assert conflict["conflicts"] == ["task.md"]


def test_agent_can_reorganize_a_file_into_a_folder_and_back(tmp_path):
    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    write(a, "notes", b"Original")
    sync(a, exchange)
    sync(b, exchange)
    (a / "brain/notes").unlink()
    write(a, "notes/INDEX.md", b"Organized")
    sync(a, exchange)
    sync(b, exchange)
    assert (b / "brain/notes/INDEX.md").read_bytes() == b"Organized"
    (a / "brain/notes/INDEX.md").unlink()
    (a / "brain/notes").rmdir()
    write(a, "notes", b"Combined")
    sync(a, exchange)
    sync(b, exchange)
    assert (b / "brain/notes").read_bytes() == b"Combined"


def test_first_connection_merges_both_sides_and_numbers_conflicts(tmp_path):
    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    write(a, "notes.md", b"CLI note")
    write(b, "notes.md", b"Cloud note")
    write(b, "notes_01.md", b"Another note")
    sync(b, exchange)
    sync(a, exchange)
    sync(b, exchange)
    for root in (a, b):
        assert (root / "brain/notes.md").read_bytes() == b"Cloud note"
        assert (root / "brain/notes_01.md").read_bytes() == b"Another note"
        assert (root / "brain/notes_02.md").read_bytes() == b"CLI note"
    sync(a, exchange)
    assert len(list((a / "brain").iterdir())) == 3


def test_legacy_note_migrates_and_syncs_as_markdown(tmp_path):
    import json

    from anomx.agent.memories import load_memories

    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    old = json.dumps(
        {
            "title": "Useful facts",
            "summary": "A short note",
            "content": "Keep this information",
            "context": {"source": "user"},
        }
    ).encode()
    write(a, "old.anomx", old)
    sync(a, exchange)
    sync(b, exchange)
    assert not (a / "brain/old.anomx").exists()
    assert (a / ".brain-legacy/old.anomx").read_bytes() == old
    assert "Keep this information" in (b / "brain/useful-facts.md").read_text()
    assert load_memories(b / "brain")[0].title == "Useful facts"


@pytest.mark.parametrize("reverse", [False, True])
def test_first_connection_preserves_file_directory_collisions(tmp_path, reverse):
    exchange = server()
    a, b = tmp_path / "cli", tmp_path / "cloud"
    write(a, "notes", b"Flat note")
    write(b, "notes/topic.md", b"Nested note")
    first, second = (b, a) if reverse else (a, b)
    sync(first, exchange)
    sync(second, exchange)
    sync(first, exchange)
    for root in (a, b):
        assert sorted(
            path.read_bytes() for path in (root / "brain").rglob("*") if path.is_file()
        ) == [b"Flat note", b"Nested note"]


def test_concurrent_notes_and_legacy_archives_keep_every_version(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor

    from anomx.agent.memories import create_memory_record, migrate_legacy_memories, write_memory

    root = tmp_path / "brain"

    def save(index):
        return write_memory(
            root,
            create_memory_record(
                title="Shared title",
                summary="Summary",
                content=f"Version {index}",
                kind="agent",
                context={},
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(save, range(8)))
    assert len({record.path for record in records}) == 8
    for index in range(2):
        write(
            tmp_path, "old.anomx", json.dumps({"title": "Legacy", "content": str(index)}).encode()
        )
        migrate_legacy_memories(root)
    assert len(list((tmp_path / ".brain-legacy").glob("*.anomx"))) == 2


def test_migration_tombstones_do_not_consume_live_note_slots():
    old = {
        f"old-{index}.anomx": {"revision": 1, "deleted": True, "sha256": "", "content": ""}
        for index in range(512)
    }
    result = merge_brain_files(
        old, [{"path": "new.md", "content": base64.b64encode(b"Note").decode()}]
    )
    assert len(result["files"]) == 513
    assert not result["files"]["new.md"]["deleted"]


@pytest.mark.parametrize("conflict", [False, True])
def test_sync_notifications_keep_role_and_message_in_order(tmp_path, conflict):
    from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
    from anomx.agent.store import AnomxHome

    def exchange():
        if conflict:
            return {"conflicts": ["notes.md"]}
        raise OSError("offline")

    runtime = AgentRuntime(AnomxHome(tmp_path / "home"), tmp_path, brain_sync=exchange)
    messages = []
    runtime.synchronize_brain(RuntimeCallbacks(
        system_message=lambda role, message: messages.append((role, message))
    ))
    assert len(messages) == 1
    role, message = messages[0]
    assert role == "system"
    assert message != "system"
    assert ("Both versions" if conflict else "could not sync") in message
