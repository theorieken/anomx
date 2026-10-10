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
    assert next((b / "brain/conflicts").iterdir()).read_bytes() == b"Cloud edit"
    sync(b, exchange)
    sync(a, exchange)
    assert next((a / "brain/conflicts").iterdir()).read_bytes() == b"Cloud edit"


def test_edit_during_exchange_is_not_lost(tmp_path):
    exchange = server()
    write(tmp_path, "task.md", b"Before")

    def racing(payload):
        result = exchange(payload)
        write(tmp_path, "task.md", b"During sync")
        return result

    assert sync(tmp_path, racing)["conflicts"] == ["task.md"]
    assert next((tmp_path / "brain/conflicts").iterdir()).read_bytes() == b"During sync"


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
    assert conflict["files"] == first["files"]
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
