"""Upgraded CLI launches migrate each user's home without manual commands."""

from __future__ import annotations

import json
from pathlib import Path

from anomx.agent.app import AnomxCliApp
from anomx.agent.store import AnomxHome


def legacy_home(root: Path, identifier: str) -> tuple[Path, bytes, bytes]:
    transcript = root / f"sessions/2026/10/21/rollout-2026-10-21-{identifier}.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({
        "type": "session_meta",
        "payload": {"id": identifier, "title": "Preserved chat", "cwd": str(root.parent)},
    }) + "\n", encoding="utf-8")
    config = b'onboarding_complete = true\n'
    auth = b'{"api_keys": {"example": "test-value"}}\n'
    (root / "config.toml").write_bytes(config)
    (root / "auth.json").write_bytes(auth)
    (root / "responses").mkdir()
    (root / "responses/result.json").write_text('{"value": 42}', encoding="utf-8")
    return transcript, config, auth


def test_normal_cli_launch_migrates_each_users_configured_home(tmp_path, monkeypatch):
    roots = [tmp_path / "alice" / ".anomx", tmp_path / "bob" / ".anomx"]
    for index, root in enumerate(roots):
        identifier = f"chat{index}"
        original, config, auth = legacy_home(root, identifier)
        monkeypatch.setenv("ANOMX_HOME", str(root))
        app = AnomxCliApp(cwd=tmp_path, isolate_runtime=True)
        try:
            assert app.home.root == root
            assert not original.exists()
            expected = root / "sessions" / identifier / "session.jsonl"
            assert app.home.list_sessions()[0].path == expected
            assert (root / "config.toml").read_bytes() == config
            assert (root / "auth.json").read_bytes() == auth
            result = root / "sessions/_legacy/tmp/responses/result.json"
            assert result.read_text() == '{"value": 42}'
        finally:
            app.runtime.close()
    assert all((root / "sessions").is_dir() for root in roots)


def test_explicit_home_migrates_automatically_and_relaunch_preserves_history(tmp_path):
    root = tmp_path / "custom-home"
    original, config, auth = legacy_home(root, "custom-chat")
    for _ in range(2):
        app = AnomxCliApp(home=AnomxHome(root), cwd=tmp_path, isolate_runtime=True)
        try:
            assert not original.exists()
            assert [record.session_id for record in app.home.list_sessions()] == ["custom-chat"]
            assert (root / "config.toml").read_bytes() == config
            assert (root / "auth.json").read_bytes() == auth
        finally:
            app.runtime.close()
