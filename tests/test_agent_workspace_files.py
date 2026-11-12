import hashlib
import json

import pytest

from anomx.agent.base.tools import ToolExecutionContext
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.runtime import AgentRuntime, RuntimeCallbacks
from anomx.agent.store import AnomxHome
from anomx.agent.tools.manage_workspace_files import ManageWorkspaceFilesTool


def context(tmp_path):
    home = AnomxHome(tmp_path / "home")
    return ToolExecutionContext(
        AgentRuntime(home, tmp_path, mode=AgentMode.BACKGROUND), RuntimeCallbacks()
    )


def call(ctx, **kwargs):
    args = {
        "area": "brain",
        "action": "write",
        "path": "checkpoint.md",
        "content": "checked",
        "expected_sha256": None,
        **kwargs,
    }
    return json.loads(ManageWorkspaceFilesTool().execute(args, ctx))


def test_background_checkpoint_requires_current_revision_for_overwrite(tmp_path):
    ctx = context(tmp_path)
    assert "manage_workspace_files" in {t.name for t in ctx.runtime._available_tools()}
    saved = call(ctx)
    assert saved["completed"]
    assert "error" in call(ctx, content="stale")
    assert call(ctx, content="updated", expected_sha256=saved["sha256"])["completed"]
    assert (ctx.runtime.home.brain_dir / "checkpoint.md").read_text() == "updated"


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", ".hidden", ".", ""])
def test_rejects_escaping_and_reserved_paths(tmp_path, path):
    assert "error" in call(context(tmp_path), path=path)


def test_rejects_symlink_escape_and_brain_deletion(tmp_path):
    ctx = context(tmp_path)
    (ctx.runtime.home.brain_dir / "escape").symlink_to(tmp_path, target_is_directory=True)
    assert "error" in call(ctx, path="escape/outside.md")
    assert "error" in call(ctx, action="delete")


def test_cleanup_deletes_only_exact_unchanged_scratch_file(tmp_path):
    ctx = context(tmp_path)
    saved = call(ctx, area="tmp", path="response.json")
    assert saved["completed"]
    sibling = ctx.runtime.home.tmp_dir / "keep.txt"
    sibling.write_text("keep")
    assert "error" in call(ctx, area="tmp", action="delete", path="response.json")
    assert call(
        ctx,
        area="tmp",
        action="delete",
        path="response.json",
        expected_sha256=hashlib.sha256(b"checked").hexdigest(),
    )["completed"]
    assert sibling.read_text() == "keep"
    assert not (ctx.runtime.home.tmp_dir / "response.json").exists()
