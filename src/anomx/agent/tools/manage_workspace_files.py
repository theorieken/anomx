"""Bounded note writes and scratch cleanup, including unattended agent runs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from anomx.agent.base.tools import BaseTool, ToolExecutionContext


class ManageWorkspaceFilesTool(BaseTool):
    def __init__(self, *, agent_tasks: bool = False) -> None:
        super().__init__(
            name="manage_workspace_files",
            description=(
                "Write compact UTF-8 notes/checkpoints in brain or intermediate files in tmp, "
                "or delete an exact tmp file. Available in Background mode without shell "
                "approval. Use relative paths. Existing files require their current SHA-256 "
                "as expected_sha256, so another chat's changes are preserved. On conflict, "
                "read and reconcile the current file before retrying. No recursive deletion."
                + (
                    " Agent users may also write task notes with area=tasks and "
                    "path=<task-id>/README.md (or another file inside an existing task folder)."
                    if agent_tasks
                    else ""
                )
            ),
            parameters={
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "enum": ["brain", "tmp", *(["tasks"] if agent_tasks else [])],
                    },
                    "action": {"type": "string", "enum": ["write", "delete"]},
                    "path": {"type": "string"},
                    "content": {
                        "type": "string",
                        "description": "Text for write; empty for delete.",
                    },
                    "expected_sha256": {"type": ["string", "null"]},
                },
                "required": ["area", "action", "path", "content", "expected_sha256"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        import fcntl

        area, action = arguments.get("area"), arguments.get("action")
        raw = str(arguments.get("path") or "")
        if "\x00" in raw or len(raw) > 512:
            return context.json_result(
                {"error": "Provide a valid relative path under 512 characters."}
            )
        relative = Path(raw)
        areas = {"brain", "tmp", "tasks"} if context.runtime.agent_user else {"brain", "tmp"}
        if area not in areas or action not in {"write", "delete"}:
            return context.json_result({"error": "Choose an available area and write/delete."})
        if (
            not raw
            or relative.is_absolute()
            or any(part.startswith(".") for part in relative.parts)
        ):
            return context.json_result(
                {"error": "Use a relative path without hidden or parent directories."}
            )
        if action == "delete" and area != "tmp":
            return context.json_result(
                {"error": "Cleanup is limited to exact files in this chat's tmp folder."}
            )
        if area == "tasks":
            root = context.runtime.workspace_root / "tasks"
            try:
                task_id = str(UUID(relative.parts[0]))
            except ValueError:
                return context.json_result({"error": "Use <task-id>/<filename> for task notes."})
            if (
                len(relative.parts) < 2
                or relative.parts[0] != task_id
                or root.resolve() != root
                or not (root / task_id).is_dir()
            ):
                return context.json_result({"error": "Write inside an existing task folder."})
        else:
            root = (
                context.runtime.home.brain_dir if area == "brain" else context.runtime.home.tmp_dir
            ).resolve()
        target = root / relative
        if target.resolve() != target or not target.is_relative_to(root) or target == root:
            return context.json_result(
                {"error": "The file must stay inside the selected directory without symlinks."}
            )
        content = str(arguments.get("content") or "").encode("utf-8")
        if len(content) > 524_288:
            return context.json_result({"error": "Keep notes and scratch files under 512 KiB."})
        root.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            root.parent / f".{root.name}-workspace-files.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if target.resolve() != target or (target.exists() and not target.is_file()):
                return context.json_result({"error": "The target must be a regular file."})
            digest = None
            if target.exists():
                with target.open("rb") as existing:
                    digest = hashlib.file_digest(existing, "sha256").hexdigest()
            if arguments.get("expected_sha256") != digest:
                return context.json_result(
                    {
                        "error": "File changed; read and reconcile it first.",
                        "current_sha256": digest,
                    }
                )
            if action == "delete":
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        output.write(content)
                        output.flush()
                        os.fsync(output.fileno())
                    temporary.replace(target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        return context.json_result(
            {
                "path": str(target),
                "action": action,
                "completed": True,
                "sha256": hashlib.sha256(content).hexdigest() if action == "write" else None,
            }
        )
