"""Explicit, task-scoped permission to mount a host folder in the CLI runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.container_runtime import SandboxStartupError
from anomx.agent.helpers.tool_manager import ApprovalChoice, CommandApprovalRequest


class MountFolderTool(BaseTool):
    """Keep host access separate from ordinary command approvals and agent mode."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="mount_folder",
            description=(
                "Ask the user to allow read/write access to a specific absolute host folder "
                "for this task. Use only when their request requires files outside the isolated "
                "task folder. Explain why and ask first when the need is unclear. A listing of "
                "the launch directory is context, not permission to mount it. Access is never "
                "automatic, even in Autonomous mode."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {"type": "string", "description": "Full absolute host folder path."},
                    "reason": {"type": "string", "description": "Why this task needs this folder."},
                },
                ["statement", "path", "reason"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        sandbox = context.runtime.sandbox_session
        if sandbox is None or context.runtime.platform_chat_id:
            return context.json_result(
                {"approved": False, "error": "Only the CLI container supports host mounts."}
            )
        raw = str(arguments.get("path", ""))
        reason = str(arguments.get("reason", "")).strip()
        try:
            if not raw or not reason:
                raise SandboxStartupError(
                    "A full host folder path and a task-specific reason are required."
                )
            path = sandbox.validate_mount_folder(Path(raw))
            already_approved = path in sandbox.approved_folders.values()
        except (OSError, ValueError, SandboxStartupError) as error:
            return context.json_result({"approved": False, "error": str(error)})
        if context.runtime.cancel_event.is_set() or (
            not already_approved and context.callbacks.approval is None
        ):
            return context.json_result(
                {
                    "approved": False,
                    "error": "Mounting a host folder requires explicit user approval; "
                    "no access was granted.",
                }
            )
        if not already_approved:
            request = CommandApprovalRequest(
                command=f"mount_folder {path}",
                canonical_command=f"mount_folder {path}",
                statement="Allow folder access for this task",
                reason=(
                    f"Allow read/write access to the host folder {path} for this task? {reason}\n"
                    "The grant is remembered when this task resumes. Other tasks require their own "
                    "approval. Changing the active task's mounts may replace the idle anomx-agent "
                    "container; task files are preserved."
                ),
                requires_user_approval=True,
            )
            decision = context.callbacks.approval(request)
            if (
                decision not in {ApprovalChoice.ALLOW, ApprovalChoice.ALWAYS_ALLOW}
                or context.runtime.cancel_event.is_set()
            ):
                return context.json_result(
                    {"approved": False, "error": "Folder access was not granted."}
                )
        try:
            target = sandbox.mount_folder(path, cancelled=context.runtime.cancel_event.is_set)
            context.runtime.tool_manager.set_trusted_roots(context.runtime.trusted_roots)
            context.runtime.tool_manager.path_aliases.update(sandbox.path_aliases)
        except (OSError, ValueError, SandboxStartupError) as error:
            return context.json_result({"approved": False, "error": str(error)})
        context.emit_operator_statement(
            self.name, arguments, default_statement="Mounted requested folder"
        )
        return context.json_result(
            {
                "approved": True,
                "host_path": str(path),
                "path": str(target),
                "access": "read_write",
                "scope": "current_task",
            }
        )
