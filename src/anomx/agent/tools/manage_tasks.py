"""Persistent self-managed tasks for autonomous platform users."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema
from anomx.agent.tools.use_anomx_api import UseAnomxApiTool


class ManageTasksTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="manage_tasks",
            description=(
                "Manage your own persistent tasks: list, add, edit or delete. Agent users only. "
                "Add requires a stable key and title; retries with the same key reuse the "
                "task. Edit updates title, description, completed or reminders. Omitted fields "
                "remain unchanged. reminders replaces the list; [] removes all reminders. "
                "Keep descriptions short and detailed content in the returned tasks/<id>/ folder. "
                "Tasks are refreshed in your system prompt before every model request. Completed "
                "tasks suspend reminders; recurring tasks normally remain open. Delete cancels all "
                "reminders but retains files. Reminders can wake you outside working hours; "
                "pauses still apply. Use ISO dates and IANA zones. Translate natural-language "
                "reminders into once/daily/weekly. Check success before claiming a task was saved."
            ),
            parameters=object_schema(
                {
                    "action": {"type": "string", "enum": ["list", "add", "edit", "delete"]},
                    "task_id": {"type": "string", "description": "Task UUID for edit/delete."},
                    "key": {
                        "type": "string",
                        "maxLength": 128,
                        "description": "Stable idempotency key for add.",
                    },
                    "title": {"type": "string", "maxLength": 200},
                    "description": {"type": "string", "maxLength": 1000},
                    "completed": {"type": "boolean"},
                    "reminders": {
                        "type": "array",
                        "maxItems": 20,
                        "items": object_schema(
                            {
                                "kind": {"type": "string", "enum": ["once", "daily", "weekly"]},
                                "timezone": {
                                    "type": "string",
                                    "description": "IANA timezone; defaults to your timezone.",
                                },
                                "at": {
                                    "type": "string",
                                    "description": (
                                        "Once: ISO date/time, e.g. 2027-03-12T12:00. "
                                        "An explicit offset is also accepted."
                                    ),
                                },
                                "time": {
                                    "type": "string",
                                    "description": "Daily/weekly: local HH:MM, e.g. 08:00.",
                                },
                                "weekdays": {
                                    "type": "array",
                                    "items": {"type": "integer", "minimum": 0, "maximum": 6},
                                    "description": "Weekly only: 0=Monday through 6=Sunday.",
                                },
                            },
                            ["kind"],
                        ),
                    },
                },
                ["action"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        if not getattr(context.runtime, "agent_user", False):
            return context.json_result(
                {"ok": False, "error": "This tool requires an autonomous agent user."}
            )
        action = arguments.get("action")
        if not isinstance(action, str) or action not in {"list", "add", "edit", "delete"}:
            return context.json_result({"ok": False, "error": "Unknown task action."})
        body = {
            key: value
            for key, value in arguments.items()
            if key in {"action", "task_id", "key", "title", "description", "completed", "reminders"}
        }
        if action in {"edit", "delete"}:
            try:
                body["task_id"] = str(UUID(str(body.get("task_id", ""))))
            except ValueError:
                return context.json_result({"ok": False, "error": "A valid task_id is required."})
        request: dict[str, Any] = {
            "method": "GET" if action == "list" else "POST",
            "path": "/users/agent-tasks",
            "statement": f"Managing my tasks: {action}",
        }
        if action != "list":
            request["body"] = body
        return UseAnomxApiTool(statement_description="Managing my tasks").execute(request, context)
