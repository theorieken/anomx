"""Deliver a workspace artifact through an explicitly attached platform host."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class SendFileTool(BaseTool):
    """Publish a generated file without embedding its bytes in the model context."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="send_file",
            description=(
                "Send a finished file from this chat's task, tmp or sub_agents directory "
                "to the user. The platform stores a private copy in their file library and "
                "shows a downloadable file in this chat. Use this for generated documents, "
                "images, plots, archives and data exports; a runtime path alone is not a "
                "user-accessible download. Maximum 64 MiB, subject to the user's upload quota. "
                "Never send credentials or internal configuration. Retries with identical "
                "content and filename in this turn reuse the same file. Include the returned "
                "object_reference as an object item in produce_output for the final answer. "
                "Only claim delivery when ok=true. This tool does not finish the turn."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "path": {
                        "type": "string",
                        "description": "Path of the finished workspace file.",
                    },
                    "filename": {
                        "type": "string",
                        "maxLength": 255,
                        "description": "Optional download filename, including its extension.",
                    },
                    "description": {
                        "type": "string",
                        "maxLength": 2000,
                        "description": "Optional short description of the file.",
                    },
                },
                ["statement", "path"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "send_file", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result(
                {
                    "ok": False,
                    "error": "send_file requires a platform chat with file delivery support.",
                }
            )
        path = context.workspace_path(arguments.get("path"))
        if isinstance(path, str):
            return context.json_result({"ok": False, "error": path})
        if not path.is_file():
            return context.json_result(
                {"ok": False, "error": "Path must be an existing regular file."}
            )
        for field, limit in (("filename", 255), ("description", 2000)):
            value = arguments.get(field, "")
            if not isinstance(value, str) or len(value) > limit:
                return context.json_result(
                    {"ok": False, "error": f"{field} must be text of at most {limit} characters."}
                )
        filename = arguments.get("filename") or path.name
        if filename in {".", ".."} or any(
            char in filename for char in ("/", "\\", "\x00", "\r", "\n")
        ):
            return context.json_result(
                {"ok": False, "error": "filename must be a filename, without directory components."}
            )
        # File creation uses the same mode policy as other persistent platform writes.
        authorization = context.runtime.tool_manager.authorize_platform_api_request(
            "POST",
            "/files/upload",
            str(arguments.get("statement") or "Sending file"),
            context.callbacks.approval,
            body={"filename": filename},
        )
        if authorization is not None:
            return context.json_result(
                {
                    "ok": False,
                    "approved": False,
                    "blocked_by_mode": authorization.blocked_by_mode,
                    "error": authorization.output,
                }
            )
        context.emit_operator_statement(self.name, arguments)
        try:
            result = callback(
                {
                    "path": str(path),
                    "filename": filename,
                    "description": arguments.get("description", ""),
                }
            )
        except (OSError, ValueError) as error:
            return context.json_result({"ok": False, "error": str(error)})
        if (
            not isinstance(result, dict)
            or result.get("ok") is not True
            or not result.get("object_reference")
        ):
            return context.json_result(
                {
                    "ok": False,
                    "error": str(result.get("error") or "File delivery was not confirmed.")
                    if isinstance(result, dict)
                    else "File delivery was not confirmed.",
                }
            )
        return context.json_result(result)
