"""Manage the host's conversation tabs without depending on platform persistence."""

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ChatTabTool(BaseTool):
    def __init__(self, operation: str, *, statement_description: str) -> None:
        self.operation = operation
        descriptions = {
            "list": "List the conversation's open object tabs and the selected tab. Read this before selecting or closing a tab.",
            "open": "Open an accessible object beside the chat, or select its existing tab. Always open verified objects when discussing, inspecting, creating or updating them so the user can follow along. Other tabs stay open.",
            "select": "Select an existing tab by its id returned by list_tabs. The user sees its full object beside the conversation.",
            "close": "Close a tab by its id returned by list_tabs. This only removes the tab, never its object.",
        }
        fields = {"statement": statement_property(statement_description)}
        required = ["statement"]
        if operation == "open":
            fields["object_reference"] = {"type": "string", "description": "Canonical object reference returned by platform tools, never a URL or guessed id."}
            required.append("object_reference")
        elif operation in {"select", "close"}:
            fields["tab_id"] = {"type": "string", "description": "An open tab id returned by list_tabs."}
            required.append("tab_id")
        super().__init__(name="list_tabs" if operation == "list" else f"{operation}_tab", description=descriptions[operation] + " Platform chats only; does not save or publish object content.", parameters=object_schema(fields, required))

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "manage_tabs", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result({"ok": False, "error": "Tabs are only available in platform chats."})
        payload = {"operation": self.operation}
        key = "object_reference" if self.operation == "open" else "tab_id" if self.operation in {"select", "close"} else None
        if key:
            value = arguments.get(key)
            if not isinstance(value, str) or not value.strip():
                return context.json_result({"ok": False, "error": f"Provide {key} returned by the platform."})
            payload[key] = value.strip()
        try:
            result = callback(payload)
        except ValueError as error:
            return context.json_result({"ok": False, "error": str(error)})
        context.emit_operator_statement(self.name, arguments)
        return context.json_result({"ok": True, **result})
