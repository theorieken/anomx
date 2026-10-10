"""Show an object card during work without finishing the platform turn."""

from __future__ import annotations

import re
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class ShowObjectTool(BaseTool):
    """Display one useful intermediate result in the ongoing work section."""

    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="show_object",
            description=(
                "Show an existing Anomx object as a card while you continue working. "
                "Use this for a dataset, training run, model or finding worth showing now. "
                "The card stays in this run's work section after completion. It does not "
                "finish the turn or replace the focused object. Platform only. "
                "Use a verified canonical object_reference and avoid repeated cards."
            ),
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "object_reference": {
                        "type": "string",
                        "description": "Existing canonical object reference.",
                    },
                },
                ["statement", "object_reference"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "output_response", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result(
                {"ok": False, "error": "show_object is only available in platform runs."}
            )
        reference = arguments.get("object_reference")
        if not isinstance(reference, str) or not re.fullmatch(
            r"[a-zA-Z0-9_]+-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", reference.strip()
        ):
            return context.json_result(
                {"ok": False, "error": "Provide an existing canonical object_reference."}
            )
        payload = {
            "end_turn": False,
            "outputs": [{"type": "object_card", "object_reference": reference.strip()}],
        }
        try:
            callback(payload)
        except ValueError as error:
            return context.json_result({"ok": False, "error": str(error)})
        context.emit_operator_statement(self.name, arguments)
        if context.session_path is not None:
            context.runtime.home.append_session_event(context.session_path, "shown_object", payload)
        return context.json_result(
            {"ok": True, "end_turn": False, "object_reference": reference.strip()}
        )
