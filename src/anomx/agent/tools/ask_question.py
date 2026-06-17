"""Ask-question tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property


class AskQuestionTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="ask_question",
            description="Ask the user an interactive question in the bottom panel.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "question": {
                        "type": "string",
                        "description": "The concise user-facing question.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["select", "text", "confirm"],
                        "description": (
                            "select uses arrow-key options, text allows typing, "
                            "confirm asks a yes/no question."
                        ),
                    },
                    "options": {
                        "type": "array",
                        "description": "Predefined choices for select questions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "User-visible option label.",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Value returned to the agent.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Short option detail.",
                                },
                            },
                            "required": ["label", "value", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "placeholder": {
                        "type": ["string", "null"],
                        "description": "Placeholder shown for text input, or null.",
                    },
                    "default": {
                        "type": ["string", "null"],
                        "description": "Default response value, or null.",
                    },
                    "allow_custom": {
                        "type": "boolean",
                        "description": (
                            "For select questions, also allow a typed custom answer."
                        ),
                    },
                },
                [
                    "statement",
                    "question",
                    "kind",
                    "options",
                    "placeholder",
                    "default",
                    "allow_custom",
                ],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.runtime._emit_operator_tool_statement(
            self.name, arguments, context.callbacks
        )
        if not context.runtime.agent_spec.can_ask_questions:
            return context.json_result(
                {"error": "This agent kind cannot ask the user questions."}
            )
        return context.runtime._ask_question_tool(arguments, context.callbacks)
