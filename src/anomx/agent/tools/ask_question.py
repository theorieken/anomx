"""Ask-question tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.interactions import QuestionOption, QuestionRequest
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
        context.emit_operator_statement(self.name, arguments)
        if not context.runtime.agent_spec.can_ask_questions:
            return context.json_result(
                {"error": "This agent kind cannot ask the user questions."}
            )
        if context.callbacks.question is None:
            return context.json_result(
                {"answered": False, "cancelled": True, "error": "No interactive UI callback."}
            )

        request_or_error = self._question_request(arguments)
        if isinstance(request_or_error, str):
            return context.json_result(
                {"answered": False, "cancelled": True, "error": request_or_error}
            )

        response = context.callbacks.question(request_or_error)
        return context.json_result(
            {
                "answered": response.answered,
                "answer": response.answer,
                "selected_label": response.selected_label,
                "kind": response.kind or request_or_error.kind,
                "cancelled": response.cancelled,
            }
        )

    def _question_request(self, arguments: dict[str, Any]) -> QuestionRequest | str:
        question = str(arguments.get("question", "")).strip()
        if not question:
            return "ask_question requires a question."

        kind = str(arguments.get("kind", "text")).strip().lower()
        if kind not in {"select", "text", "confirm"}:
            return "ask_question kind must be select, text, or confirm."

        options = self._question_options(arguments.get("options"))
        if kind == "select" and not options and not bool(arguments.get("allow_custom", False)):
            return "select questions require options unless allow_custom is true."

        return QuestionRequest(
            question=question,
            kind=kind,
            options=options,
            placeholder=str(arguments.get("placeholder") or "").strip(),
            default=str(arguments.get("default") or "").strip(),
            allow_custom=bool(arguments.get("allow_custom", False)),
        )

    def _question_options(self, raw_options: object) -> tuple[QuestionOption, ...]:
        if not isinstance(raw_options, list):
            return ()

        options: list[QuestionOption] = []
        for raw_option in raw_options:
            if not isinstance(raw_option, dict):
                continue
            label = str(raw_option.get("label", "")).strip()
            value = str(raw_option.get("value", "")).strip() or label
            if not label:
                continue
            options.append(
                QuestionOption(
                    label=label,
                    value=value,
                    description=str(raw_option.get("description", "")).strip(),
                )
            )
        return tuple(options)
