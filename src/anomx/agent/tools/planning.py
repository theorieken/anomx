"""Planning tools."""

from __future__ import annotations

from anomx.agent.base.tools import BaseTool, object_schema, statement_property


def plan_schema(*, require_position: bool) -> dict[str, object]:
    properties: dict[str, dict[str, str]] = {
        "title": {
            "type": "string",
            "description": "Short user-visible plan item title.",
        },
        "description": {
            "type": "string",
            "description": "Private operator-facing detail for this step.",
        },
        "is_done": {
            "type": "boolean",
            "description": "Whether this step is complete.",
        },
    }
    required = ["title", "description", "is_done"]
    if require_position:
        properties = {
            "position": {
                "type": "integer",
                "description": "One-based plan position.",
            },
            **properties,
        }
        required = ["position", *required]
    return object_schema(
        {
            "statement": statement_property(
                "Persistent user-visible working message for this tool call."
            ),
            "steps": {
                "type": "array",
                "description": "Ordered plan steps.",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        ["statement", "steps"],
    )


class CreatePlanTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="create_plan",
            description="Create a user-visible ordered plan.",
            parameters=plan_schema(require_position=False),
        )


class UpdatePlanTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="update_plan",
            description="Update the user-visible ordered plan.",
            parameters=plan_schema(require_position=True),
        )


class RemovePlanTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="remove_plan",
            description="Clear the current user-visible plan.",
            parameters=object_schema(
                {"statement": statement_property(statement_description)},
                ["statement"],
            ),
        )


class FinishAnywaysTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="finish_anyways",
            description=(
                "Clear the current user-visible plan and allow final delivery after "
                "the plan-finish checker asks for an explicit override."
            ),
            parameters=object_schema(
                {"statement": statement_property(statement_description)},
                ["statement"],
            ),
        )
