"""Permission-checked email delivery through the connected platform."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema
from anomx.agent.tools.use_anomx_api import UseAnomxApiTool


class SendEmailTool(BaseTool):
    """Send as the autonomous account without exposing SMTP credentials."""

    def __init__(self) -> None:
        super().__init__(
            name="send_email",
            description=(
                "Send an email as your autonomous agent account using the platform sender. "
                "Requires Write E-Mails permission. Use a stable delivery_key for this message "
                "and reuse it with identical content for retries. Never claim delivery unless "
                "the response says sent=true. An uncertain delivery must not be retried with "
                "a new key without checking whether it arrived. Recipient and message content "
                "must follow your assigned task and guardrails."
            ),
            parameters=object_schema(
                {
                    "recipient": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "maxLength": 255},
                    "body": {
                        "type": "string",
                        "maxLength": 100000,
                        "description": "Plain-text message.",
                    },
                    "delivery_key": {
                        "type": "string",
                        "maxLength": 200,
                        "description": "Stable identifier for this exact message.",
                    },
                },
                ["recipient", "subject", "body", "delivery_key"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        if not getattr(context.runtime, "agent_user", False):
            return context.json_result(
                {"ok": False, "error": "Only autonomous agent accounts may send emails."}
            )
        limits = {"recipient": 254, "subject": 255, "body": 100000, "delivery_key": 200}
        for field, limit in limits.items():
            value = arguments.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                return context.json_result(
                    {
                        "ok": False,
                        "error": f"{field} must be nonempty text of at most {limit} characters.",
                    }
                )
        if any(
            character in arguments[field]
            for field in ("recipient", "subject")
            for character in "\r\n"
        ):
            return context.json_result(
                {"ok": False, "error": "Recipient and subject must each be a single line."}
            )
        return UseAnomxApiTool(statement_description="Sending email").execute(
            {
                "method": "POST",
                "path": "/agents/emails",
                "statement": "Sending email",
                "body": {field: arguments[field] for field in limits},
            },
            context,
        )
