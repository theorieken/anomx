"""Ordered, structured final output for the platform presentation adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema


class ProduceOutputTool(BaseTool):
    """Deliver a final response through a platform-provided callback."""

    def __init__(self) -> None:
        super().__init__(
            name="produce_output",
            description=(
                "Deliver the final response in the Anomx Platform and finish this turn. "
                "Call only after the work is complete. Items render in the given order, except "
                "that a proposition renders after all other body items and references always "
                "render last. Include references whenever you used sources. "
                "text content: a Markdown string. object content: {object_reference: string}, "
                "showing the full object inline. objects content: an ordered array of object "
                "reference strings, shown as horizontally scrolling object cards. "
                "focus_object already displays the full object prominently beside the chat. "
                "Do not duplicate the currently focused object in object or objects items "
                "unless the user explicitly requests it. After focusing, use text and any "
                "necessary reference items; an object item is not required to finish. "
                "database content: {model_reference: string, query?: object, search?: string, "
                "view?: 'list'|'grid', title?: string}. The database queries live, authorized "
                "platform objects of that model; query uses the model's documented API list "
                "filters. Example: {model_reference: 'data_channel', search: 'temperature', "
                "view: 'list'}. Discover "
                "valid model references and filters first; never invent them. "
                "proposition content: {prompt: string, label: string, icon: string, "
                "description: string}. It offers "
                "the user one follow-up action as a button showing label and icon, where icon "
                "is an Untitled UI icon name in PascalCase. When the user clicks the button, "
                "the platform starts another round with prompt as a hidden instruction to you: "
                "the user never sees prompt and it looks as if you simply continue working. "
                "Write prompt as a complete, self-contained instruction for that follow-up "
                "work, and make label say what the click will do. Write description as one "
                "short, user-facing sentence (at most 300 characters) explaining the scope "
                "and useful outcome of this action; it appears in a hover tooltip. Do not "
                "just repeat the label or expose internal instructions. The platform also keeps "
                "the proposition as a recommendation on the user's home page. Offer one only when "
                "you are convinced that this concrete next step genuinely benefits the user, "
                "based on what they asked for and what you found. Never add one by default, to "
                "round off the output, as a generic offer of more help, or for something the "
                "user declined; most outputs need none. Include at most one "
                "proposition per call. It always renders as the second-to-last element: after "
                "all other body items and directly before the references, which stay last. "
                "Without references it is the last element. This order is intended; do not "
                "try to place the proposition elsewhere. "
                "Example, offering to schedule the finished work as a planned prompt: "
                "{kind: 'proposition', content: {label: 'Run this every morning', "
                "description: 'Schedule this analysis for 08:00 daily to track changes.', "
                "icon: 'ClockFastForward', prompt: 'Create a planned prompt that repeats this "
                "analysis every day at 08:00.'}}. "
                "reference content: {url: 'https://...', title?: string} for a website, or "
                "{object_reference: string, title?: string} for a platform source. "
                "Use real references returned by platform tools. Do not repeat the output "
                "in another final text response. This tool is unavailable in the CLI."
            ),
            parameters=object_schema(
                {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": object_schema(
                            {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "text",
                                        "object",
                                        "objects",
                                        "database",
                                        "proposition",
                                        "reference",
                                    ],
                                },
                                "content": {
                                "description": "Payload for this kind; see the tool instructions.",
                                    "anyOf": [
                                        {"type": "string"},
                                        {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "maxItems": 50,
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "object_reference": {"type": "string"},
                                                "model_reference": {"type": "string"},
                                                "url": {"type": "string"},
                                                "title": {"type": "string"},
                                                "query": {
                                                    "type": "object",
                                                    "additionalProperties": True,
                                                },
                                                "search": {"type": "string"},
                                                "view": {
                                                    "type": "string",
                                                    "enum": ["list", "grid"],
                                                },
                                                "prompt": {"type": "string"},
                                                "label": {"type": "string"},
                                                "icon": {"type": "string"},
                                                "description": {
                                                    "type": "string",
                                                    "maxLength": 300,
                                                    "description": (
                                                        "Short user-facing explanation of the "
                                                        "follow-up action for its hover tooltip."
                                                    ),
                                                },
                                            },
                                            "additionalProperties": False,
                                        },
                                    ],
                                },
                            },
                            ["kind", "content"],
                        ),
                    },
                },
                ["items"],
            ),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        callback = getattr(context.callbacks, "output_response", None)
        if not context.runtime.can_output_response() or callback is None:
            return context.json_result(
                {"ok": False, "error": "produce_output is only available in platform runs."}
            )
        items = arguments.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= 50:
            return context.json_result({"ok": False, "error": "items must contain 1 to 50 items."})
        for index, item in enumerate(items):
            error = self.validate_item(item)
            if error:
                return context.json_result({"ok": False, "error": f"items[{index}]: {error}"})
        if sum(item["kind"] == "proposition" for item in items) > 1:
            return context.json_result(
                {"ok": False, "error": "items may contain at most one proposition."}
            )
        # Stable partition: never change the order within the body or references.
        ordered = [item for item in items if item["kind"] not in ("proposition", "reference")]
        ordered.extend(item for item in items if item["kind"] == "proposition")
        ordered.extend(item for item in items if item["kind"] == "reference")
        callback({"items": ordered, "end_turn": True})
        context.runtime.produced_output = "\n\n".join(
            item["content"] for item in ordered if item["kind"] == "text"
        )
        if context.session_path is not None:
            context.runtime.home.append_session_event(
                context.session_path, "produced_output", {"items": ordered}
            )
        return context.json_result({"ok": True, "end_turn": True, "item_count": len(items)})

    @staticmethod
    def validate_item(item: object) -> str | None:
        """Validate payloads before invoking the persistence adapter."""
        if not isinstance(item, dict) or set(item) != {"kind", "content"}:
            return "Each item requires exactly kind and content."
        kind, content = item["kind"], item["content"]
        if not isinstance(kind, str):
            return "kind must be a string."
        if kind == "text":
            return (
                None
                if isinstance(content, str) and content.strip()
                else "text requires nonempty Markdown."
            )
        if kind == "objects":
            return (
                None
                if isinstance(content, list)
                and 1 <= len(content) <= 50
                and all(isinstance(ref, str) and ref.strip() for ref in content)
                else "objects requires 1 to 50 object reference strings."
            )
        if not isinstance(content, dict):
            return f"{kind} requires an object as content."
        allowed_fields = {
            "object": {"object_reference"},
            "database": {"model_reference", "query", "search", "view", "title"},
            "proposition": {"prompt", "label", "icon", "description"},
            "reference": {"object_reference", "url", "title"},
        }
        if kind not in allowed_fields or set(content) - allowed_fields[kind]:
            return "Unsupported kind or content fields."
        if "title" in content and not isinstance(content["title"], str):
            return "title must be a string."
        if kind == "object":
            return (
                None
                if isinstance(content.get("object_reference"), str)
                and content["object_reference"].strip()
                else "object_reference is required."
            )
        if kind == "database":
            if (
                not isinstance(content.get("model_reference"), str)
                or not content["model_reference"].strip()
            ):
                return "database requires model_reference."
            if not isinstance(content.get("query", {}), dict) or content.get(
                "view", "list"
            ) not in ("list", "grid"):
                return "Invalid database query or view."
            if not isinstance(content.get("search", ""), str):
                return "database search must be a string."
            for value in content.get("query", {}).values():
                values = value if isinstance(value, list) else [value]
                if any(
                    entry is not None and not isinstance(entry, (str, int, float, bool))
                    for entry in values
                ):
                    return "database query values must be scalars or lists of scalars."
            return None
        if kind == "proposition":
            if not all(
                isinstance(content.get(field), str) and content[field].strip()
                for field in ("prompt", "label", "icon", "description")
            ):
                return "proposition requires nonempty prompt, label, icon, and description strings."
            if len(content["description"]) > 300:
                return "proposition description must be at most 300 characters."
            return None
        if kind == "reference":
            ref, url = content.get("object_reference"), content.get("url")
            if isinstance(ref, str) and ref.strip() and not url:
                return None
            if isinstance(url, str) and not ref:
                try:
                    parsed = urlparse(url)
                except ValueError:
                    return "Invalid reference URL."
                if (
                    parsed.scheme in ("https", "http")
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                ):
                    return None
            return "reference requires either object_reference or an HTTP(S) URL."
        return "Unsupported kind."
