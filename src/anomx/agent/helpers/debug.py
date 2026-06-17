"""Debug step logger for the Anomx CLI agent.

Structures debug output as a session-oriented directory tree:

    {debug_location}/
      debug/
        {session_id}/
          meta.json
          steps/
            step-1.json
            step-2.json
            ...
          subagents/
            {agent_id}/
              steps/
                step-1.json
                step-2.json
                ...
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from anomx.agent.helpers.utils import utc_now_iso


def _message_content_from_item(item: dict[str, Any]) -> str:
    """Extract a human-readable content string from a message item.

    Handles content that may be a plain string, a list of content blocks,
    or tool_use / tool_result blocks.
    """
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", ""))
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    parts.append(text)
            elif block_type == "tool_use":
                name = str(block.get("name", ""))
                tool_input = str(block.get("input", ""))
                parts.append(f"[tool_use: {name}({tool_input})]")
            elif block_type == "tool_result":
                result = str(block.get("content", ""))
                parts.append(f"[tool_result: {result}]")
            elif block_type == "thinking":
                thinking = str(block.get("thinking", ""))
                if thinking:
                    parts.append(f"[thinking: {thinking[:200]}...]")
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        tool_use_id = item.get("id", item.get("tool_use_id", ""))
        name = item.get("name", "")
        tool_input = str(item.get("input", "{}"))
        if name or tool_use_id:
            return f"[tool_use: {name}({tool_input})]"
    return str(content or "")


class SessionDebugLogger:
    """Persists per-session step snapshots for debug inspection.

    Each step captures the normalized conversation messages sent to the AI
    backend together with metadata such as model, provider, and timestamp.
    """

    def __init__(self, debug_location: Path) -> None:
        self._debug_location = debug_location.expanduser().resolve()
        self._step_counters: dict[str, int] = {}
        self._subagent_step_counters: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_payload_messages(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Normalise provider-specific request payloads to ``[{role, content}]``.

        Handles the varying payload shapes used by OpenAI (``input`` /
        ``previous_response_id``), Anthropic / DESY (``system`` + ``messages``),
        and Ollama (``messages``).
        """
        messages: list[dict[str, Any]] = []

        system = payload.get("system")
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system})

        instructions = payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            messages.append({"role": "system", "content": instructions})

        for field in ("messages", "input"):
            raw = payload.get(field)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, dict):
                    continue
                if "role" in item:
                    role = str(item.get("role", "user"))
                    content = _message_content_from_item(item)
                    messages.append({"role": role, "content": content})
                elif "type" in item:
                    role = str(item.get("type", "user"))
                    messages.append({"role": role, "content": str(item)})

        return messages

    def write_step(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        provider: str = "",
        subagent_id: str | None = None,
    ) -> Path | None:
        """Write a single step file for a (sub)agent conversation.

        Parameters
        ----------
        session_id:
            The parent session identifier.
        messages:
            Normalised conversation history as ``[{role, content}, ...]``.
        model:
            AI model name used for this request.
        provider:
            AI provider key (e.g. ``"anthropic"``).
        subagent_id:
            When set, the step is written under ``subagents/{subagent_id}/steps/``.

        Returns the path that was written, or ``None`` on failure.
        """
        try:
            if subagent_id:
                key = (session_id, subagent_id)
                self._subagent_step_counters.setdefault(key, 0)
                self._subagent_step_counters[key] += 1
                step = self._subagent_step_counters[key]
                step_dir = (
                    self._sessions_dir
                    / session_id
                    / "subagents"
                    / subagent_id
                    / "steps"
                )
            else:
                self._step_counters.setdefault(session_id, 0)
                self._step_counters[session_id] += 1
                step = self._step_counters[session_id]
                step_dir = self._sessions_dir / session_id / "steps"

            step_dir.mkdir(parents=True, exist_ok=True)
            path = step_dir / f"step-{step}.json"

            payload: dict[str, Any] = {
                "step": step,
                "timestamp": utc_now_iso(),
                "model": model,
                "provider": provider,
            }
            if subagent_id is not None:
                payload["subagent_id"] = subagent_id
            payload["messages"] = messages

            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            return path
        except OSError:
            return None

    def write_subagent_step(
        self,
        session_id: str,
        agent_id: str,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        provider: str = "",
    ) -> Path | None:
        """Convenience wrapper around *write_step* for subagent conversations."""
        return self.write_step(
            session_id,
            messages,
            model=model,
            provider=provider,
            subagent_id=agent_id,
        )

    def ensure_session(
        self,
        session_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the session directory and initialise *meta.json*.

        If *meta.json* already exists it is left unchanged so that existing
        metadata is never overwritten.
        """
        meta_dir = self._sessions_dir / session_id
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "meta.json"
        if meta_path.exists():
            return
        payload = dict(metadata) if metadata is not None else {}
        payload.setdefault("session_id", session_id)
        payload.setdefault("created_at", utc_now_iso())
        payload.setdefault("updated_at", payload["created_at"])
        meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def update_meta(
        self,
        session_id: str,
        updates: Mapping[str, Any],
    ) -> None:
        """Merge *updates* into the session's *meta.json*."""
        meta_dir = self._sessions_dir / session_id
        meta_path = meta_dir / "meta.json"
        if not meta_path.exists():
            self.ensure_session(session_id, dict(updates))
            return
        try:
            existing: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        existing["updated_at"] = utc_now_iso()
        existing.update(updates)
        meta_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _sessions_dir(self) -> Path:
        path = self._debug_location / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path
