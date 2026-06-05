"""Model backend and local tool runtime for the Anomx CLI agent."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, MutableSet
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from anomx.agent.mode import AgentMode
from anomx.agent.state import (
    build_plan_steps,
    latest_plan_steps,
    merge_plan_steps,
    serialize_plan_steps,
    worker_snapshots,
)
from anomx.agent.store import AnomxHome, model_metadata, utc_now_iso
from anomx.agent.tool_manager import (
    ApprovalCallback,
    CliToolManager,
    CommandResult,
    CommandSafety,
)

StatusCallback = Callable[[str], None]
MessageCallback = Callable[[str], None]
ToolMessageCallback = Callable[[str], None]
DeltaCallback = Callable[[str], None]
SystemMessageCallback = Callable[[str, str], None]
CommandCallback = Callable[[str, str, str], None]

OPERATOR_SYSTEM_PROMPT = """\
You are the Anomx Operator agent.

Role:
- You are always the agent in contact with the user.
- First decide whether the task is simple enough to answer directly or complex enough
  to need an explicit plan. For complex work, create a plan with create_plan, then move
  into execution. A plan is not a stopping point.
- Manage the work deliberately: delegate focused subtasks to Worker agents, monitor their
  progress, update the plan, validate important results yourself, and synthesize the final
  answer for the user.
- Worker agents run in background threads and report back as system messages. Running
  workers are listed in your runtime context with their id, name, current statement, and
  runtime duration.
- Default delegation policy: if the user asks for implementation, repository creation,
  code changes, multi-file investigation, or validation, start at least one Worker agent
  with start_agent unless the task is trivial enough to complete with one or two direct
  read-only commands. Good worker names include Engineer, Reviewer, Researcher, and
  Designer.
- After creating a plan for a non-trivial task, immediately start the first Worker agent
  in the same tool loop whenever possible. Do not end your response after only creating
  or describing the plan.
- Start independent research, implementation, or review tasks with start_agent. Use
  prompt_agent to start a new worker or to prompt a finished worker again. Use stop_agent
  when a worker is no longer useful or is clearly off track.
- When workers are still running and you need their results, use wait. While waiting you
  may still send output_message updates, inspect worker progress with check_agent, validate
  state with run_command, or stop a worker.
- The wait tool has no arguments and is only available while at least one Worker is
  running. It waits up to 60 seconds, but returns early as soon as all workers finish.
- Use check_agent when you need to inspect a Worker's command/statement thread before
  deciding whether to wait longer, prompt it again, or stop it.
- Use remove_plan when starting a new unrelated task and the previous plan no longer
  describes the active work.
- Do not produce a final answer while required workers are still running. Use wait, review
  their reports, update the plan, and then finalize.
- Use run_command(statement, command) for your own validation, inspection, review, and
  final checks. Every Operator tool except output_message and wait requires statement.
  The statement is a persistent working message visible to the user, for example
  "Checking repository state" or "Starting Engineer Worker".
- Confirm Mode does not mean "ask the user in prose before doing work." If a command needs
  approval, call run_command or start a Worker anyway; the command approval UI will ask the
  user at the moment approval is required.
- Make pragmatic default choices for ordinary scaffolding details such as folder names,
  package managers, and starter options. Ask the user only when a missing detail would make
  the work destructive, ambiguous in a high-impact way, or impossible to validate.
- Keep the user updated with output_message for multi-stage work, but do not over-narrate.
- Final answers should state the outcome, important changes or findings, validation, and
  any residual risk. Do not prefix messages with "Agent:" or "You:".
"""

WORKER_SYSTEM_PROMPT = """\
You are an Anomx Worker agent.

Role:
- You are prompted only by the Operator agent and are not in direct contact with the user.
- Your job is to complete the specific request from the Operator using run_command.
- Keep using run_command successively until the request is fulfilled or clearly blocked.
- Every run_command call must include a concise statement that describes the current action;
  the Operator sees this as your current worker status.
- If a command needs user approval, still call run_command. The approval UI handles that
  flow; do not stop with a prose request for permission.
- Return a concise final report to the Operator with concrete findings, files changed if
  relevant, validation performed, and any blockers. Do not ask the user questions.
"""

OPERATOR_TOOL_DESCRIPTIONS = (
    "output_message(message): send a concise working update to the user.",
    (
        "run_command(statement, command): run a safe CLI command for operator validation "
        "or inspection and persist statement as a working message."
    ),
    "start_agent(statement, name, prompt): start a background Worker agent for a focused task.",
    (
        "prompt_agent(statement, agent_id, name, prompt): prompt a finished Worker again, "
        "or start a new Worker."
    ),
    "check_agent(statement, agent_id): inspect a Worker agent's command/statement thread.",
    "stop_agent(statement, agent_id): request that a running Worker stop.",
    "create_plan(statement, steps): create a user-visible ordered plan.",
    "update_plan(statement, steps): update the user-visible ordered plan.",
    "remove_plan(statement): clear the current user-visible plan.",
    "wait(): wait up to 60 seconds for running Worker agents.",
)

WORKER_TOOL_DESCRIPTIONS = (
    (
        "run_command(statement, command): run a safe CLI command and publish statement "
        "as worker status."
    ),
)

MAX_TOOL_ITERATIONS = 128
OPENAI_MAX_TOOL_CALLS = 128
DESY_MESSAGES_ENDPOINT = "https://assistant.desy.de/api/v1/messages"


class AgentRole(StrEnum):
    """Runtime role for a model-backed agent."""

    OPERATOR = "operator"
    WORKER = "worker"


@dataclass
class WorkerAgentState:
    """Mutable in-process state for a background worker agent."""

    worker_id: str
    name: str
    prompt: str
    status: str
    statement: str
    started_at: str
    cancel_event: threading.Event
    finished_at: str = ""
    response: str = ""
    thread: threading.Thread | None = None
    command_history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeCallbacks:
    """UI callbacks used while a model response is in progress."""

    status: StatusCallback | None = None
    message: MessageCallback | None = None
    tool_message: ToolMessageCallback | None = None
    command: CommandCallback | None = None
    delta: DeltaCallback | None = None
    approval: ApprovalCallback | None = None
    system_message: SystemMessageCallback | None = None


@dataclass(frozen=True)
class OpenAIToolCall:
    """Function call emitted by the Responses API."""

    name: str
    call_id: str
    arguments: str


@dataclass(frozen=True)
class OpenAIStreamResponse:
    """Result collected from a streamed OpenAI response."""

    response_id: str | None
    text: str
    tool_calls: tuple[OpenAIToolCall, ...]


@dataclass(frozen=True)
class AnthropicToolCall:
    """Tool call emitted by the Anthropic Messages API."""

    name: str
    tool_use_id: str
    input: dict[str, Any]


@dataclass(frozen=True)
class AnthropicStreamResponse:
    """Result collected from a streamed Anthropic response."""

    text: str
    tool_calls: tuple[AnthropicToolCall, ...]
    content: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OllamaToolCall:
    """Function call emitted by Ollama chat responses."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class OllamaStreamResponse:
    """Result collected from a streamed Ollama response."""

    text: str
    thinking: str
    tool_calls: tuple[OllamaToolCall, ...]
    message: dict[str, Any]


class AgentRuntime:
    """Execute model requests and local tools for an agent session."""

    def __init__(
        self,
        home: AnomxHome,
        cwd: Path,
        session_allowed_commands: MutableSet[str] | None = None,
        session_rejected_commands: MutableSet[str] | None = None,
        mode: AgentMode = AgentMode.CONFIRM,
        role: AgentRole = AgentRole.OPERATOR,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.home = home
        self.cwd = cwd
        self.tool_manager = CliToolManager(
            cwd,
            session_allowed_commands,
            session_rejected_commands,
            mode,
        )
        self.session_allowed_commands = session_allowed_commands
        self.session_rejected_commands = session_rejected_commands
        self.role = role
        self.cancel_event = threading.Event() if cancel_event is None else cancel_event
        self._workers: dict[str, WorkerAgentState] = {}
        self._worker_lock = threading.Lock()

    def set_mode(self, mode: AgentMode) -> None:
        """Set the active command execution mode."""

        self.tool_manager.set_mode(mode)

    def backend_response(
        self,
        session_path: Path,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a backend response for the current session."""

        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        if provider == "openai":
            return self.openai_response(session_path, model, active_callbacks)
        if provider == "anthropic":
            return self.anthropic_response(session_path, model, active_callbacks)
        if provider == "desy":
            return self.desy_response(session_path, model, active_callbacks)
        if provider == "ollama":
            return self.ollama_response(session_path, model, active_callbacks)
        return f"{provider}/{model} backend is unavailable."

    def backend_response_for_prompt(
        self,
        prompt: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a response for a one-off worker prompt."""

        self.home.ensure()
        worker_dir = self.home.root / "worker_sessions"
        worker_dir.mkdir(parents=True, exist_ok=True)
        session_path = worker_dir / f"worker-{uuid4().hex}.jsonl"
        self.home.append_session_event(session_path, "user_message", {"message": prompt})
        return self.backend_response(session_path, callbacks=callbacks)

    def openai_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a response through the OpenAI Responses API."""

        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        api_key = self._api_key("openai", "OPENAI_API_KEY")
        if api_key is None:
            return self._missing_api_key_message("OpenAI", "OPENAI_API_KEY")

        self._status(active_callbacks.status)
        payload: dict[str, Any] = {
            "model": model,
            "instructions": self._instructions(session_path),
            "input": self.conversation_messages(session_path),
            "reasoning": {"summary": "auto"},
            "tools": self._openai_tools(),
            "tool_choice": "auto",
            "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
            "stream": True,
        }

        for _ in range(MAX_TOOL_ITERATIONS):
            self._status(active_callbacks.status)
            response = self._stream_openai_response(
                api_key,
                payload,
                active_callbacks.delta,
                active_callbacks.status,
            )
            if isinstance(response, str):
                return response

            tool_outputs = self._execute_requested_tools(
                response,
                active_callbacks,
                session_path,
            )
            if not tool_outputs:
                return response.text

            if response.response_id is None:
                return "OpenAI requested tools but did not return a response id."

            payload = {
                "model": model,
                "instructions": self._instructions(session_path),
                "previous_response_id": response.response_id,
                "input": tool_outputs,
                "reasoning": {"summary": "auto"},
                "tools": self._openai_tools(),
                "tool_choice": "auto",
                "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                "stream": True,
            }

        return f"OpenAI tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def anthropic_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a response through the Anthropic Messages API."""

        return self._messages_api_response(
            session_path,
            model,
            callbacks,
            provider_key="anthropic",
            provider_label="Anthropic",
            env_var="ANTHROPIC_API_KEY",
            stream_response=self._stream_anthropic_response,
            include_thinking=True,
        )

    def desy_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a response through the DESY Assistant Messages API."""

        return self._messages_api_response(
            session_path,
            model,
            callbacks,
            provider_key="desy",
            provider_label="DESY Assistant",
            env_var="DESY_ASSISTANT_API_KEY",
            stream_response=self._stream_desy_response,
            include_thinking=False,
        )

    def _messages_api_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None,
        *,
        provider_key: str,
        provider_label: str,
        env_var: str,
        stream_response: Callable[
            [str, dict[str, Any], DeltaCallback | None, StatusCallback | None],
            AnthropicStreamResponse | str,
        ],
        include_thinking: bool,
    ) -> str:
        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        api_key = self._api_key(provider_key, env_var)
        if api_key is None:
            return self._missing_api_key_message(provider_label, env_var)

        self._status(active_callbacks.status)
        messages = self._anthropic_messages(self.conversation_messages(session_path))
        payload: dict[str, Any] = {
            "model": model,
            "system": self._instructions(session_path),
            "messages": messages,
            "tools": self._anthropic_tools(),
            "max_tokens": self._max_output_tokens(model, 4_096),
            "stream": True,
        }
        if include_thinking:
            payload["thinking"] = self._anthropic_thinking_config(model)

        for _ in range(MAX_TOOL_ITERATIONS):
            self._status(active_callbacks.status)
            response = stream_response(
                api_key,
                payload,
                active_callbacks.delta,
                active_callbacks.status,
            )
            if isinstance(response, str):
                return response

            tool_outputs = self._execute_anthropic_requested_tools(
                response,
                active_callbacks,
                session_path,
            )
            if not tool_outputs:
                return response.text or self._extract_anthropic_text(response.content)

            messages.append({"role": "assistant", "content": list(response.content)})
            messages.append({"role": "user", "content": tool_outputs})
            payload = {
                "model": model,
                "system": self._instructions(session_path),
                "messages": messages,
                "tools": self._anthropic_tools(),
                "max_tokens": self._max_output_tokens(model, 4_096),
                "stream": True,
            }
            if include_thinking:
                payload["thinking"] = self._anthropic_thinking_config(model)

        return f"{provider_label} tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def ollama_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        """Generate a response through a local Ollama server."""

        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        messages = [
            {"role": "system", "content": self._instructions(session_path)},
            *self.conversation_messages(session_path),
        ]
        for _ in range(MAX_TOOL_ITERATIONS):
            response = self._stream_ollama_response(model, messages, active_callbacks)
            if isinstance(response, str):
                return response

            if response.message:
                messages.append(response.message)
            if not response.tool_calls:
                return response.text or "No response."

            messages.extend(
                self._execute_ollama_requested_tools(
                    response,
                    active_callbacks,
                    session_path,
                )
            )

        return f"Ollama tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def conversation_messages(self, session_path: Path) -> list[dict[str, str]]:
        """Return stored user/assistant messages for a backend conversation."""

        messages: list[dict[str, str]] = []
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type == "user_message" and message:
                messages.append({"role": "user", "content": message})
            elif event_type == "agent_message" and message:
                messages.append({"role": "assistant", "content": message})
            elif event_type == "system_message" and message:
                messages.append({"role": "system", "content": message})
        return messages[-20:]

    def suggest_session_title(self, session_path: Path) -> str | None:
        """Suggest a compact title for a session."""

        messages = self.conversation_messages(session_path)
        if not messages:
            return None

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        if provider == "openai":
            title = self._suggest_openai_session_title(messages, model)
            if title:
                return title
        elif provider == "anthropic":
            title = self._suggest_anthropic_session_title(messages, model)
            if title:
                return title
        elif provider == "desy":
            title = self._suggest_desy_session_title(messages, model)
            if title:
                return title
        elif provider == "ollama":
            title = self._suggest_ollama_session_title(messages, model)
            if title:
                return title
        return self._heuristic_session_title(messages)

    def extract_openai_text(self, data: dict[str, Any]) -> str:
        """Extract text from an OpenAI Responses API payload."""

        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
        return "\n".join(parts).strip() or "No response."

    def extract_anthropic_text(self, data: dict[str, Any]) -> str:
        """Extract text from an Anthropic Messages API payload."""

        content = data.get("content")
        if not isinstance(content, list):
            return "No response."
        return self._extract_anthropic_text(content)

    def _stream_openai_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> OpenAIStreamResponse | str:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_id: str | None = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[OpenAIToolCall] = []
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line.removeprefix("data:").strip()
                    if event_data == "[DONE]":
                        break
                    event = cast(dict[str, Any], json.loads(event_data))
                    event_type = str(event.get("type", ""))
                    if event_type == "response.output_text.delta":
                        delta = str(event.get("delta", ""))
                        if delta:
                            text_parts.append(delta)
                            if delta_callback is not None:
                                delta_callback(delta)
                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = str(event.get("delta", ""))
                        if delta:
                            reasoning_parts.append(delta)
                            self._reasoning_status(status_callback, "".join(reasoning_parts))
                    elif event_type == "response.output_item.done":
                        tool_call = self._tool_call_from_stream_item(event.get("item"))
                        if tool_call is not None:
                            tool_calls.append(tool_call)
                    elif event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            maybe_id = response_payload.get("id")
                            if isinstance(maybe_id, str):
                                response_id = maybe_id
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            return self._api_error("openai", "OpenAI", "OPENAI_API_KEY", error.code, error_body)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            return f"OpenAI request failed: {error}"
        return OpenAIStreamResponse(response_id, "".join(text_parts).strip(), tuple(tool_calls))

    def _stream_anthropic_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> AnthropicStreamResponse | str:
        return self._stream_anthropic_compatible_response(
            "https://api.anthropic.com/v1/messages",
            {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            "anthropic",
            "Anthropic",
            "ANTHROPIC_API_KEY",
            payload,
            delta_callback,
            status_callback,
        )

    def _stream_desy_response(
        self,
        api_key: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> AnthropicStreamResponse | str:
        return self._stream_anthropic_compatible_response(
            DESY_MESSAGES_ENDPOINT,
            {
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            "desy",
            "DESY Assistant",
            "DESY_ASSISTANT_API_KEY",
            payload,
            delta_callback,
            status_callback,
        )

    def _stream_anthropic_compatible_response(
        self,
        endpoint: str,
        headers: dict[str, str],
        provider_key: str,
        provider_label: str,
        env_var: str,
        payload: dict[str, Any],
        delta_callback: DeltaCallback | None,
        status_callback: StatusCallback | None,
    ) -> AnthropicStreamResponse | str:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        text_parts: list[str] = []
        content_by_index: dict[int, dict[str, Any]] = {}
        tool_json_parts: dict[int, list[str]] = {}
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line.removeprefix("data:").strip()
                    if not event_data:
                        continue
                    event = cast(dict[str, Any], json.loads(event_data))
                    event_type = str(event.get("type", ""))
                    if event_type == "content_block_start":
                        index = event.get("index")
                        block = event.get("content_block")
                        if not isinstance(index, int) or not isinstance(block, dict):
                            continue
                        block_type = str(block.get("type", ""))
                        if block_type == "text":
                            text = str(block.get("text", ""))
                            content_by_index[index] = {"type": "text", "text": text}
                            if text:
                                text_parts.append(text)
                                if delta_callback is not None:
                                    delta_callback(text)
                        elif block_type == "tool_use":
                            content_by_index[index] = {
                                "type": "tool_use",
                                "id": str(block.get("id", "")),
                                "name": str(block.get("name", "")),
                                "input": {},
                            }
                            tool_json_parts[index] = []
                        elif block_type == "thinking":
                            content_by_index[index] = {
                                "type": "thinking",
                                "thinking": str(block.get("thinking", "")),
                                "signature": str(block.get("signature", "")),
                            }
                    elif event_type == "content_block_delta":
                        index = event.get("index")
                        delta = event.get("delta")
                        if not isinstance(index, int) or not isinstance(delta, dict):
                            continue
                        delta_type = str(delta.get("type", ""))
                        if delta_type == "text_delta":
                            text = str(delta.get("text", ""))
                            if not text:
                                continue
                            block = content_by_index.get(index)
                            if isinstance(block, dict) and block.get("type") == "text":
                                block["text"] = f"{block.get('text', '')}{text}"
                            text_parts.append(text)
                            if delta_callback is not None:
                                delta_callback(text)
                        elif delta_type == "input_json_delta":
                            partial_json = str(delta.get("partial_json", ""))
                            if partial_json:
                                tool_json_parts.setdefault(index, []).append(partial_json)
                        elif delta_type == "thinking_delta":
                            thinking = str(delta.get("thinking", ""))
                            if not thinking:
                                continue
                            block = content_by_index.get(index)
                            if isinstance(block, dict) and block.get("type") == "thinking":
                                block["thinking"] = f"{block.get('thinking', '')}{thinking}"
                                self._reasoning_status(
                                    status_callback,
                                    str(block.get("thinking", "")),
                                )
                        elif delta_type == "signature_delta":
                            signature = str(delta.get("signature", ""))
                            block = content_by_index.get(index)
                            if (
                                signature
                                and isinstance(block, dict)
                                and block.get("type") == "thinking"
                            ):
                                block["signature"] = signature
                    elif event_type == "content_block_stop":
                        index = event.get("index")
                        if not isinstance(index, int):
                            continue
                        self._finalize_anthropic_tool_input(
                            content_by_index,
                            tool_json_parts,
                            index,
                        )
                    elif event_type == "error":
                        error = event.get("error")
                        if isinstance(error, dict):
                            message = str(error.get("message", "")).strip()
                            if message:
                                return f"{provider_label} request failed: {message}"
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            return self._api_error(provider_key, provider_label, env_var, error.code, error_body)
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            return f"{provider_label} request failed: {error}"

        for index in tuple(tool_json_parts):
            self._finalize_anthropic_tool_input(content_by_index, tool_json_parts, index)

        ordered_content = tuple(content_by_index[index] for index in sorted(content_by_index))
        tool_calls = tuple(
            AnthropicToolCall(
                name=str(block.get("name", "")),
                tool_use_id=str(block.get("id", "")),
                input=cast(dict[str, Any], block.get("input", {})),
            )
            for block in ordered_content
            if block.get("type") == "tool_use"
        )
        return AnthropicStreamResponse(
            "".join(text_parts).strip(),
            tool_calls,
            ordered_content,
        )

    def _stream_ollama_response(
        self,
        model: str,
        messages: list[dict[str, Any]],
        callbacks: RuntimeCallbacks,
    ) -> OllamaStreamResponse | str:
        payload = {
            "model": model,
            "messages": messages,
            "tools": self._ollama_tools(),
            "stream": True,
            "think": True,
        }
        self._status(callbacks.status, "Loading model")
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        thinking_parts: list[str] = []
        text_parts: list[str] = []
        tool_calls: list[OllamaToolCall] = []
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self._status(callbacks.status, "Thinking")
                for raw_line in response:
                    stripped = raw_line.decode("utf-8", errors="replace").strip()
                    if not stripped:
                        continue
                    data = cast(dict[str, Any], json.loads(stripped))
                    message = data.get("message")
                    if not isinstance(message, dict):
                        continue
                    thinking = message.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        thinking_parts.append(thinking)
                        self._reasoning_status(callbacks.status, "".join(thinking_parts))
                    content = message.get("content")
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                        if callbacks.delta is not None:
                            callbacks.delta(content)
                    raw_tool_calls = message.get("tool_calls")
                    if isinstance(raw_tool_calls, list):
                        for item in raw_tool_calls:
                            tool_call = self._ollama_tool_call(item)
                            if tool_call is not None:
                                tool_calls.append(tool_call)
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            detail = error_body.strip() or "No error detail."
            return f"Ollama request failed ({error.code}): {detail}"
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            return f"Ollama request failed: {error}"

        message: dict[str, Any] = {"role": "assistant"}
        if thinking_parts:
            message["thinking"] = "".join(thinking_parts)
        if text_parts:
            message["content"] = "".join(text_parts)
        if tool_calls:
            message["tool_calls"] = [
                self._ollama_tool_payload(tool_call) for tool_call in tool_calls
            ]
        return OllamaStreamResponse(
            "".join(text_parts).strip(),
            "".join(thinking_parts).strip(),
            tuple(tool_calls),
            message,
        )

    def _execute_requested_tools(
        self,
        response: OpenAIStreamResponse,
        callbacks: RuntimeCallbacks,
        session_path: Path,
    ) -> list[dict[str, str]]:
        outputs: list[dict[str, str]] = []
        for tool_call in response.tool_calls:
            arguments = self._parse_tool_arguments(tool_call.arguments)
            output = self._execute_tool(tool_call.name, arguments, callbacks, session_path)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": output,
                }
            )
        return outputs

    def _execute_anthropic_requested_tools(
        self,
        response: AnthropicStreamResponse,
        callbacks: RuntimeCallbacks,
        session_path: Path,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for tool_call in response.tool_calls:
            output = self._execute_tool(tool_call.name, tool_call.input, callbacks, session_path)
            outputs.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.tool_use_id,
                    "content": output,
                }
            )
        return outputs

    def _tool_call_from_stream_item(self, item: object) -> OpenAIToolCall | None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return None
        name = item.get("name")
        call_id = item.get("call_id")
        arguments = item.get("arguments")
        if isinstance(name, str) and isinstance(call_id, str) and isinstance(arguments, str):
            return OpenAIToolCall(name, call_id, arguments)
        return None

    def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
        session_path: Path | None = None,
    ) -> str:
        if name == "output_message":
            if self.role != AgentRole.OPERATOR:
                return self._json_tool_result({"error": "output_message is operator-only."})
            message = str(arguments.get("message", "")).strip()
            if callbacks.message is not None and message:
                callbacks.message(message)
            return self._json_tool_result({"delivered": bool(message)})

        if name in {"run_command", "run_cli_command"}:
            command = str(arguments.get("command", "")).strip()
            statement = str(arguments.get("statement", "")).strip()
            if self.cancel_event.is_set():
                return self._json_tool_result(
                    {
                        "approved": False,
                        "output": "Worker was stopped before the command could run.",
                    }
                )
            if self.role == AgentRole.WORKER and callbacks.tool_message is not None and statement:
                callbacks.tool_message(statement)
            if self.role == AgentRole.OPERATOR:
                self._emit_operator_tool_statement(name, arguments, callbacks)
            result = self.tool_manager.run_command(
                command,
                statement or "Operator command",
                callbacks.approval,
            )
            if self.role == AgentRole.WORKER and callbacks.command is not None:
                callbacks.command(statement, command, result.output)
            self._emit_command_system_message(callbacks, result)
            return self._json_tool_result(
                {
                    "approved": result.approved,
                    "output": result.output,
                }
            )

        if self.role != AgentRole.OPERATOR:
            return self._json_tool_result({"error": f"Unknown worker tool: {name}"})

        if name in {
            "create_plan",
            "update_plan",
            "start_agent",
            "prompt_agent",
            "check_agent",
            "stop_agent",
        }:
            self._emit_operator_tool_statement(name, arguments, callbacks)

        if name == "create_plan":
            return self._create_plan_tool(arguments, session_path)
        if name == "update_plan":
            return self._update_plan_tool(arguments, session_path)
        if name == "start_agent":
            return self._start_agent_tool(arguments, session_path, callbacks)
        if name == "prompt_agent":
            return self._prompt_agent_tool(arguments, session_path, callbacks)
        if name == "check_agent":
            return self._check_agent_tool(arguments, callbacks)
        if name == "stop_agent":
            return self._stop_agent_tool(arguments, session_path)
        if name == "wait":
            return self._wait_tool(arguments, callbacks)
        if name == "remove_plan":
            return self._remove_plan_tool(session_path, callbacks)
        return self._json_tool_result({"error": f"Unknown tool: {name}"})

    def _create_plan_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "create_plan requires a session."})

        steps = build_plan_steps(arguments.get("steps"))
        if not steps:
            return self._json_tool_result({"error": "create_plan requires at least one step."})

        payload = {"steps": serialize_plan_steps(steps)}
        self.home.append_session_event(session_path, "plan_update", payload)
        return self._json_tool_result({"created": True, **payload})

    def _update_plan_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "update_plan requires a session."})

        current = latest_plan_steps(self.home.read_session_events(session_path))
        steps = merge_plan_steps(current, arguments.get("steps"))
        if not steps:
            return self._json_tool_result({"error": "update_plan requires plan steps."})

        payload = {"steps": serialize_plan_steps(steps)}
        self.home.append_session_event(session_path, "plan_update", payload)
        return self._json_tool_result({"updated": True, **payload})

    def _remove_plan_tool(
        self,
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "remove_plan requires a session."})
        self.home.append_session_event(session_path, "plan_update", {"steps": []})
        if callbacks.tool_message is not None:
            callbacks.tool_message("Removed plan")
        return self._json_tool_result({"removed": True})

    def _start_agent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        prompt = str(arguments.get("prompt", "")).strip()
        name = self._optional_worker_name(arguments.get("name"))
        worker = self._start_worker_agent(
            prompt=prompt,
            name=name,
            session_path=session_path,
            callbacks=callbacks,
        )
        if isinstance(worker, str):
            return self._json_tool_result({"error": worker})
        return self._json_tool_result(
            {
                "agent_id": worker.worker_id,
                "name": worker.name,
                "status": worker.status,
                "statement": worker.statement,
            }
        )

    def _prompt_agent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        prompt = str(arguments.get("prompt", "")).strip()
        requested_agent_id = str(arguments.get("agent_id") or "").strip()
        name = self._optional_worker_name(arguments.get("name"))
        worker = self._start_worker_agent(
            prompt=prompt,
            name=name,
            session_path=session_path,
            callbacks=callbacks,
            requested_agent_id=requested_agent_id or None,
        )
        if isinstance(worker, str):
            return self._json_tool_result({"error": worker})
        return self._json_tool_result(
            {
                "agent_id": worker.worker_id,
                "name": worker.name,
                "status": worker.status,
                "statement": worker.statement,
            }
        )

    def _stop_agent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        worker_id = str(arguments.get("agent_id") or arguments.get("worker_id") or "").strip()
        if not worker_id:
            return self._json_tool_result({"error": "stop_agent requires an agent_id."})

        with self._worker_lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return self._json_tool_result({"stopped": False, "error": "Unknown agent id."})
            worker.cancel_event.set()
            if worker.status == "running":
                worker.status = "stopped"
                worker.statement = "stopped"
                worker.finished_at = utc_now_iso()
                self._append_worker_event(session_path, worker)
                self._append_worker_system_message(
                    session_path,
                    worker,
                    "stopped",
                    "Worker was stopped by the Operator.",
                )
        return self._json_tool_result({"stopped": True, "agent_id": worker_id})

    def _check_agent_tool(
        self,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
    ) -> str:
        worker_id = str(arguments.get("agent_id") or arguments.get("worker_id") or "").strip()
        if not worker_id:
            return self._json_tool_result({"error": "check_agent requires an agent_id."})
        with self._worker_lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return self._json_tool_result({"error": "Unknown agent id."})
            payload = {
                "agent_id": worker.worker_id,
                "name": worker.name,
                "status": worker.status,
                "statement": worker.statement,
                "commands": list(worker.command_history),
            }
        return self._json_tool_result(payload)

    def _wait_tool(
        self,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        del arguments
        seconds = 60.0
        started_at = time.monotonic()
        if not self._running_worker_states():
            return self._json_tool_result(
                {
                    "waited_seconds": 0.0,
                    "workers": [
                        self._worker_state_payload(worker) for worker in self._worker_states()
                    ],
                }
            )
        if callbacks is not None and callbacks.status is not None:
            callbacks.status(f"Waiting:{seconds}")
        deadline = started_at + seconds
        while time.monotonic() < deadline:
            if not self._running_worker_states():
                break
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        waited_seconds = min(seconds, max(0.0, time.monotonic() - started_at))
        return self._json_tool_result(
            {
                "waited_seconds": waited_seconds,
                "workers": [self._worker_state_payload(worker) for worker in self._worker_states()],
            }
        )

    def _start_worker_agent(
        self,
        *,
        prompt: str,
        name: str | None,
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
        requested_agent_id: str | None = None,
    ) -> WorkerAgentState | str:
        if session_path is None:
            return "Worker agents require an operator session."
        if not prompt:
            return "Worker agents require a prompt."

        with self._worker_lock:
            existing = self._workers.get(requested_agent_id or "")
            if existing is not None and existing.status == "running":
                return "That worker is still running."
            worker_id = requested_agent_id or uuid4().hex[:8]
            worker_name = name or (existing.name if existing is not None else "Worker")
            worker = WorkerAgentState(
                worker_id=worker_id,
                name=worker_name,
                prompt=prompt,
                status="running",
                statement="thinking",
                started_at=utc_now_iso(),
                cancel_event=threading.Event(),
            )
            self._workers[worker_id] = worker

        self._append_worker_event(session_path, worker)
        worker.thread = threading.Thread(
            target=self._run_worker_agent,
            args=(worker, session_path, callbacks),
            daemon=True,
        )
        worker.thread.start()
        return worker

    def _run_worker_agent(
        self,
        worker: WorkerAgentState,
        operator_session_path: Path,
        operator_callbacks: RuntimeCallbacks,
    ) -> None:
        def update_statement(message: str) -> None:
            statement = message.strip() or "thinking"
            with self._worker_lock:
                if worker.worker_id not in self._workers or worker.status != "running":
                    return
                worker.statement = statement
                self._append_worker_event(operator_session_path, worker)

        def record_command(statement: str, command: str, output: str) -> None:
            with self._worker_lock:
                if worker.worker_id not in self._workers:
                    return
                worker.command_history.append(
                    {
                        "statement": statement or "Running command",
                        "command": command,
                        "output": self._compact_worker_output(output),
                    }
                )
                self._append_worker_event(operator_session_path, worker)

        worker_runtime = AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.tool_manager.mode,
            role=AgentRole.WORKER,
            cancel_event=worker.cancel_event,
        )
        response = ""
        status = "finished"
        try:
            response = worker_runtime.backend_response_for_prompt(
                worker.prompt,
                callbacks=RuntimeCallbacks(
                    status=update_statement,
                    tool_message=update_statement,
                    command=record_command,
                    approval=operator_callbacks.approval,
                ),
            )
        except Exception as error:  # pragma: no cover - defensive thread boundary
            status = "failed"
            response = f"Worker failed: {error}"

        with self._worker_lock:
            if worker.cancel_event.is_set() or worker.status == "stopped":
                worker.status = "stopped"
                worker.statement = "stopped"
                worker.finished_at = worker.finished_at or utc_now_iso()
                worker.response = response
                self._append_worker_event(operator_session_path, worker)
                return

            worker.status = status
            worker.statement = status
            worker.finished_at = utc_now_iso()
            worker.response = response
            self._append_worker_event(operator_session_path, worker)

        self._append_worker_system_message(
            operator_session_path,
            worker,
            status,
            response,
        )

    def _append_worker_event(
        self,
        session_path: Path | None,
        worker: WorkerAgentState,
    ) -> None:
        if session_path is None:
            return
        self.home.append_session_event(
            session_path,
            "worker_event",
            self._worker_state_payload(worker),
        )

    def _append_worker_system_message(
        self,
        session_path: Path | None,
        worker: WorkerAgentState,
        status: str,
        response: str,
    ) -> None:
        if session_path is None:
            return
        compact_response = response.strip() or "No worker response."
        self.home.append_session_event(
            session_path,
            "system_message",
            {
                "message": (
                    f"Worker {worker.name} ({worker.worker_id}) {status}:\n"
                    f"{compact_response}"
                ),
                "role": "worker",
                "worker_id": worker.worker_id,
            },
        )

    def _worker_states(self) -> tuple[WorkerAgentState, ...]:
        with self._worker_lock:
            return tuple(self._workers.values())

    def _running_worker_states(self) -> tuple[WorkerAgentState, ...]:
        return tuple(worker for worker in self._worker_states() if worker.status == "running")

    def _worker_state_payload(self, worker: WorkerAgentState) -> dict[str, str]:
        return {
            "worker_id": worker.worker_id,
            "name": worker.name,
            "status": worker.status,
            "statement": worker.statement,
            "prompt": worker.prompt,
            "response": worker.response,
            "started_at": worker.started_at,
            "finished_at": worker.finished_at,
        }

    def _optional_worker_name(self, value: object) -> str | None:
        name = str(value or "").strip()
        return name or None

    def _bounded_wait_seconds(self, value: object) -> float:
        try:
            seconds = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            seconds = 60.0
        return max(0.25, min(seconds, 60.0))

    def _emit_operator_tool_statement(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
    ) -> None:
        callback = callbacks.tool_message or callbacks.status
        if callback is None:
            return
        statement = str(arguments.get("statement", "")).strip()
        callback(statement or self._default_operator_tool_statement(tool_name))

    def _default_operator_tool_statement(self, tool_name: str) -> str:
        return {
            "run_command": "Running command",
            "run_cli_command": "Running command",
            "create_plan": "Creating plan",
            "update_plan": "Updating plan",
            "start_agent": "Starting Worker",
            "prompt_agent": "Prompting Worker",
            "check_agent": "Checking Worker",
            "stop_agent": "Stopping Worker",
            "remove_plan": "Removing plan",
        }.get(tool_name, "Working")

    def _compact_worker_output(self, output: str) -> str:
        compact = " ".join(output.strip().split())
        if len(compact) <= 500:
            return compact
        return f"{compact[:497]}..."

    def _api_key(self, provider: str, env_var: str) -> str | None:
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key
        api_keys = self.home.load_auth().get("api_keys")
        if not isinstance(api_keys, dict):
            return None
        configured_key = api_keys.get(provider)
        if isinstance(configured_key, str) and configured_key.strip():
            return configured_key.strip()
        return None

    def _api_error(
        self,
        provider_key: str,
        provider_label: str,
        env_var: str,
        status: int,
        body: str,
    ) -> str:
        detail, error_type = self._parse_api_error(body)
        if self._looks_like_invalid_api_key(provider_key, status, error_type, detail):
            return self._invalid_api_key_message(provider_label, env_var)
        return f"{provider_label} request failed ({status}): {detail or 'No error detail.'}"

    def _parse_api_error(self, body: str) -> tuple[str, str | None]:
        detail = body.strip()
        error_type: str | None = None
        with suppress(json.JSONDecodeError):
            payload = json.loads(body)
            if isinstance(payload, dict):
                if isinstance(payload.get("detail"), str):
                    detail = payload["detail"]
                error = payload.get("error")
                if isinstance(error, dict):
                    if isinstance(error.get("message"), str):
                        detail = error["message"]
                    if isinstance(error.get("type"), str):
                        error_type = error["type"]
                elif isinstance(error, str):
                    detail = error
        return detail or "No error detail.", error_type

    def _missing_api_key_message(self, provider_label: str, env_var: str) -> str:
        return (
            f"{provider_label} API key is not configured. "
            f"Add it during onboarding or set {env_var}."
        )

    def _invalid_api_key_message(self, provider_label: str, env_var: str) -> str:
        return (
            f"{provider_label} credentials were rejected. "
            f"Check {env_var} or update the saved {provider_label} API key in Anomx. "
            "The key may be invalid, expired, or revoked."
        )

    def _looks_like_invalid_api_key(
        self,
        provider_key: str,
        status: int,
        error_type: str | None,
        detail: str,
    ) -> bool:
        lowered = detail.lower()
        if provider_key == "openai":
            if error_type == "authentication_error" and "member of an organization" not in lowered:
                return True
            if status != 401:
                return False
            return any(
                needle in lowered
                for needle in (
                    "incorrect api key",
                    "invalid api key",
                    "invalid authentication",
                    "invalid_api_key",
                    "revoked",
                    "expired",
                )
            )
        if provider_key in {"anthropic", "desy"}:
            return status == 401 or error_type == "authentication_error"
        return False

    def _suggest_openai_session_title(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> str | None:
        api_key = self._api_key("openai", "OPENAI_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "input": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_output_tokens": 24,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_openai_text(data))

    def _suggest_anthropic_session_title(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> str | None:
        api_key = self._api_key("anthropic", "ANTHROPIC_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_tokens": 24,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_anthropic_text(data))

    def _suggest_desy_session_title(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> str | None:
        api_key = self._api_key("desy", "DESY_ASSISTANT_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            DESY_MESSAGES_ENDPOINT,
            data=json.dumps(
                {
                    "model": model,
                    "system": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_tokens": 24,
                    "stream": False,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_title(self.extract_anthropic_text(data))

    def _suggest_ollama_session_title(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Name this CLI session in 3 to 6 words. "
                        "Return only the title. No quotes. No trailing punctuation."
                    ),
                },
                {"role": "user", "content": self._title_prompt(messages)},
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_title(str(message.get("content", "")))

    def _title_prompt(self, messages: list[dict[str, str]]) -> str:
        conversation = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages[-6:]
        )
        return f"Conversation:\n{conversation}"

    def _heuristic_session_title(self, messages: list[dict[str, str]]) -> str | None:
        first_user_message = next(
            (
                message["content"]
                for message in messages
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        return self._sanitize_title(first_user_message)

    def _sanitize_title(self, title: str) -> str | None:
        cleaned = " ".join(title.strip().strip("\"'`").split())
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned:
            return None
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[:8])
        return cleaned[:60] or None

    def _instructions(self, session_path: Path | None = None) -> str:
        sections = [self.tool_manager.mode.system_prompt_statement]
        session_policy = self.tool_manager.session_policy_prompt_lines()
        if session_policy:
            sections.append("\n".join(session_policy))

        if self.role == AgentRole.WORKER:
            tools = "\n".join(f"- {tool}" for tool in WORKER_TOOL_DESCRIPTIONS)
            return "\n\n".join(
                [
                    WORKER_SYSTEM_PROMPT,
                    *sections,
                    f"Available tools:\n{tools}",
                ]
            )

        tools = "\n".join(f"- {tool}" for tool in OPERATOR_TOOL_DESCRIPTIONS)
        runtime_context = self._operator_runtime_context(session_path)
        return "\n\n".join(
            [
                OPERATOR_SYSTEM_PROMPT,
                *sections,
                runtime_context,
                f"Available tools:\n{tools}",
            ]
        )

    def _operator_runtime_context(self, session_path: Path | None) -> str:
        if session_path is None:
            return "Runtime context:\n- No active session context."

        events = self.home.read_session_events(session_path)
        plan_steps = latest_plan_steps(events)
        workers = worker_snapshots(events)
        lines = ["Runtime context:"]
        if plan_steps:
            lines.append("- Current plan:")
            for step in plan_steps:
                state = "done" if step.is_done else "open"
                lines.append(f"  {step.position}. [{state}] {step.title}: {step.description}")
        else:
            lines.append("- Current plan: none.")

        running_workers = tuple(
            worker for worker in workers if worker.status in {"running"}
        )
        if running_workers:
            lines.append("- Running worker agents:")
            for worker in running_workers:
                lines.append(
                    "  "
                    f"{worker.worker_id} · {worker.name} · {worker.statement} · "
                    f"running for {self._worker_runtime_duration(worker.started_at)}"
                )
        else:
            lines.append("- Running worker agents: none.")

        finished_workers = tuple(
            worker for worker in workers if worker.status not in {"running"}
        )
        if finished_workers:
            lines.append("- Finished/stopped worker agents:")
            for worker in finished_workers[-5:]:
                lines.append(f"  {worker.worker_id} · {worker.name} · {worker.status}")
        return "\n".join(lines)

    def _worker_runtime_duration(self, started_at: str) -> str:
        with suppress(ValueError):
            from datetime import UTC, datetime

            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            minutes, remaining_seconds = divmod(seconds, 60)
            return f"{minutes:02d}:{remaining_seconds:02d}"
        return "unknown"

    def _anthropic_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                self._append_anthropic_message(converted, "assistant", content)
            elif role == "system":
                self._append_anthropic_message(converted, "user", f"[System note]\n{content}")
            else:
                self._append_anthropic_message(converted, "user", content)
        return converted

    def _append_anthropic_message(
        self,
        messages: list[dict[str, Any]],
        role: str,
        text: str,
    ) -> None:
        block = {"type": "text", "text": text}
        if messages and messages[-1].get("role") == role:
            content = messages[-1].get("content")
            if isinstance(content, list):
                content.append(block)
                return
        messages.append({"role": role, "content": [block]})

    def _extract_anthropic_text(
        self,
        content: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        for block in content:
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip() or "No response."

    def _finalize_anthropic_tool_input(
        self,
        content_by_index: dict[int, dict[str, Any]],
        tool_json_parts: dict[int, list[str]],
        index: int,
    ) -> None:
        block = content_by_index.get(index)
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return
        raw_json = "".join(tool_json_parts.get(index, [])).strip()
        if not raw_json:
            block["input"] = {}
            return
        with suppress(json.JSONDecodeError):
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                block["input"] = parsed
                return
        block["input"] = {"raw_input": raw_json}

    def _max_output_tokens(self, model: str, fallback: int) -> int:
        metadata = model_metadata(model)
        if metadata is None or metadata.max_output_tokens is None:
            return fallback
        return metadata.max_output_tokens

    def _anthropic_thinking_config(self, model: str) -> dict[str, Any]:
        if model in {"claude-opus-4-8", "claude-sonnet-4-6"}:
            return {"type": "adaptive", "display": "summarized"}
        max_tokens = self._max_output_tokens(model, 4_096)
        budget_tokens = max(1_024, min(2_048, max_tokens - 1))
        return {
            "type": "enabled",
            "budget_tokens": budget_tokens,
            "display": "summarized",
        }

    def _anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in self._tool_definitions()
        ]

    def _ollama_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self._tool_definitions()
        ]

    def _openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "strict": True,
                "parameters": tool["parameters"],
            }
            for tool in self._tool_definitions()
        ]

    def _tool_definitions(self) -> list[dict[str, Any]]:
        if self.role == AgentRole.WORKER:
            return [
                {
                    "name": "run_command",
                    "description": "Run a CLI command and publish the statement as worker status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "statement": {
                                "type": "string",
                                "description": "Concise current-action statement for the Operator.",
                            },
                            "command": {
                                "type": "string",
                                "description": "A single shell-free command, for example 'ls -la'.",
                            },
                        },
                        "required": ["statement", "command"],
                        "additionalProperties": False,
                    },
                }
            ]

        statement_description = "Persistent user-visible working message for this tool call."
        tools = [
            {
                "name": "output_message",
                "description": "Send an intermediate progress update to the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Short user-facing progress update.",
                        },
                    },
                    "required": ["message"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_command",
                "description": "Run a CLI command for operator inspection or validation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": (
                                "Persistent user-visible working message for this tool call, "
                                "for example 'Checking repository state'."
                            ),
                        },
                        "command": {
                            "type": "string",
                            "description": "A single shell-free command, for example 'ls -la'.",
                        },
                    },
                    "required": ["statement", "command"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "start_agent",
                "description": "Start a background Worker agent for a focused task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "name": {
                            "type": ["string", "null"],
                            "description": "Optional short worker name, such as Engineer.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Specific task prompt for the Worker.",
                        },
                    },
                    "required": ["statement", "name", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "prompt_agent",
                "description": "Prompt a finished Worker again, or start a new Worker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": ["string", "null"],
                            "description": "Existing finished Worker id, or null for a new Worker.",
                        },
                        "name": {
                            "type": ["string", "null"],
                            "description": "Optional Worker name when starting a new Worker.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Specific task prompt for the Worker.",
                        },
                    },
                    "required": ["statement", "agent_id", "name", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "check_agent",
                "description": "Inspect a Worker agent's command and statement history.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Worker id to inspect.",
                        },
                    },
                    "required": ["statement", "agent_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "stop_agent",
                "description": "Request that a running Worker stop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Worker id to stop.",
                        },
                    },
                    "required": ["statement", "agent_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "create_plan",
                "description": "Create a user-visible ordered plan.",
                "parameters": self._plan_schema(require_position=False),
            },
            {
                "name": "update_plan",
                "description": "Update the user-visible ordered plan.",
                "parameters": self._plan_schema(require_position=True),
            },
            {
                "name": "remove_plan",
                "description": "Clear the current user-visible plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                    },
                    "required": ["statement"],
                    "additionalProperties": False,
                },
            },
        ]
        if self._running_worker_states():
            tools.append(
                {
                    "name": "wait",
                    "description": "Wait up to 60 seconds for running Worker agents.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    def _plan_schema(self, *, require_position: bool) -> dict[str, Any]:
        properties: dict[str, Any] = {
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
        return {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "Persistent user-visible working message for this tool call.",
                },
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
            "required": ["statement", "steps"],
            "additionalProperties": False,
        }

    def _json_tool_result(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _parse_tool_arguments(self, raw_arguments: str) -> dict[str, Any]:
        with suppress(json.JSONDecodeError):
            payload = json.loads(raw_arguments)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        return {}

    def _ollama_tool_call(self, item: object) -> OllamaToolCall | None:
        if not isinstance(item, dict):
            return None
        function = item.get("function")
        if not isinstance(function, dict):
            return None
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str):
            return None
        if isinstance(arguments, dict):
            parsed_arguments = cast(dict[str, Any], arguments)
        elif isinstance(arguments, str):
            parsed_arguments = self._parse_tool_arguments(arguments)
        else:
            parsed_arguments = {}
        return OllamaToolCall(name=name, arguments=parsed_arguments)

    def _ollama_tool_payload(self, tool_call: OllamaToolCall) -> dict[str, Any]:
        return {
            "function": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
        }

    def _execute_ollama_requested_tools(
        self,
        response: OllamaStreamResponse,
        callbacks: RuntimeCallbacks,
        session_path: Path,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for tool_call in response.tool_calls:
            output = self._execute_tool(
                tool_call.name,
                tool_call.arguments,
                callbacks,
                session_path,
            )
            outputs.append(
                {
                    "role": "tool",
                    "tool_name": tool_call.name,
                    "content": output,
                }
            )
        return outputs

    def _status(
        self,
        status_callback: StatusCallback | None,
        message: str = "Thinking",
    ) -> None:
        if status_callback is not None:
            status_callback(message)

    def _reasoning_status(
        self,
        status_callback: StatusCallback | None,
        reasoning_text: str,
    ) -> None:
        if status_callback is not None and reasoning_text.strip():
            status_callback("Thinking")

    def _compact_reasoning_text(self, reasoning_text: str) -> str:
        compact = " ".join(reasoning_text.strip().split())
        if not compact:
            return ""
        if len(compact) <= 120:
            return compact
        tail = compact[-117:]
        first_space = tail.find(" ")
        if first_space > 0:
            tail = tail[first_space + 1 :]
        return f"…{tail}"

    def _emit_command_system_message(
        self,
        callbacks: RuntimeCallbacks,
        result: CommandResult,
    ) -> None:
        if callbacks.system_message is None:
            return
        if result.blocked_by_mode:
            return
        if result.safety == CommandSafety.FORBIDDEN:
            callbacks.system_message(
                "forbidden",
                f"Blocked command: {result.command} · {result.reason}",
            )
