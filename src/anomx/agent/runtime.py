"""Model backend and local tool runtime for the Anomx CLI agent."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, MutableSet
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO, TypeAlias, cast
from uuid import uuid4

from anomx.agent.mode import AgentMode
from anomx.agent.state import (
    WORKER_STATE_INTERRUPTED,
    WORKER_STATE_READY,
    WORKER_STATE_REMOVED,
    WORKER_STATE_WORKING,
    PlanStep,
    WorkerAgentSnapshot,
    build_plan_steps,
    latest_plan_steps,
    merge_plan_steps,
    running_process_snapshots,
    serialize_plan_steps,
    worker_snapshots,
)
from anomx.agent.store import (
    THINKING_INTENSITY_AUTO,
    AnomxHome,
    model_context_window,
    model_metadata,
    normalize_thinking_intensity,
    thinking_intensity_options,
    utc_now_iso,
)
from anomx.agent.tool_manager import (
    ApprovalCallback,
    CliToolManager,
    CommandProcessResult,
    CommandResult,
    CommandSafety,
    discover_workspace_root,
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
- Worker agents run in background threads and report back as system messages. Workers
  are listed in your runtime context with their id, name, state, current statement when
  working, and runtime duration when working.
- Worker agent states are working, ready, and interrupted. start_agent creates a new
  Worker and immediately moves it to working. prompt_agent sends a new prompt only to a
  Worker in ready or interrupted state, then moves it back to working. interrupt_agent
  interrupts a working Worker when it is off track or no longer useful. remove_agent
  removes a Worker from the active bottom panel and runtime context.
- Default delegation policy: if the user asks for implementation, repository creation,
  code changes, multi-file investigation, or validation, start at least one Worker agent
  with start_agent unless the task is trivial enough to complete with one or two direct
  read-only commands. Good worker names include Engineer, Reviewer, Researcher, and
  Designer.
- After creating a plan for a non-trivial task, immediately start the first Worker agent
  in the same tool loop whenever possible. Do not end your response after only creating
  or describing the plan.
- Start independent research, implementation, or review tasks with start_agent. Use
  prompt_agent only to continue an existing ready or interrupted Worker. Use
  interrupt_agent when a working Worker is no longer useful or is clearly off track.
- Use start_process for long-running async CLI commands such as npm run dev. Async
  processes are shown beside Worker agents, are listed in your runtime context, and can
  continue running after your final answer. Use end_process when a process is no longer
  needed.
- If your own run_command call becomes long-running, it is promoted to a command tool call
  with a command id, displayed beside Workers and processes, and listed in runtime context.
  In that state use the temporary command-control tools that appear in Available tools.
  Do not produce a final answer while a required run_command tool call is still running.
- Worker-owned long-running command calls may appear in runtime context and the bottom
  panel, but their command-control tools are scoped to that Worker. Use check_agent to
  inspect the Worker's status instead of controlling those command calls directly.
- When workers are still working and you need their results, use wait. While waiting you
  may still send output_message updates, validate state with run_command, or interrupt a
  worker.
- The wait tool has no arguments and is only available while at least one Worker is in
  working state or one of your own long-running command tool calls is active. It waits up
  to 60 seconds, but returns early as soon as all wait targets leave running state.
- Use remove_plan when starting a new unrelated task and the previous plan no longer
  describes the active work.
- Use finish_anyways only when the plan-finish checker asks for an explicit override
  and you are sure the final answer should be delivered despite open plan steps.
- Do not produce a final answer while required workers are still working. Use wait, review
  their reports, update the plan, and then finalize. A running async process is not by
  itself a reason to delay the final answer.
- Use run_command(statement, command) for your own validation, inspection, review, and
  final checks. Every Operator tool except output_message, wait, and temporary
  command-control tools requires statement. The statement is a persistent working message
  visible to the user, for example "Checking repository state" or "Starting Engineer
  Worker".
- Confirm Mode does not mean "ask the user in prose before doing work." If a command needs
  approval, call run_command or start a Worker anyway; the command approval UI will ask the
  user at the moment approval is required.
- Use ask_question when you genuinely need the user's choice or typed input before a
  high-impact, ambiguous, or impossible-to-infer decision. Choose kind=select for
  predefined options, kind=text for free-form input, and kind=confirm for yes/no decisions.
- Make pragmatic default choices for ordinary scaffolding details such as folder names,
  package managers, and starter options. Ask the user only when a missing detail would make
  the work destructive, ambiguous in a high-impact way, or impossible to validate.
- Keep the user updated with output_message more often during multi-stage work: send a
  brief update when you start meaningful investigation or implementation, when you learn
  something that changes the next step, before longer waits or validations, and when you
  move from one major phase to another. Avoid narrating every tiny command.
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
- If run_command becomes long-running, it is promoted to a command tool call with a command
  id. While it is active, use wait, check_command_status, or kill_command as needed. The
  Operator sees you as waiting for that tool call, but only you receive those command
  control tools.
- If a command needs user approval, still call run_command. The approval UI handles that
  flow; do not stop with a prose request for permission.
- Return a concise final report to the Operator with concrete findings, files changed if
  relevant, validation performed, and any blockers. Do not return a final report while a
  required command tool call is still running. Do not ask the user questions.
"""

OPERATOR_TOOL_DESCRIPTIONS = (
    "output_message(message): send a concise working update to the user.",
    (
        "run_command(statement, command): run a safe CLI command for operator validation "
        "or inspection and persist statement as a working message."
    ),
    "start_agent(statement, name, prompt): start a background Worker agent for a focused task.",
    (
        "prompt_agent(statement, agent_id, prompt): prompt a ready or interrupted Worker again."
    ),
    "interrupt_agent(statement, agent_id): interrupt a working Worker agent.",
    "remove_agent(statement, agent_id): remove a Worker agent from active context.",
    "start_process(statement, command): start a long-running async CLI process.",
    "end_process(statement, process_id): end a running async CLI process.",
    (
        "ask_question(statement, question, kind, options, placeholder, default, "
        "allow_custom): ask the user for a choice or typed answer."
    ),
    "create_plan(statement, steps): create a user-visible ordered plan.",
    "update_plan(statement, steps): update the user-visible ordered plan.",
    "remove_plan(statement): clear the current user-visible plan.",
    (
        "finish_anyways(statement): clear the current plan and finish after the "
        "plan-finish checker asks for an explicit override."
    ),
)

WORKER_TOOL_DESCRIPTIONS = (
    (
        "run_command(statement, command): run a safe CLI command and publish statement "
        "as worker status."
    ),
)

MAX_TOOL_ITERATIONS = 128
OPENAI_MAX_TOOL_CALLS = 128
MAX_PLAN_FINISH_REPROMPTS = 3
DESY_MESSAGES_ENDPOINT = "https://assistant.desy.de/api/v1/messages"
MODEL_REQUEST_RETRY_STATUS_CODES = frozenset({404, 500, 503})
MODEL_REQUEST_RETRY_DELAYS_SECONDS = tuple(float(seconds) for seconds in range(10, 101, 10))
MODEL_REQUEST_RETRY_SLEEP_SLICE_SECONDS = 0.25
CONTEXT_CHARACTERS_PER_TOKEN = 4
MESSAGE_CONTEXT_OVERHEAD_TOKENS = 4
INSTRUCTIONS_CONTEXT_OVERHEAD_TOKENS = 8
MESSAGE_IMAGE_CONTEXT_TOKENS = 2_000
SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
IMAGE_FILE_EXTENSIONS = (".gif", ".jpeg", ".jpg", ".png", ".webp")
OLLAMA_IMAGE_MODEL_MARKERS = frozenset(
    {
        "bakllava",
        "gemma3",
        "llama3.2-vision",
        "llava",
        "minicpm-v",
        "moondream",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen3-vl",
        "vision",
    }
)


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
    context_command_history: list[dict[str, str]] = field(default_factory=list)
    context_tokens: int = 0
    context_percent: int = 0
    session_path: Path | None = None


@dataclass
class AsyncProcessState:
    """Mutable in-process state for a long-running async command."""

    process_id: str
    command: str
    statement: str
    status: str
    started_at: str
    process: subprocess.Popen[str]
    finished_at: str = ""
    output: str = ""
    exit_code: int | None = None
    source: str = "process"
    owner_id: str = ""
    owner_name: str = ""
    session_path: Path | None = None
    output_chunks: list[str] = field(default_factory=list, repr=False)
    debug_log_written: bool = False
    thread: threading.Thread | None = None


ProcessCallback = Callable[[AsyncProcessState], None]


@dataclass(frozen=True)
class RuntimeCleanupResult:
    """Summary of stale runtime state removed from a session transcript."""

    workers_removed: int = 0
    processes_ended: int = 0


@dataclass(frozen=True)
class QuestionOption:
    """A user-selectable option for an operator question."""

    label: str
    value: str
    description: str = ""


@dataclass(frozen=True)
class QuestionRequest:
    """Interactive question request shown by the CLI."""

    question: str
    kind: str
    options: tuple[QuestionOption, ...] = ()
    placeholder: str = ""
    default: str = ""
    allow_custom: bool = False


@dataclass(frozen=True)
class QuestionResponse:
    """Answer returned from an interactive CLI question."""

    answered: bool
    answer: str = ""
    selected_label: str = ""
    kind: str = ""
    cancelled: bool = False


QuestionCallback = Callable[[QuestionRequest], QuestionResponse]


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
    question: QuestionCallback | None = None
    process: ProcessCallback | None = None


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


ModelRequestStreamResponse: TypeAlias = OpenAIStreamResponse | AnthropicStreamResponse | str


@dataclass(frozen=True)
class ImageAttachment:
    """Image file attached to a user message."""

    label: str
    token: str
    path: Path
    mime_type: str

    def to_payload(self) -> dict[str, str]:
        """Serialize image metadata for session storage."""

        return {
            "label": self.label,
            "token": self.token,
            "path": self.path.as_posix(),
            "mime_type": self.mime_type,
        }


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


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for display-only context accounting."""

    return (len(text) + CONTEXT_CHARACTERS_PER_TOKEN - 1) // CONTEXT_CHARACTERS_PER_TOKEN


def image_mime_type(path: Path) -> str | None:
    """Return a supported image MIME type inferred from a path."""

    mime_type, _encoding = mimetypes.guess_type(path.name)
    return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else None


def backend_supports_image_input(provider_key: str, model: str) -> bool:
    """Return whether the selected backend/model can receive image input."""

    if provider_key in {"openai", "anthropic"}:
        return True
    if provider_key == "ollama":
        normalized = model.lower()
        return any(marker in normalized for marker in OLLAMA_IMAGE_MODEL_MARKERS)
    return False


def estimate_backend_context_tokens(
    instructions: str,
    messages: Iterable[Mapping[str, Any]],
) -> int:
    """Estimate the context sent to a backend request.

    The agent supports several providers without adding tokenizer dependencies, so this
    intentionally uses a conservative character heuristic over the same instructions and
    messages that are passed to the backend.
    """

    tokens = estimate_text_tokens(instructions) + INSTRUCTIONS_CONTEXT_OVERHEAD_TOKENS
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        images = normalized_image_attachments(message.get("images"))
        if not content and not images:
            continue
        tokens += MESSAGE_CONTEXT_OVERHEAD_TOKENS
        tokens += estimate_text_tokens(role)
        tokens += estimate_text_tokens(content)
        tokens += len(images) * MESSAGE_IMAGE_CONTEXT_TOKENS
    return max(1, tokens)


def normalized_image_attachments(raw_images: object) -> tuple[ImageAttachment, ...]:
    """Build readable image attachments from persisted message metadata."""

    if not isinstance(raw_images, (list, tuple)):
        return ()

    attachments: list[ImageAttachment] = []
    for raw_image in raw_images:
        if not isinstance(raw_image, Mapping):
            continue
        raw_path = raw_image.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser()
        mime_type = str(raw_image.get("mime_type") or image_mime_type(path) or "").strip()
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            continue
        label = str(raw_image.get("label") or path.name).strip() or path.name
        token = str(raw_image.get("token") or f"[image: {label}]").strip()
        attachments.append(
            ImageAttachment(
                label=label,
                token=token,
                path=path,
                mime_type=mime_type,
            )
        )
    return tuple(attachments)


def context_usage_percent(used_tokens: int, context_window: int | None) -> int:
    """Return clamped percent of context window currently used."""

    if context_window is None or context_window <= 0 or used_tokens <= 0:
        return 0
    percent = round((used_tokens / context_window) * 100)
    return max(1, min(100, percent))


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
        workspace_root: Path | None = None,
        process_owner_id: str = "",
        process_owner_name: str = "",
    ) -> None:
        self.home = home
        self.cwd = cwd.expanduser().resolve()
        self.workspace_root = (
            discover_workspace_root(self.cwd)
            if workspace_root is None
            else workspace_root.expanduser().resolve()
        )
        self.cancel_event = threading.Event() if cancel_event is None else cancel_event
        self.tool_manager = CliToolManager(
            self.workspace_root,
            session_allowed_commands,
            session_rejected_commands,
            mode,
            current_dir=self.cwd,
            cancel_event=self.cancel_event,
        )
        self.session_allowed_commands = session_allowed_commands
        self.session_rejected_commands = session_rejected_commands
        self.role = role
        self.process_owner_id = process_owner_id
        self.process_owner_name = process_owner_name
        self._turn_abort_event = threading.Event()
        self._workers: dict[str, WorkerAgentState] = {}
        self._worker_lock = threading.Lock()
        self._processes: dict[str, AsyncProcessState] = {}
        self._process_lock = threading.Lock()
        self._debug_session_path: Path | None = None

    def set_mode(self, mode: AgentMode) -> None:
        """Set the active command execution mode."""

        self.tool_manager.set_mode(mode)

    def abort_current_turn(
        self,
        session_path: Path | None = None,
        *,
        stop_workers: bool = True,
    ) -> None:
        """Request cancellation of the active response loop and running workers."""

        self._turn_abort_event.set()
        if stop_workers:
            self._stop_running_workers(
                session_path,
                "Worker was interrupted because Anomx was interrupted.",
            )

    def shutdown(self, session_path: Path | None = None) -> RuntimeCleanupResult:
        """Stop live runtime children before the CLI process exits."""

        self._turn_abort_event.set()
        workers_removed = self._remove_all_worker_states(session_path)
        processes_ended = self._end_all_process_states(session_path)
        return RuntimeCleanupResult(
            workers_removed=workers_removed,
            processes_ended=processes_ended,
        )

    def cleanup_session_runtime_state(self, session_path: Path) -> RuntimeCleanupResult:
        """Clear stale worker and process cards from a stored session transcript.

        Worker threads and command handles are process-local. After a CLI restart, any
        worker or running process shown by the transcript is stale even though the
        conversation history is still useful.
        """

        events = self.home.read_session_events(session_path)
        workers_removed = 0
        for worker in worker_snapshots(events):
            if self._remove_worker_by_id(worker.worker_id, session_path):
                workers_removed += 1
                continue
            self._append_stale_worker_removed_event(session_path, worker)
            workers_removed += 1

        processes_ended = 0
        for process in running_process_snapshots(events):
            if self._end_process_by_id_if_known(process.process_id, session_path):
                processes_ended += 1
                continue
            self.home.append_session_event(
                session_path,
                "process_event",
                {
                    "process_id": process.process_id,
                    "command": process.command,
                    "statement": process.statement,
                    "status": "ended",
                    "output": process.output,
                    "started_at": process.started_at,
                    "finished_at": utc_now_iso(),
                    "exit_code": process.exit_code,
                    "source": process.source,
                    "owner_id": process.owner_id,
                    "owner_name": process.owner_name,
                    "pid": process.pid,
                },
            )
            processes_ended += 1

        return RuntimeCleanupResult(
            workers_removed=workers_removed,
            processes_ended=processes_ended,
        )

    def _turn_aborted(self) -> bool:
        return self._turn_abort_event.is_set() or self.cancel_event.is_set()

    def backend_response(
        self,
        session_path: Path,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
    ) -> str:
        """Generate a backend response for the current session."""

        self._turn_abort_event.clear()
        previous_debug_session_path = self._debug_session_path
        self._debug_session_path = debug_session_path or session_path
        try:
            active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
            config = self.home.load_config()
            provider = str(config.get("provider", ""))
            model = str(config.get("model", ""))
            thinking_intensity = normalize_thinking_intensity(config.get("thinking_intensity"))
            if provider == "openai":
                return self.openai_response(
                    session_path,
                    model,
                    active_callbacks,
                    thinking_intensity=thinking_intensity,
                )
            if provider == "anthropic":
                return self.anthropic_response(
                    session_path,
                    model,
                    active_callbacks,
                    thinking_intensity=thinking_intensity,
                )
            if provider == "desy":
                return self.desy_response(session_path, model, active_callbacks)
            if provider == "ollama":
                return self.ollama_response(session_path, model, active_callbacks)
            return f"{provider}/{model} backend is unavailable."
        finally:
            self._debug_session_path = previous_debug_session_path

    def backend_response_for_prompt(
        self,
        prompt: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        debug_session_path: Path | None = None,
        parent_session_path: Path | None = None,
        worker_name: str = "",
        worker_id: str = "",
    ) -> str:
        """Generate a response for a one-off worker prompt."""

        resolved_worker_name = worker_name.strip() or self.process_owner_name or "Worker"
        resolved_worker_id = worker_id.strip() or self.process_owner_id or uuid4().hex[:8]
        session_path = self.home.append_worker_session_prompt(
            parent_session_path=parent_session_path or debug_session_path,
            worker_name=resolved_worker_name,
            worker_id=resolved_worker_id,
            prompt=prompt,
        )
        return self.backend_response(
            session_path,
            callbacks=callbacks,
            debug_session_path=debug_session_path,
        )

    def openai_response(
        self,
        session_path: Path,
        model: str,
        callbacks: RuntimeCallbacks | None = None,
        *,
        thinking_intensity: str | None = None,
    ) -> str:
        """Generate a response through the OpenAI Responses API."""

        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        api_key = self._api_key("openai", "OPENAI_API_KEY")
        if api_key is None:
            return self._missing_api_key_message("OpenAI", "OPENAI_API_KEY")

        self._status(active_callbacks.status)
        reasoning = self._openai_reasoning_config(model, thinking_intensity)
        payload: dict[str, Any] = {
            "model": model,
            "instructions": self._instructions(session_path),
            "input": self._openai_messages(self.conversation_messages(session_path), model),
            "reasoning": reasoning,
            "tools": self._openai_tools(),
            "tool_choice": "auto",
            "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
            "stream": True,
        }

        plan_finish_attempts = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self._turn_aborted():
                return ""
            self._status(active_callbacks.status)
            response = self._stream_openai_response(
                api_key,
                payload,
                active_callbacks.delta,
                active_callbacks.status,
            )
            if isinstance(response, str):
                return response
            if self._turn_aborted():
                return ""

            tool_outputs = self._execute_requested_tools(
                response,
                active_callbacks,
                session_path,
            )
            if not tool_outputs:
                continuation_prompt, used_plan_guard = self._continuation_prompt_after_text(
                    response.text,
                    active_callbacks,
                    session_path,
                    plan_finish_attempts,
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    if response.response_id is None:
                        return "OpenAI returned a continuation update without a response id."
                    payload = {
                        "model": model,
                        "instructions": self._instructions(session_path),
                        "previous_response_id": response.response_id,
                        "input": [
                            {
                                "role": "user",
                                "content": continuation_prompt,
                            }
                        ],
                        "reasoning": reasoning,
                        "tools": self._openai_tools(),
                        "tool_choice": "auto",
                        "max_tool_calls": OPENAI_MAX_TOOL_CALLS,
                        "stream": True,
                    }
                    continue
                return response.text

            if response.response_id is None:
                return "OpenAI requested tools but did not return a response id."

            payload = {
                "model": model,
                "instructions": self._instructions(session_path),
                "previous_response_id": response.response_id,
                "input": tool_outputs,
                "reasoning": reasoning,
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
        *,
        thinking_intensity: str | None = None,
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
            thinking_intensity=thinking_intensity,
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
        thinking_intensity: str | None = None,
    ) -> str:
        active_callbacks = RuntimeCallbacks() if callbacks is None else callbacks
        api_key = self._api_key(provider_key, env_var)
        if api_key is None:
            return self._missing_api_key_message(provider_label, env_var)

        self._status(active_callbacks.status)
        messages = self._anthropic_messages(
            self.conversation_messages(session_path),
            provider_key,
            model,
        )
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
            output_config = self._anthropic_output_config(model, thinking_intensity)
            if output_config:
                payload["output_config"] = output_config

        plan_finish_attempts = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self._turn_aborted():
                return ""
            self._status(active_callbacks.status)
            response = stream_response(
                api_key,
                payload,
                active_callbacks.delta,
                active_callbacks.status,
            )
            if isinstance(response, str):
                return response
            if self._turn_aborted():
                return ""

            tool_outputs = self._execute_anthropic_requested_tools(
                response,
                active_callbacks,
                session_path,
            )
            if not tool_outputs:
                text = response.text or self._extract_anthropic_text(response.content)
                continuation_prompt, used_plan_guard = self._continuation_prompt_after_text(
                    text,
                    active_callbacks,
                    session_path,
                    plan_finish_attempts,
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    assistant_content = list(response.content) or [
                        {"type": "text", "text": text}
                    ]
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append(
                        {
                            "role": "user",
                            "content": continuation_prompt,
                        }
                    )
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
                        output_config = self._anthropic_output_config(
                            model,
                            thinking_intensity,
                        )
                        if output_config:
                            payload["output_config"] = output_config
                    continue
                return text

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
                output_config = self._anthropic_output_config(model, thinking_intensity)
                if output_config:
                    payload["output_config"] = output_config

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
            *self._ollama_messages(self.conversation_messages(session_path), model),
        ]
        plan_finish_attempts = 0
        for _ in range(MAX_TOOL_ITERATIONS):
            if self._turn_aborted():
                return ""
            messages[0] = {"role": "system", "content": self._instructions(session_path)}
            response = self._stream_ollama_response(model, messages, active_callbacks)
            if isinstance(response, str):
                return response
            if self._turn_aborted():
                return ""

            if response.message:
                messages.append(response.message)
            if not response.tool_calls:
                text = response.text or "No response."
                continuation_prompt, used_plan_guard = self._continuation_prompt_after_text(
                    text,
                    active_callbacks,
                    session_path,
                    plan_finish_attempts,
                )
                if continuation_prompt is not None:
                    if used_plan_guard:
                        plan_finish_attempts += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": continuation_prompt,
                        }
                    )
                    continue
                return text

            messages.extend(
                self._execute_ollama_requested_tools(
                    response,
                    active_callbacks,
                    session_path,
                )
            )

        return f"Ollama tool loop stopped after {MAX_TOOL_ITERATIONS} tool batches."

    def conversation_messages(self, session_path: Path) -> list[dict[str, Any]]:
        """Return stored user/assistant messages for a backend conversation."""

        messages: list[dict[str, Any]] = []
        for event in self.home.read_session_events(session_path):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            backend_message = str(payload.get("backend_message", message)).strip()
            image_attachments = normalized_image_attachments(
                payload.get("image_attachments")
            )
            if event_type == "user_message" and (backend_message or image_attachments):
                user_message: dict[str, Any] = {
                    "role": "user",
                    "content": backend_message,
                }
                if image_attachments:
                    user_message["images"] = [
                        attachment.to_payload() for attachment in image_attachments
                    ]
                messages.append(user_message)
            elif event_type == "skill_invocation":
                prompt = str(payload.get("prompt", "")).strip()
                if prompt:
                    messages.append({"role": "user", "content": prompt})
            elif event_type == "agent_message" and message:
                messages.append({"role": "assistant", "content": message})
            elif event_type == "system_message" and message:
                messages.append({"role": "system", "content": message})
        return messages[-20:]

    def estimate_session_context_tokens(self, session_path: Path) -> int:
        """Estimate the current backend context for this runtime/session."""

        return estimate_backend_context_tokens(
            self._instructions(session_path),
            self.conversation_messages(session_path),
        )

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

    def suggest_session_continuation(self, session_path: Path, workspace_name: str) -> str:
        """Suggest the startup prompt copy for continuing an existing session."""

        messages = self.conversation_messages(session_path)
        fallback = self._heuristic_session_continuation(messages, workspace_name)
        if not messages:
            return fallback

        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        if provider == "openai":
            statement = self._suggest_openai_session_continuation(messages, model)
            if statement:
                return statement
        elif provider == "anthropic":
            statement = self._suggest_anthropic_session_continuation(messages, model)
            if statement:
                return statement
        elif provider == "desy":
            statement = self._suggest_desy_session_continuation(messages, model)
            if statement:
                return statement
        elif provider == "ollama":
            statement = self._suggest_ollama_session_continuation(messages, model)
            if statement:
                return statement
        return fallback

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
        def stream_once() -> OpenAIStreamResponse:
            self._debug_log_backend_request(
                "openai",
                payload,
                endpoint="https://api.openai.com/v1/responses",
            )
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
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    if self._turn_aborted():
                        return OpenAIStreamResponse(None, "", ())
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
            return OpenAIStreamResponse(
                response_id,
                "".join(text_parts).strip(),
                tuple(tool_calls),
            )

        if self._turn_aborted():
            return ""
        response = self._model_request_with_retries(
            provider_key="openai",
            provider_label="OpenAI",
            env_var="OPENAI_API_KEY",
            status_callback=status_callback,
            stream_once=stream_once,
        )
        if self._turn_aborted():
            return ""
        return cast(OpenAIStreamResponse | str, response)

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
        def stream_once() -> AnthropicStreamResponse | str:
            self._debug_log_backend_request(provider_key, payload, endpoint=endpoint)
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            text_parts: list[str] = []
            content_by_index: dict[int, dict[str, Any]] = {}
            tool_json_parts: dict[int, list[str]] = {}
            with urllib.request.urlopen(request, timeout=120) as response:
                for raw_line in response:
                    if self._turn_aborted():
                        return ""
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

            for index in tuple(tool_json_parts):
                self._finalize_anthropic_tool_input(content_by_index, tool_json_parts, index)

            ordered_content = tuple(
                content_by_index[index] for index in sorted(content_by_index)
            )
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

        if self._turn_aborted():
            return ""
        response = self._model_request_with_retries(
            provider_key=provider_key,
            provider_label=provider_label,
            env_var=env_var,
            status_callback=status_callback,
            stream_once=stream_once,
        )
        if self._turn_aborted():
            return ""
        return cast(AnthropicStreamResponse | str, response)

    def _model_request_with_retries(
        self,
        *,
        provider_key: str,
        provider_label: str,
        env_var: str,
        status_callback: StatusCallback | None,
        stream_once: Callable[[], ModelRequestStreamResponse],
    ) -> ModelRequestStreamResponse:
        retry_delays = MODEL_REQUEST_RETRY_DELAYS_SECONDS
        for attempt_index in range(len(retry_delays) + 1):
            try:
                return stream_once()
            except urllib.error.HTTPError as error:
                error_body = error.read().decode("utf-8", errors="replace")
                message = self._api_error(
                    provider_key,
                    provider_label,
                    env_var,
                    error.code,
                    error_body,
                )
                if (
                    error.code not in MODEL_REQUEST_RETRY_STATUS_CODES
                    or attempt_index >= len(retry_delays)
                ):
                    return message
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    f"HTTP {error.code}",
                    retry_delays[attempt_index],
                    attempt_index + 1,
                    len(retry_delays),
                    status_callback,
                ):
                    return ""
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                message = f"{provider_label} request failed: {error}"
                if attempt_index >= len(retry_delays):
                    return message
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    str(error),
                    retry_delays[attempt_index],
                    attempt_index + 1,
                    len(retry_delays),
                    status_callback,
                ):
                    return ""
        return f"{provider_label} request failed."

    def _sleep_before_model_request_retry(
        self,
        provider_label: str,
        failure: str,
        delay_seconds: float,
        retry_number: int,
        retry_count: int,
        status_callback: StatusCallback | None,
    ) -> bool:
        delay_text = self._format_retry_delay(delay_seconds)
        self._status(
            status_callback,
            (
                f"{provider_label} request failed ({failure}); retrying in "
                f"{delay_text}s ({retry_number}/{retry_count})"
            ),
        )
        deadline = time.monotonic() + delay_seconds
        while True:
            if self._turn_aborted():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(MODEL_REQUEST_RETRY_SLEEP_SLICE_SECONDS, remaining))

    def _format_retry_delay(self, delay_seconds: float) -> str:
        if float(delay_seconds).is_integer():
            return str(int(delay_seconds))
        return f"{delay_seconds:.1f}"

    def _debug_log_backend_request(
        self,
        provider: str,
        payload: Mapping[str, Any],
        *,
        endpoint: str = "",
        purpose: str = "chat",
    ) -> Path | None:
        actor = "worker" if self.role == AgentRole.WORKER else "orchestrator"
        with suppress(OSError, TypeError, ValueError):
            return self.home.write_backend_request_log(
                provider=provider,
                payload=payload,
                endpoint=endpoint,
                purpose=purpose,
                session_path=self._debug_session_path,
                actor=actor,
                worker_name=self.process_owner_name,
                worker_id=self.process_owner_id,
            )
        return None

    def _debug_log_process(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
    ) -> Path | None:
        log_session_path = session_path or process_state.session_path or self._debug_session_path
        if log_session_path is None:
            return None
        if process_state.source == "process":
            kind = "process"
        elif process_state.source in {"command", "worker_command"}:
            kind = "command"
        else:
            return None

        with self._process_lock:
            current = self._processes.get(process_state.process_id, process_state)
            if current.debug_log_written:
                return None
            current.debug_log_written = True
            output = "".join(current.output_chunks) or current.output
            payload = self._process_state_payload(current)
            if kind == "command":
                payload["command_id"] = current.process_id

        with suppress(OSError, TypeError, ValueError):
            return self.home.write_async_execution_log(
                session_path=log_session_path,
                kind=kind,
                payload=payload,
                output=output,
            )
        return None

    def _debug_log_crash(
        self,
        error: BaseException,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Path | None:
        with suppress(OSError, TypeError, ValueError):
            return self.home.write_crash_log(error, context=context)
        return None

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
        self._debug_log_backend_request(
            "ollama",
            payload,
            endpoint="http://127.0.0.1:11434/api/chat",
        )
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
                    if self._turn_aborted():
                        return ""
                    stripped = raw_line.decode("utf-8", errors="replace").strip()
                    if not stripped:
                        continue
                    data = cast(dict[str, Any], json.loads(stripped))
                    stream_message = data.get("message")
                    if not isinstance(stream_message, dict):
                        continue
                    thinking = stream_message.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        thinking_parts.append(thinking)
                        self._reasoning_status(callbacks.status, "".join(thinking_parts))
                    content = stream_message.get("content")
                    if isinstance(content, str) and content:
                        text_parts.append(content)
                        if callbacks.delta is not None:
                            callbacks.delta(content)
                    raw_tool_calls = stream_message.get("tool_calls")
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

        assistant_message: dict[str, Any] = {"role": "assistant"}
        if thinking_parts:
            assistant_message["thinking"] = "".join(thinking_parts)
        if text_parts:
            assistant_message["content"] = "".join(text_parts)
        if tool_calls:
            assistant_message["tool_calls"] = [
                self._ollama_tool_payload(tool_call) for tool_call in tool_calls
            ]
        return OllamaStreamResponse(
            "".join(text_parts).strip(),
            "".join(thinking_parts).strip(),
            tuple(tool_calls),
            assistant_message,
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
        if self._turn_aborted():
            return self._json_tool_result({"error": "Agent turn was aborted by user."})

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
            long_running_command: AsyncProcessState | None = None

            def publish_long_running_command(process: subprocess.Popen[str]) -> str | None:
                nonlocal long_running_command
                if (
                    session_path is None
                    or long_running_command is not None
                ):
                    return None
                process_id = uuid4().hex[:8]
                source = "worker_command" if self.role == AgentRole.WORKER else "command"
                long_running_command = AsyncProcessState(
                    process_id=process_id,
                    command=command,
                    statement=statement or "Running command",
                    status="running",
                    started_at=utc_now_iso(),
                    process=process,
                    source=source,
                    owner_id=self.process_owner_id,
                    owner_name=self.process_owner_name,
                    session_path=session_path,
                )
                with self._process_lock:
                    self._processes[long_running_command.process_id] = long_running_command
                self._publish_process_state(long_running_command, session_path, callbacks)
                self._start_process_monitor(long_running_command, session_path, callbacks)
                if self.role == AgentRole.WORKER and callbacks.tool_message is not None:
                    callbacks.tool_message(f"Waiting for tool call {process_id}")
                elif callbacks.status is not None:
                    callbacks.status(f"Waiting:{60.0}")
                return f"Command {process_id} is still running."

            if self.cancel_event.is_set():
                return self._json_tool_result(
                    {
                        "approved": False,
                        "output": "Worker was interrupted before the command could run.",
                    }
                )
            if self.role == AgentRole.WORKER and callbacks.tool_message is not None and statement:
                callbacks.tool_message(statement)
            result = self.tool_manager.run_command(
                command,
                statement or "Operator command",
                callbacks.approval,
                long_running_callback=publish_long_running_command,
            )
            tool_payload: dict[str, object] = {
                "approved": result.approved,
                "output": result.output,
            }
            command_history_output = result.output
            if long_running_command is not None:
                wait_payload = self._wait_for_command_state(long_running_command, callbacks)
                output = str(wait_payload.get("output") or result.output)
                tool_payload.update(wait_payload)
                tool_payload["approved"] = result.approved
                tool_payload["output"] = output
                command_history_output = output or str(wait_payload.get("status", ""))
                if callbacks.status is not None:
                    callbacks.status("Thinking")
            if self.role == AgentRole.WORKER and callbacks.command is not None:
                callbacks.command(statement, command, command_history_output)
            if self.role == AgentRole.OPERATOR and result.approved:
                statement_text = statement or self._default_operator_tool_statement(name)
                if callbacks.command is not None:
                    callbacks.command(statement_text, command, command_history_output)
                elif callbacks.tool_message is not None:
                    callbacks.tool_message(statement_text)
            self._emit_command_system_message(callbacks, result, statement)
            return self._json_tool_result(tool_payload)

        if self.role == AgentRole.WORKER:
            if name == "wait":
                return self._wait_tool(arguments, callbacks)
            if name == "check_command_status":
                return self._check_command_status_tool(arguments)
            if name == "kill_command":
                return self._kill_command_tool(arguments, session_path, callbacks)
            return self._json_tool_result({"error": f"Unknown worker tool: {name}"})

        if name in {
            "create_plan",
            "update_plan",
            "start_agent",
            "prompt_agent",
            "interrupt_agent",
            "remove_agent",
            "start_process",
            "end_process",
            "check_command_status",
            "kill_command",
            "ask_question",
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
        if name == "interrupt_agent":
            return self._interrupt_agent_tool(arguments, session_path)
        if name == "remove_agent":
            return self._remove_agent_tool(arguments, session_path)
        if name == "start_process":
            return self._start_process_tool(arguments, session_path, callbacks)
        if name == "end_process":
            return self._end_process_tool(arguments, session_path)
        if name == "check_command_status":
            return self._check_command_status_tool(arguments)
        if name == "kill_command":
            return self._kill_command_tool(arguments, session_path, callbacks)
        if name == "ask_question":
            return self._ask_question_tool(arguments, callbacks)
        if name == "check_agent":
            return self._check_agent_tool(arguments, callbacks)
        if name == "stop_agent":
            return self._interrupt_agent_tool(arguments, session_path)
        if name == "wait":
            return self._wait_tool(arguments, callbacks)
        if name == "remove_plan":
            return self._remove_plan_tool(arguments, session_path, callbacks)
        if name == "finish_anyways":
            return self._finish_anyways_tool(arguments, session_path, callbacks)
        return self._json_tool_result({"error": f"Unknown tool: {name}"})

    def _continuation_prompt_after_text(
        self,
        message: str,
        callbacks: RuntimeCallbacks,
        session_path: Path | None,
        plan_finish_attempts: int,
    ) -> tuple[str | None, bool]:
        if self._running_command_states():
            delivered = message.strip()
            if self.role == AgentRole.OPERATOR and delivered and callbacks.message is not None:
                callbacks.message(delivered)
            elif self.role == AgentRole.WORKER and callbacks.tool_message is not None:
                command = self._running_command_states()[0]
                callbacks.tool_message(f"Waiting for tool call {command.process_id}")
            wait_output = self._wait_tool({}, callbacks)
            return self._command_continuation_prompt(wait_output), False

        if self.role != AgentRole.OPERATOR or not self._running_worker_states():
            return self._plan_finish_continuation_prompt(
                session_path,
                plan_finish_attempts,
            )
        delivered = message.strip()
        if delivered and callbacks.message is not None:
            callbacks.message(delivered)
        wait_output = self._wait_tool({}, callbacks)
        return self._worker_continuation_prompt(wait_output), False

    def _plan_finish_continuation_prompt(
        self,
        session_path: Path | None,
        plan_finish_attempts: int,
    ) -> tuple[str | None, bool]:
        if self.role != AgentRole.OPERATOR or session_path is None:
            return None, False
        plan_steps = latest_plan_steps(self.home.read_session_events(session_path))
        if not plan_steps:
            return None, False
        if all(step.is_done for step in plan_steps):
            self.home.append_session_event(session_path, "plan_update", {"steps": []})
            return None, False
        if plan_finish_attempts >= MAX_PLAN_FINISH_REPROMPTS:
            return None, False

        self.home.append_session_event(
            session_path,
            "work_message",
            {
                "message": "Validating whether the plan is finished",
                "role": "tool",
            },
        )
        return self._unfinished_plan_finish_prompt(plan_steps), True

    def _unfinished_plan_finish_warning(
        self,
        plan_steps: tuple[PlanStep, ...],
    ) -> str:
        open_titles = ", ".join(step.title for step in plan_steps if not step.is_done)
        if not open_titles:
            open_titles = "unknown open steps"
        return (
            "There is still an active plan that is not fully done. "
            "Are you sure you want to finish? Open plan steps: "
            f"{open_titles}."
        )

    def _unfinished_plan_finish_prompt(
        self,
        plan_steps: tuple[PlanStep, ...],
    ) -> str:
        lines = [
            "Your previous assistant message was not delivered because there is still "
            "an active plan with unfinished steps. Do not finish yet unless the plan is "
            "genuinely complete.",
            "",
            "Current plan:",
        ]
        for step in plan_steps:
            state = "done" if step.is_done else "open"
            lines.append(f"{step.position}. [{state}] {step.title}: {step.description}")
        lines.extend(
            [
                "",
                "Continue working, update the plan, or remove it if it is stale. "
                "If every item is complete, mark the plan done; the UI will remove it "
                "automatically when you finish. If you are sure the final answer should "
                "be delivered despite open plan steps, call finish_anyways(statement).",
            ]
        )
        return "\n".join(lines)

    def _worker_continuation_prompt(self, wait_output: str) -> str:
        return (
            "Your previous assistant message was delivered to the user as an intermediate "
            "progress update because worker agents were still working. It was not a final "
            "answer. Continue orchestrating the task. Worker wait result:\n"
            f"{wait_output}\n\n"
            "If all required workers are ready or interrupted, inspect their results and "
            "produce the final answer. If workers are still working, use wait, "
            "output_message, or interrupt_agent as appropriate."
        )

    def _command_continuation_prompt(self, wait_output: str) -> str:
        return (
            "A long-running command tool call is still active or just finished. Continue "
            "from the command wait result below. Do not produce a final answer while a "
            "required command is still running; use wait, check_command_status, or "
            "kill_command as appropriate.\n"
            f"{wait_output}"
        )

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
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "remove_plan requires a session."})
        self.home.append_session_event(session_path, "plan_update", {"steps": []})
        statement = str(arguments.get("statement", "")).strip() or "Removed plan"
        if callbacks.command is not None:
            callbacks.command(statement, self._operator_tool_detail("remove_plan", arguments), "")
        elif callbacks.tool_message is not None:
            callbacks.tool_message(statement)
        elif callbacks.status is not None:
            callbacks.status(statement)
        return self._json_tool_result({"removed": True})

    def _finish_anyways_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "finish_anyways requires a session."})
        self.home.append_session_event(session_path, "plan_update", {"steps": []})
        statement = str(arguments.get("statement", "")).strip() or "Finishing anyway"
        if callbacks.command is not None:
            callbacks.command(
                statement,
                self._operator_tool_detail("finish_anyways", arguments),
                "",
            )
        elif callbacks.tool_message is not None:
            callbacks.tool_message(statement)
        elif callbacks.status is not None:
            callbacks.status(statement)
        return self._json_tool_result({"finish_anyways": True, "removed_plan": True})

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
        worker = self._start_worker_agent(
            prompt=prompt,
            name=None,
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

    def _interrupt_agent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        worker_id = str(arguments.get("agent_id") or arguments.get("worker_id") or "").strip()
        if not worker_id:
            return self._json_tool_result({"error": "interrupt_agent requires an agent_id."})

        with self._worker_lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return self._json_tool_result(
                    {"interrupted": False, "error": "Unknown agent id."}
                )
            worker.cancel_event.set()
            if worker.status == WORKER_STATE_WORKING:
                worker.status = WORKER_STATE_INTERRUPTED
                worker.statement = ""
                worker.finished_at = utc_now_iso()
                self._append_worker_event(session_path, worker)
                self._append_worker_system_message(
                    session_path,
                    worker,
                    WORKER_STATE_INTERRUPTED,
                    "Worker was interrupted by the Operator.",
                )
        return self._json_tool_result({"interrupted": True, "agent_id": worker_id})

    def _remove_agent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        worker_id = str(arguments.get("agent_id") or arguments.get("worker_id") or "").strip()
        if not worker_id:
            return self._json_tool_result({"error": "remove_agent requires an agent_id."})

        with self._worker_lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return self._json_tool_result({"removed": False, "error": "Unknown agent id."})
            worker.cancel_event.set()
            worker.status = WORKER_STATE_REMOVED
            worker.statement = ""
            worker.finished_at = worker.finished_at or utc_now_iso()
            self._append_worker_event(session_path, worker)
            del self._workers[worker_id]
        return self._json_tool_result({"removed": True, "agent_id": worker_id})

    def _remove_worker_by_id(
        self,
        worker_id: str,
        session_path: Path | None = None,
    ) -> bool:
        with self._worker_lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False
            self._remove_worker_locked(worker, session_path)
            return True

    def _remove_all_worker_states(self, session_path: Path | None = None) -> int:
        with self._worker_lock:
            workers = tuple(self._workers.values())
            for worker in workers:
                self._remove_worker_locked(worker, session_path)
        return len(workers)

    def _remove_worker_locked(
        self,
        worker: WorkerAgentState,
        session_path: Path | None = None,
    ) -> None:
        worker.cancel_event.set()
        worker.status = WORKER_STATE_REMOVED
        worker.statement = ""
        worker.finished_at = worker.finished_at or utc_now_iso()
        self._append_worker_event(worker.session_path or session_path, worker)
        self._workers.pop(worker.worker_id, None)

    def _append_stale_worker_removed_event(
        self,
        session_path: Path,
        worker: WorkerAgentSnapshot,
    ) -> None:
        self.home.append_session_event(
            session_path,
            "worker_event",
            {
                "worker_id": worker.worker_id,
                "name": worker.name,
                "status": WORKER_STATE_REMOVED,
                "state": WORKER_STATE_REMOVED,
                "statement": "",
                "prompt": worker.prompt,
                "response": worker.response,
                "started_at": worker.started_at,
                "finished_at": worker.finished_at or utc_now_iso(),
                "context_tokens": worker.context_tokens,
                "context_percent": worker.context_percent,
            },
        )

    def _stop_running_workers(self, session_path: Path | None, message: str) -> None:
        with self._worker_lock:
            running_workers = [
                worker
                for worker in self._workers.values()
                if worker.status == WORKER_STATE_WORKING
            ]
            for worker in running_workers:
                worker.cancel_event.set()
                worker.status = WORKER_STATE_INTERRUPTED
                worker.statement = ""
                worker.finished_at = utc_now_iso()
                self._append_worker_event(session_path, worker)
                self._append_worker_system_message(
                    session_path,
                    worker,
                    WORKER_STATE_INTERRUPTED,
                    message,
                )

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
            payload: dict[str, object] = {
                "agent_id": worker.worker_id,
                "name": worker.name,
                "status": worker.status,
                "statement": worker.statement,
                "commands": list(worker.command_history),
            }
        with self._process_lock:
            worker_commands = [
                process_state
                for process_state in self._processes.values()
                if process_state.source == "worker_command"
                and process_state.owner_id == worker_id
            ]
        payload["long_running_commands"] = [
            self._command_state_payload(process_state)
            for process_state in worker_commands
        ]
        return self._json_tool_result(payload)

    def _start_process_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "start_process requires a session."})

        command = str(arguments.get("command", "")).strip()
        statement = str(arguments.get("statement", "")).strip() or "Starting process"
        if not command:
            return self._json_tool_result({"error": "start_process requires a command."})

        result = self.tool_manager.start_process(command, statement, callbacks.approval)
        self._emit_command_system_message(callbacks, result, statement)
        if result.process is None:
            return self._json_tool_result(
                {
                    "approved": result.approved,
                    "started": False,
                    "output": result.output,
                }
            )

        process_id = uuid4().hex[:8]
        process_state = AsyncProcessState(
            process_id=process_id,
            command=command,
            statement=statement,
            status="running",
            started_at=utc_now_iso(),
            process=result.process,
            session_path=session_path,
        )
        with self._process_lock:
            self._processes[process_id] = process_state

        self._publish_process_state(process_state, session_path, callbacks)
        self._start_process_monitor(process_state, session_path, callbacks)
        return self._json_tool_result(
            {
                "approved": True,
                "started": True,
                "process_id": process_id,
                "status": process_state.status,
                "command": command,
            }
        )

    def _end_process_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        process_id = str(arguments.get("process_id") or "").strip()
        if not process_id:
            return self._json_tool_result({"error": "end_process requires a process_id."})
        return self.end_process(process_id, session_path)

    def _ask_question_tool(
        self,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks,
    ) -> str:
        if callbacks.question is None:
            return self._json_tool_result(
                {"answered": False, "cancelled": True, "error": "No interactive UI callback."}
            )

        request_or_error = self._question_request(arguments)
        if isinstance(request_or_error, str):
            return self._json_tool_result(
                {"answered": False, "cancelled": True, "error": request_or_error}
            )

        response = callbacks.question(request_or_error)
        return self._json_tool_result(
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

    def end_process(self, process_id: str, session_path: Path | None = None) -> str:
        """End a running async process by id and append an updated process event."""

        return self._end_process(process_id, session_path)

    def _end_process_by_id_if_known(
        self,
        process_id: str,
        session_path: Path | None = None,
    ) -> bool:
        with self._process_lock:
            if process_id not in self._processes:
                return False
        self._end_process(process_id, session_path)
        return True

    def _end_all_process_states(self, session_path: Path | None = None) -> int:
        with self._process_lock:
            running_processes = tuple(
                process_state
                for process_state in self._processes.values()
                if process_state.status == "running"
            )

        ended_count = 0
        for process_state in running_processes:
            result = self._end_process(
                process_state.process_id,
                process_state.session_path or session_path,
            )
            with suppress(json.JSONDecodeError):
                if json.loads(result).get("ended") is True:
                    ended_count += 1
        return ended_count

    def _end_process(
        self,
        process_id: str,
        session_path: Path | None = None,
        callbacks: RuntimeCallbacks | None = None,
        *,
        allowed_sources: set[str] | None = None,
    ) -> str:
        with self._process_lock:
            process_state = self._processes.get(process_id)
            if process_state is None:
                return self._json_tool_result(
                    {"ended": False, "error": "Unknown process id."}
                )
            if allowed_sources is not None and process_state.source not in allowed_sources:
                return self._json_tool_result(
                    {"ended": False, "error": "Unknown command id."}
                )
            if process_state.status != "running":
                return self._json_tool_result(
                    {
                        "ended": True,
                        "process_id": process_id,
                        "status": process_state.status,
                    }
                )
            process_state.status = "ended"
            process_state.finished_at = utc_now_iso()
            self._publish_process_state(
                process_state,
                session_path or process_state.session_path,
                callbacks,
            )
            process = process_state.process

        self.tool_manager.terminate_process(process)
        with self._process_lock:
            process_state.exit_code = process.poll()
            self._publish_process_state(
                process_state,
                session_path or process_state.session_path,
                callbacks,
            )
        return self._json_tool_result(
            {"ended": True, "process_id": process_id, "status": process_state.status}
        )

    def _monitor_process(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        readers: list[threading.Thread] = []
        for stream in (process_state.process.stdout, process_state.process.stderr):
            if stream is None:
                continue
            reader = threading.Thread(
                target=self._read_process_stream,
                args=(process_state, stream),
                daemon=True,
            )
            reader.start()
            readers.append(reader)

        return_code: int | None
        try:
            return_code = process_state.process.wait()
        except OSError as error:  # pragma: no cover - defensive process boundary
            self._append_process_output(process_state, str(error))
            return_code = process_state.process.poll()

        for reader in readers:
            reader.join(timeout=1)

        should_log = False
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.exit_code = return_code
            if current.status == "running":
                current.status = "ended"
                current.finished_at = utc_now_iso()
                self._publish_process_state(current, session_path, callbacks)
            should_log = True
        if should_log:
            self._debug_log_process(process_state, session_path)

    def _read_process_stream(
        self,
        process_state: AsyncProcessState,
        stream: TextIO,
    ) -> None:
        try:
            while True:
                chunk = stream.readline()
                if chunk == "":
                    break
                self._append_process_output(process_state, chunk)
        except OSError as error:  # pragma: no cover - defensive process boundary
            self._append_process_output(process_state, str(error))

    def _append_process_output(
        self,
        process_state: AsyncProcessState,
        chunk: str,
    ) -> None:
        if not chunk:
            return
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.output_chunks.append(chunk)
            current.output = self._compact_process_output("".join(current.output_chunks))

    def _publish_process_state(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        self._append_process_event(session_path or process_state.session_path, process_state)
        if callbacks is not None and callbacks.process is not None:
            callbacks.process(process_state)

    def _start_process_monitor(
        self,
        process_state: AsyncProcessState,
        session_path: Path | None,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        process_state.thread = threading.Thread(
            target=self._monitor_process,
            args=(process_state, session_path, callbacks),
            daemon=True,
        )
        process_state.thread.start()

    def _append_process_event(
        self,
        session_path: Path | None,
        process_state: AsyncProcessState,
    ) -> None:
        if session_path is None:
            return
        self.home.append_session_event(
            session_path,
            "process_event",
            self._process_state_payload(process_state),
        )

    def _process_state_payload(self, process_state: AsyncProcessState) -> dict[str, object]:
        return {
            "process_id": process_state.process_id,
            "command": process_state.command,
            "statement": process_state.statement,
            "status": process_state.status,
            "output": process_state.output,
            "started_at": process_state.started_at,
            "finished_at": process_state.finished_at,
            "exit_code": process_state.exit_code,
            "source": process_state.source,
            "owner_id": process_state.owner_id,
            "owner_name": process_state.owner_name,
            "pid": process_state.process.pid,
        }

    def _command_state_payload(self, process_state: AsyncProcessState) -> dict[str, object]:
        payload = self._process_state_payload(process_state)
        payload["command_id"] = process_state.process_id
        payload["output"] = self._current_process_output(process_state)
        return payload

    def _current_process_output(self, process_state: AsyncProcessState) -> str:
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return process_state.output
            if current.output_chunks:
                return self._compact_process_output("".join(current.output_chunks))
            return current.output

    def _compact_process_output(self, output: str) -> str:
        compact = output.strip()
        if len(compact) <= 2_000:
            return compact
        return f"{compact[-1_997:]}"

    def _command_tool_sources(self) -> set[str]:
        if self.role == AgentRole.WORKER:
            return {"worker_command"}
        return {"command"}

    def _command_states(self) -> tuple[AsyncProcessState, ...]:
        sources = self._command_tool_sources()
        with self._process_lock:
            return tuple(
                process_state
                for process_state in self._processes.values()
                if process_state.source in sources
            )

    def _running_command_states(self) -> tuple[AsyncProcessState, ...]:
        return tuple(
            process_state
            for process_state in self._command_states()
            if process_state.status == "running"
        )

    def _command_state(self, command_id: str) -> AsyncProcessState | None:
        sources = self._command_tool_sources()
        with self._process_lock:
            process_state = self._processes.get(command_id)
            if process_state is None or process_state.source not in sources:
                return None
            return process_state

    def _check_command_status_tool(self, arguments: dict[str, Any]) -> str:
        command_id = str(
            arguments.get("command_id") or arguments.get("process_id") or ""
        ).strip()
        if not command_id:
            return self._json_tool_result(
                {"error": "check_command_status requires a command_id."}
            )
        process_state = self._command_state(command_id)
        if process_state is None:
            return self._json_tool_result({"error": "Unknown command id."})
        return self._json_tool_result(self._command_state_payload(process_state))

    def _kill_command_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        command_id = str(
            arguments.get("command_id") or arguments.get("process_id") or ""
        ).strip()
        if not command_id:
            return self._json_tool_result({"error": "kill_command requires a command_id."})
        return self._end_process(
            command_id,
            session_path,
            callbacks,
            allowed_sources=self._command_tool_sources(),
        )

    def _wait_for_command_state(
        self,
        process_state: AsyncProcessState,
        callbacks: RuntimeCallbacks | None = None,
    ) -> dict[str, object]:
        seconds = 60.0
        started_at = time.monotonic()
        if callbacks is not None and callbacks.status is not None:
            callbacks.status(f"Waiting:{seconds}")
        deadline = started_at + seconds
        while time.monotonic() < deadline:
            if self._turn_aborted():
                break
            current = self._command_state(process_state.process_id)
            if current is None or current.status != "running":
                break
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

        waited_seconds = min(seconds, max(0.0, time.monotonic() - started_at))
        current = self._command_state(process_state.process_id) or process_state
        payload = self._command_state_payload(current)
        payload["waited_seconds"] = waited_seconds
        return payload

    def _wait_tool(
        self,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        del arguments
        seconds = 60.0
        started_at = time.monotonic()
        if not self._has_running_wait_targets():
            return self._json_tool_result(
                {
                    "waited_seconds": 0.0,
                    "workers": [
                        self._worker_state_payload(worker) for worker in self._worker_states()
                    ],
                    "commands": [
                        self._command_state_payload(command)
                        for command in self._command_states()
                    ],
                }
            )
        if callbacks is not None and callbacks.status is not None:
            callbacks.status(f"Waiting:{seconds}")
        deadline = started_at + seconds
        while time.monotonic() < deadline:
            if self._turn_aborted() or not self._has_running_wait_targets():
                break
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        waited_seconds = min(seconds, max(0.0, time.monotonic() - started_at))
        return self._json_tool_result(
            {
                "waited_seconds": waited_seconds,
                "workers": [self._worker_state_payload(worker) for worker in self._worker_states()],
                "commands": [
                    self._command_state_payload(command)
                    for command in self._command_states()
                ],
            }
        )

    def _has_running_wait_targets(self) -> bool:
        return bool(
            self._running_command_states()
            or (self.role == AgentRole.OPERATOR and self._running_worker_states())
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
            if requested_agent_id is not None and existing is None:
                return "Unknown agent id."
            if existing is not None and existing.status == WORKER_STATE_WORKING:
                return "That worker is still working."
            if requested_agent_id is not None and existing is not None and existing.status not in {
                WORKER_STATE_READY,
                WORKER_STATE_INTERRUPTED,
            }:
                return "prompt_agent can only prompt agents in ready or interrupted state."
            worker_id = requested_agent_id or uuid4().hex[:8]
            worker_name = name or (existing.name if existing is not None else "Worker")
            worker = WorkerAgentState(
                worker_id=worker_id,
                name=worker_name,
                prompt=prompt,
                status=WORKER_STATE_WORKING,
                statement="Thinking",
                started_at=utc_now_iso(),
                cancel_event=threading.Event(),
                command_history=[] if existing is None else existing.command_history,
                context_command_history=[],
                session_path=session_path,
            )
            self._refresh_worker_context(worker)
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
        def update_status(message: str) -> None:
            del message
            with self._worker_lock:
                if (
                    self._workers.get(worker.worker_id) is not worker
                    or worker.status != WORKER_STATE_WORKING
                    or not self._is_initial_worker_statement(worker.statement)
                ):
                    return
                if worker.statement != "Thinking":
                    worker.statement = "Thinking"
                    self._append_worker_event(operator_session_path, worker)

        def update_statement(message: str) -> None:
            statement = message.strip()
            if not statement:
                return
            with self._worker_lock:
                if (
                    self._workers.get(worker.worker_id) is not worker
                    or worker.status != WORKER_STATE_WORKING
                ):
                    return
                worker.statement = statement
                self._append_worker_event(operator_session_path, worker)

        def record_command(statement: str, command: str, output: str) -> None:
            with self._worker_lock:
                if (
                    self._workers.get(worker.worker_id) is not worker
                    or worker.status != WORKER_STATE_WORKING
                ):
                    return
                if statement:
                    worker.statement = statement
                command_record = {
                    "statement": statement or "Running command",
                    "command": command,
                    "output": self._compact_worker_output(output),
                }
                worker.command_history.append(command_record)
                worker.context_command_history.append(command_record)
                self._refresh_worker_context(worker)
                self._append_worker_event(operator_session_path, worker)

        def record_process(process_state: AsyncProcessState) -> None:
            with self._process_lock:
                self._processes[process_state.process_id] = process_state
            with self._worker_lock:
                current = self._workers.get(worker.worker_id)
                if current is worker and worker.status == WORKER_STATE_WORKING:
                    waiting_statement = f"Waiting for tool call {process_state.process_id}"
                    if process_state.status == "running":
                        worker.statement = waiting_statement
                    elif worker.statement == waiting_statement:
                        worker.statement = f"Tool call {process_state.process_id} finished"
                    self._append_worker_event(operator_session_path, worker)
            self._append_process_event(operator_session_path, process_state)

        worker_runtime = AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.tool_manager.mode,
            role=AgentRole.WORKER,
            cancel_event=worker.cancel_event,
            workspace_root=self.workspace_root,
            process_owner_id=worker.worker_id,
            process_owner_name=worker.name,
        )
        response = ""
        status = WORKER_STATE_READY
        try:
            response = worker_runtime.backend_response_for_prompt(
                worker.prompt,
                callbacks=RuntimeCallbacks(
                    status=update_status,
                    tool_message=update_statement,
                    command=record_command,
                    approval=operator_callbacks.approval,
                    process=record_process,
                ),
                debug_session_path=operator_session_path,
                parent_session_path=operator_session_path,
                worker_name=worker.name,
                worker_id=worker.worker_id,
            )
        except Exception as error:  # pragma: no cover - defensive thread boundary
            status = WORKER_STATE_INTERRUPTED
            response = f"Worker failed: {error}"
            crash_path = self._debug_log_crash(
                error,
                context={
                    "role": self.role.value,
                    "worker_id": worker.worker_id,
                    "worker_name": worker.name,
                    "operator_session_path": str(operator_session_path),
                },
            )
            if crash_path is not None:
                response = f"{response}\nCrash log: {crash_path}"

        with self._worker_lock:
            if self._workers.get(worker.worker_id) is not worker:
                return
            if (
                worker.cancel_event.is_set()
                or worker.status == WORKER_STATE_INTERRUPTED
                or worker.status == WORKER_STATE_REMOVED
            ):
                worker.status = WORKER_STATE_INTERRUPTED
                worker.statement = ""
                worker.finished_at = worker.finished_at or utc_now_iso()
                worker.response = response
                self._refresh_worker_context(worker)
                self._append_worker_event(operator_session_path, worker)
                return

            worker.status = status
            worker.statement = ""
            worker.finished_at = utc_now_iso()
            worker.response = response
            self._refresh_worker_context(worker)
            self._append_worker_event(operator_session_path, worker)

        self._append_worker_system_message(
            operator_session_path,
            worker,
            status,
            response,
        )

    def _is_initial_worker_statement(self, statement: str) -> bool:
        return not statement.strip() or statement.strip().lower() == "thinking"

    def _append_worker_event(
        self,
        session_path: Path | None,
        worker: WorkerAgentState,
    ) -> None:
        target_path = session_path or worker.session_path
        if target_path is None:
            return
        self.home.append_session_event(
            target_path,
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
        target_path = session_path or worker.session_path
        if target_path is None:
            return
        compact_response = response.strip() or "No worker response."
        self.home.append_session_event(
            target_path,
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
        return tuple(
            worker
            for worker in self._worker_states()
            if worker.status == WORKER_STATE_WORKING
        )

    def _worker_state_payload(self, worker: WorkerAgentState) -> dict[str, object]:
        return {
            "worker_id": worker.worker_id,
            "name": worker.name,
            "status": worker.status,
            "state": worker.status,
            "statement": worker.statement,
            "prompt": worker.prompt,
            "response": worker.response,
            "started_at": worker.started_at,
            "finished_at": worker.finished_at,
            "context_tokens": worker.context_tokens,
            "context_percent": worker.context_percent,
        }

    def _refresh_worker_context(self, worker: WorkerAgentState) -> None:
        tokens = self._estimate_worker_context_tokens(worker)
        model = str(self.home.load_config().get("model", ""))
        worker.context_tokens = tokens
        worker.context_percent = context_usage_percent(tokens, model_context_window(model))

    def _estimate_worker_context_tokens(self, worker: WorkerAgentState) -> int:
        return estimate_backend_context_tokens(
            self._worker_instructions(),
            self._worker_context_messages(worker),
        )

    def _worker_context_messages(
        self,
        worker: WorkerAgentState,
    ) -> tuple[dict[str, str], ...]:
        messages: list[dict[str, str]] = [{"role": "user", "content": worker.prompt}]
        if worker.context_command_history:
            command_blocks = [
                (
                    f"Statement: {command.get('statement', '')}\n"
                    f"Command: {command.get('command', '')}\n"
                    f"Output: {command.get('output', '')}"
                )
                for command in worker.context_command_history
            ]
            messages.append(
                {
                    "role": "system",
                    "content": "Worker command history:\n\n" + "\n\n".join(command_blocks),
                }
            )
        if worker.response:
            messages.append({"role": "assistant", "content": worker.response})
        return tuple(messages)

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
        statement = str(arguments.get("statement", "")).strip()
        statement = statement or self._default_operator_tool_statement(tool_name)
        if callbacks.command is not None:
            callbacks.command(statement, self._operator_tool_detail(tool_name, arguments), "")
            return
        callback = callbacks.tool_message or callbacks.status
        if callback is not None:
            callback(statement)

    def _operator_tool_detail(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        parameters = {
            key: value
            for key, value in arguments.items()
            if key != "statement"
        }
        if not parameters:
            return f"Tool: {tool_name}\nParameters: none"
        return (
            f"Tool: {tool_name}\n"
            "Parameters:\n"
            f"{json.dumps(parameters, indent=2, ensure_ascii=False, default=str)}"
        )

    def _default_operator_tool_statement(self, tool_name: str) -> str:
        return {
            "run_command": "Running command",
            "run_cli_command": "Running command",
            "create_plan": "Creating plan",
            "update_plan": "Updating plan",
            "start_agent": "Starting Worker",
            "prompt_agent": "Prompting Worker",
            "check_agent": "Checking Worker",
            "stop_agent": "Interrupting Worker",
            "interrupt_agent": "Interrupting Worker",
            "remove_agent": "Removing Worker",
            "start_process": "Starting process",
            "end_process": "Ending process",
            "check_command_status": "Checking command",
            "kill_command": "Killing command",
            "ask_question": "Asking question",
            "remove_plan": "Removing plan",
            "finish_anyways": "Finishing anyway",
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
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
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

    def _suggest_openai_session_continuation(
        self,
        messages: list[dict[str, Any]],
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
                    "instructions": self._continuation_system_prompt(),
                    "input": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_output_tokens": 48,
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
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_openai_text(data))

    def _suggest_anthropic_session_continuation(
        self,
        messages: list[dict[str, Any]],
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
                    "system": self._continuation_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_tokens": 48,
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
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_anthropic_text(data))

    def _suggest_desy_session_continuation(
        self,
        messages: list[dict[str, Any]],
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
                    "system": self._continuation_system_prompt(),
                    "messages": [
                        {
                            "role": "user",
                            "content": self._title_prompt(messages),
                        }
                    ],
                    "max_tokens": 48,
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
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        return self._sanitize_continuation_statement(self.extract_anthropic_text(data))

    def _suggest_ollama_session_continuation(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._continuation_system_prompt()},
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
            with urllib.request.urlopen(request, timeout=8) as response:
                data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError):
            return None
        message = data.get("message")
        if not isinstance(message, dict):
            return None
        return self._sanitize_continuation_statement(str(message.get("content", "")))

    def _continuation_system_prompt(self) -> str:
        return (
            "Write one concise second-person question for an Anomx startup resume prompt. "
            "Ask whether to continue the previous CLI session and mention the concrete "
            "work if it is clear. Start with 'Do you want to continue'. Use 14 to 28 "
            "words. Return only the question. No quotes."
        )

    def _title_prompt(self, messages: list[dict[str, Any]]) -> str:
        conversation = "\n".join(
            f"{message.get('role', '')}: {message.get('content', '')}"
            for message in messages[-6:]
        )
        return f"Conversation:\n{conversation}"

    def _heuristic_session_continuation(
        self,
        messages: list[dict[str, Any]],
        workspace_name: str,
    ) -> str:
        topic = self._heuristic_session_title(messages)
        if topic:
            return f"Do you want to continue the previous session titled {topic}?"
        return f"Do you want to continue the previous Anomx session in {workspace_name}?"

    def _heuristic_session_title(self, messages: list[dict[str, Any]]) -> str | None:
        first_user_message = next(
            (
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        return self._sanitize_title(first_user_message)

    def _sanitize_continuation_statement(self, statement: str) -> str | None:
        cleaned = " ".join(statement.strip().strip("\"'`").split())
        if not cleaned:
            return None
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        words = cleaned.split()
        if len(words) > 32:
            cleaned = " ".join(words[:32]).rstrip("?") + "?"
        return cleaned[:180] or None

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
        if self.role == AgentRole.WORKER:
            return self._worker_instructions()

        tools = "\n".join(f"- {tool}" for tool in self._operator_tool_descriptions())
        runtime_context = self._operator_runtime_context(session_path)
        return "\n\n".join(
            [
                OPERATOR_SYSTEM_PROMPT,
                *self._instruction_environment_sections(),
                runtime_context,
                f"Available tools:\n{tools}",
            ]
        )

    def _instruction_environment_sections(self) -> list[str]:
        sections = [self.tool_manager.mode.system_prompt_statement]
        sections.append("\n".join(self.tool_manager.workspace_prompt_lines()))
        session_policy = self.tool_manager.session_policy_prompt_lines()
        if session_policy:
            sections.append("\n".join(session_policy))
        return sections

    def _operator_tool_descriptions(self) -> tuple[str, ...]:
        descriptions = list(OPERATOR_TOOL_DESCRIPTIONS)
        if self._running_command_states():
            descriptions.extend(
                [
                    (
                        "check_command_status(command_id): inspect your own active "
                        "long-running command and read its current CLI output."
                    ),
                    "kill_command(command_id): kill your own active long-running command.",
                ]
            )
        if self._running_worker_states() or self._running_command_states():
            descriptions.append(
                "wait(): wait 60 seconds for working Workers or your active commands."
            )
        return tuple(descriptions)

    def _worker_instructions(self) -> str:
        tools = "\n".join(f"- {tool}" for tool in self._worker_tool_descriptions())
        return "\n\n".join(
            [
                WORKER_SYSTEM_PROMPT,
                *self._instruction_environment_sections(),
                self._worker_runtime_context(),
                f"Available tools:\n{tools}",
            ]
        )

    def _worker_tool_descriptions(self) -> tuple[str, ...]:
        descriptions = list(WORKER_TOOL_DESCRIPTIONS)
        if self._running_command_states():
            descriptions.extend(
                [
                    (
                        "check_command_status(command_id): inspect one of your active "
                        "long-running command tool calls and read its current CLI output."
                    ),
                    (
                        "kill_command(command_id): kill one of your active long-running "
                        "command tool calls."
                    ),
                    "wait(): wait 60 seconds for your active command tool calls.",
                ]
            )
        return tuple(descriptions)

    def _worker_runtime_context(self) -> str:
        commands = self._command_states()
        if not commands:
            return "Worker runtime context:\n- Active command tool calls: none."
        lines = ["Worker runtime context:", "- Active command tool calls:"]
        for command in commands:
            lines.append(
                "  "
                f"{command.process_id} · state={command.status} · "
                f"{command.statement or command.command} · "
                f"running for {self._runtime_duration(command.started_at)} · "
                f"command: {command.command}"
            )
        return "\n".join(lines)

    def _operator_runtime_context(self, session_path: Path | None) -> str:
        if session_path is None:
            return "Runtime context:\n- No active session context."

        events = self.home.read_session_events(session_path)
        plan_steps = latest_plan_steps(events)
        workers = worker_snapshots(events)
        processes = running_process_snapshots(events)
        lines = ["Runtime context:"]
        if plan_steps:
            lines.append("- Current plan:")
            for step in plan_steps:
                state = "done" if step.is_done else "open"
                lines.append(f"  {step.position}. [{state}] {step.title}: {step.description}")
            lines.append(
                "- Plan guidance: keep the current plan up to date as work changes. "
                "Ideally, every plan step should be marked done before ending operations; "
                "update or remove stale steps instead of leaving them open."
            )
        else:
            lines.append("- Current plan: none.")

        if workers:
            lines.append("- Worker agents:")
            for worker in workers:
                context = self._worker_context_runtime_label(worker.context_percent)
                if worker.status == WORKER_STATE_WORKING:
                    statement = worker.statement or "Thinking"
                    lines.append(
                        "  "
                        f"{worker.worker_id} · {worker.name} · state=working · "
                        f"{context} · {statement} · working for "
                        f"{self._runtime_duration(worker.started_at)}"
                    )
                else:
                    lines.append(
                        "  "
                        f"{worker.worker_id} · {worker.name} · state={worker.status} · "
                        f"{context}"
                    )
        else:
            lines.append("- Worker agents: none.")

        if processes:
            lines.append("- Async processes:")
            for process in processes:
                label = process.statement or process.command
                source_label = self._process_runtime_source_label(process)
                lines.append(
                    "  "
                    f"{process.process_id} · {source_label} · state=running · {label} · "
                    f"running for {self._runtime_duration(process.started_at)} · "
                    f"command: {process.command}"
                )
        else:
            lines.append("- Async processes: none.")
        return "\n".join(lines)

    def _worker_context_runtime_label(self, context_percent: int) -> str:
        if context_percent <= 0:
            return "context=unknown"
        return f"context={context_percent}%"

    def _process_runtime_source_label(self, process: object) -> str:
        source = str(getattr(process, "source", "")).strip()
        if source == "command":
            return "operator command"
        if source == "worker_command":
            owner_name = str(getattr(process, "owner_name", "")).strip()
            owner_id = str(getattr(process, "owner_id", "")).strip()
            owner = owner_name or owner_id or "worker"
            return f"worker command owned by {owner}"
        return "process"

    def _worker_runtime_duration(self, started_at: str) -> str:
        return self._runtime_duration(started_at)

    def _runtime_duration(self, started_at: str) -> str:
        with suppress(ValueError):
            from datetime import UTC, datetime

            started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            seconds = max(0, int((datetime.now(tz=UTC) - started).total_seconds()))
            minutes, remaining_seconds = divmod(seconds, 60)
            return f"{minutes:02d}:{remaining_seconds:02d}"
        return "unknown"

    def _openai_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input("openai", model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            image_blocks = [
                block
                for image in images
                if (block := self._openai_image_block(image)) is not None
            ]
            if not content and not image_blocks:
                continue
            if image_blocks and role == "user":
                content_blocks: list[dict[str, Any]] = []
                text = self._content_with_image_labels(content, images)
                if text:
                    content_blocks.append({"type": "input_text", "text": text})
                content_blocks.extend(image_blocks)
                converted.append({"role": role, "content": content_blocks})
            else:
                converted.append({"role": role, "content": content})
        return converted

    def _anthropic_messages(
        self,
        messages: list[dict[str, Any]],
        provider_key: str,
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input(provider_key, model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            blocks = self._anthropic_content_blocks(content, images)
            if not blocks:
                continue
            if role == "assistant":
                self._append_anthropic_message(converted, "assistant", content)
            elif role == "system":
                self._append_anthropic_message(converted, "user", f"[System note]\n{content}")
            else:
                self._append_anthropic_blocks(converted, "user", blocks)
        return converted

    def _append_anthropic_message(
        self,
        messages: list[dict[str, Any]],
        role: str,
        text: str,
    ) -> None:
        self._append_anthropic_blocks(messages, role, ({"type": "text", "text": text},))

    def _append_anthropic_blocks(
        self,
        messages: list[dict[str, Any]],
        role: str,
        blocks: Iterable[dict[str, Any]],
    ) -> None:
        content_blocks = list(blocks)
        if not content_blocks:
            return
        if messages and messages[-1].get("role") == role:
            content = messages[-1].get("content")
            if isinstance(content, list):
                content.extend(content_blocks)
                return
        messages.append({"role": role, "content": content_blocks})

    def _anthropic_content_blocks(
        self,
        content: str,
        images: tuple[ImageAttachment, ...],
    ) -> tuple[dict[str, Any], ...]:
        blocks: list[dict[str, Any]] = []
        text = self._content_with_image_labels(content, images)
        if text:
            blocks.append({"type": "text", "text": text})
        for image in images:
            block = self._anthropic_image_block(image)
            if block is not None:
                blocks.append(block)
        return tuple(blocks)

    def _ollama_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        supports_images = backend_supports_image_input("ollama", model)
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "")).strip()
            images = (
                normalized_image_attachments(message.get("images"))
                if role == "user" and supports_images
                else ()
            )
            encoded_images = [
                encoded
                for image in images
                if (encoded := self._image_base64(image)) is not None
            ]
            if not content and not encoded_images:
                continue
            converted_message: dict[str, Any] = {
                "role": role,
                "content": self._content_with_image_labels(content, images),
            }
            if encoded_images and role == "user":
                converted_message["images"] = encoded_images
            converted.append(converted_message)
        return converted

    def _content_with_image_labels(
        self,
        content: str,
        images: tuple[ImageAttachment, ...],
    ) -> str:
        text = content.strip()
        if not images:
            return text
        image_lines = "\n".join(f"- {image.label}" for image in images)
        attachment_note = f"Attached images:\n{image_lines}"
        return f"{text}\n\n{attachment_note}" if text else attachment_note

    def _openai_image_block(self, image: ImageAttachment) -> dict[str, str] | None:
        encoded = self._image_base64(image)
        if encoded is None:
            return None
        return {
            "type": "input_image",
            "image_url": f"data:{image.mime_type};base64,{encoded}",
        }

    def _anthropic_image_block(self, image: ImageAttachment) -> dict[str, Any] | None:
        encoded = self._image_base64(image)
        if encoded is None:
            return None
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": encoded,
            },
        }

    def _image_base64(self, image: ImageAttachment) -> str | None:
        with suppress(OSError):
            return base64.b64encode(image.path.read_bytes()).decode("ascii")
        return None

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

    def _openai_reasoning_config(
        self,
        model: str,
        thinking_intensity: str | None,
    ) -> dict[str, Any]:
        reasoning: dict[str, Any] = {"summary": "auto"}
        intensity = self._supported_thinking_intensity("openai", model, thinking_intensity)
        if intensity is not None:
            reasoning["effort"] = intensity
        return reasoning

    def _anthropic_output_config(
        self,
        model: str,
        thinking_intensity: str | None,
    ) -> dict[str, Any]:
        intensity = self._supported_thinking_intensity("anthropic", model, thinking_intensity)
        return {} if intensity is None else {"effort": intensity}

    def _supported_thinking_intensity(
        self,
        provider_key: str,
        model: str,
        thinking_intensity: str | None,
    ) -> str | None:
        intensity = normalize_thinking_intensity(thinking_intensity)
        if intensity == THINKING_INTENSITY_AUTO:
            return None
        supported = {option.value for option in thinking_intensity_options(provider_key, model)}
        return intensity if intensity in supported else None

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
            tools = [
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
                                "description": (
                                    "A single CLI command, for example 'ls -la'. Shell "
                                    "operators and redirection may be used when necessary; "
                                    "paths must resolve inside the trusted workspace root."
                                ),
                            },
                        },
                        "required": ["statement", "command"],
                        "additionalProperties": False,
                    },
                }
            ]
            if self._running_command_states():
                tools.extend(self._command_control_tool_definitions())
                tools.append(self._wait_tool_definition("active command tool calls"))
            return tools

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
                            "description": (
                                "A single CLI command, for example 'ls -la'. Shell "
                                "operators and redirection may be used when necessary; "
                                "paths must resolve inside the trusted workspace root."
                            ),
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
                "description": "Prompt a ready or interrupted Worker again.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Existing ready or interrupted Worker id.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Specific task prompt for the Worker.",
                        },
                    },
                    "required": ["statement", "agent_id", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "interrupt_agent",
                "description": "Interrupt a working Worker agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Working Worker id to interrupt.",
                        },
                    },
                    "required": ["statement", "agent_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "remove_agent",
                "description": "Remove a Worker agent from active context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Worker id to remove.",
                        },
                    },
                    "required": ["statement", "agent_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "start_process",
                "description": "Start a long-running async CLI process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "command": {
                            "type": "string",
                            "description": (
                                "Long-running CLI command, for example 'npm run dev'. "
                                "It continues after the agent turn until ended."
                            ),
                        },
                    },
                    "required": ["statement", "command"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "end_process",
                "description": "End a running async CLI process.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "process_id": {
                            "type": "string",
                            "description": "Async process id to end.",
                        },
                    },
                    "required": ["statement", "process_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "ask_question",
                "description": "Ask the user an interactive question in the bottom panel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
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
                    "required": [
                        "statement",
                        "question",
                        "kind",
                        "options",
                        "placeholder",
                        "default",
                        "allow_custom",
                    ],
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
            {
                "name": "finish_anyways",
                "description": (
                    "Clear the current user-visible plan and allow final delivery after "
                    "the plan-finish checker asks for an explicit override."
                ),
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
        if self._running_command_states():
            tools.extend(self._command_control_tool_definitions())
        if self._running_worker_states() or self._running_command_states():
            tools.append(self._wait_tool_definition("working Workers or active commands"))
        return tools

    def _command_control_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "check_command_status",
                "description": (
                    "Check a currently running long-running command tool call and read "
                    "its current CLI output."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command_id": {
                            "type": "string",
                            "description": "Long-running command id to inspect.",
                        },
                    },
                    "required": ["command_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "kill_command",
                "description": "Kill a currently running long-running command tool call.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command_id": {
                            "type": "string",
                            "description": "Long-running command id to kill.",
                        },
                    },
                    "required": ["command_id"],
                    "additionalProperties": False,
                },
            },
        ]

    def _wait_tool_definition(self, target_description: str) -> dict[str, Any]:
        return {
            "name": "wait",
            "description": f"Wait up to 60 seconds for {target_description}.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }

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
        result: CommandResult | CommandProcessResult,
        statement: str,
    ) -> None:
        if callbacks.system_message is None:
            return
        if result.blocked_by_mode:
            return
        if result.safety == CommandSafety.FORBIDDEN:
            display_statement = statement.strip() or result.command
            callbacks.system_message(
                "forbidden",
                f"Blocked: {display_statement}\nCommand: {result.command}\nReason: {result.reason}",
            )
