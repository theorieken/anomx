"""Model backend and local tool runtime for the Anomx CLI agent."""

from __future__ import annotations

import base64
import html
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, MutableSet
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, TypeAlias, cast

if TYPE_CHECKING:
    from anomx.agent.helpers.sandbox import SandboxSession
from uuid import uuid4

from anomx.agent.agents import AgentKind, AgentSpec, agent_spec
from anomx.agent.helpers.debug import session_id_from_path
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.state import (
    PlanStep,
    SubagentSnapshot,
    build_plan_steps,
    latest_plan_steps,
    merge_plan_steps,
    running_process_snapshots,
    running_subagent_snapshots,
    serialize_plan_steps,
    subagent_snapshots,
)
from anomx.agent.helpers.tool_manager import (
    ApprovalCallback,
    ApprovalChoice,
    CliToolManager,
    CommandApprovalRequest,
    CommandProcessResult,
    CommandResult,
    CommandSafety,
    discover_workspace_root,
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

StatusCallback = Callable[[str], None]
MessageCallback = Callable[[str], None]
ToolMessageCallback = Callable[[str], None]
DeltaCallback = Callable[[str], None]
SystemMessageCallback = Callable[[str, str], None]
CommandCallback = Callable[[str, str, str], None]
FinishCallback = Callable[[str], None]

BUILD_TOOL_DESCRIPTIONS = (
    (
        "run_command(statement, command): run a safe CLI command for operator validation "
        "or inspection and persist statement as a working message."
    ),
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
    (
        "start_subagent(statement, agent_kind, name, prompt): start an async subagent "
        "of kind general, explore, or scout."
    ),
    "prompt_subagent(statement, agent_id, prompt): send another prompt to an idle subagent.",
    "remove_subagent(statement, agent_id): remove a subagent from prompt context and UI.",
    "get_subagent_info(agent_id): inspect the latest outputs from a subagent.",
)

MAX_TOOL_ITERATIONS = 128
OPENAI_MAX_TOOL_CALLS = 128
MAX_PLAN_FINISH_REPROMPTS = 3
DESY_MESSAGES_ENDPOINT = "https://assistant.desy.de/api/v1/messages"
MODEL_REQUEST_RETRY_STATUS_CODES = frozenset({400, 404, 429, 500, 502, 503})
MODEL_REQUEST_RETRY_COUNT = 10
MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS = 1.0
MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS = 60.0
MODEL_REQUEST_RETRY_BACKOFF_FACTOR = 2.0
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

    BUILD = "build"
    OPERATOR = "build"
    GENERAL = "general"
    EXPLORE = "explore"
    SCOUT = "scout"
    WORKER = "general"


SUBAGENT_EVENT_TYPE = "subagent_event"
SUBAGENT_MAX_CONCURRENT = 5
SUBAGENT_WAIT_SECONDS = 60.0 * 5
WEB_FETCH_MAX_CHARS = 20_000
WEB_SEARCH_MAX_RESULTS = 8




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
    last_output_event_at: float = 0.0
    last_output_event_text: str = ""
    thread: threading.Thread | None = None


@dataclass
class SubagentRuntimeState:
    """Mutable in-process state for one asynchronous subagent."""

    agent_id: str
    kind: AgentKind
    name: str
    prompt: str
    status: str
    statement: str
    started_at: str
    runtime: AgentRuntime | None = None
    session_path: Path | None = None
    worker: threading.Thread | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    response: str = ""
    error: str = ""
    finished_at: str = ""
    context_tokens: int = 0
    context_percent: int = 0
    command_history: list[dict[str, str]] = field(default_factory=list)


ProcessCallback = Callable[[AsyncProcessState], None]


@dataclass(frozen=True)
class RuntimeCleanupResult:
    """Summary of stale runtime state removed from a session transcript."""

    processes_ended: int = 0
    subagents_removed: int = 0
    workers_removed: int = 0


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
    finish: FinishCallback | None = None


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


ModelRequestStreamResponse: TypeAlias = (
    OpenAIStreamResponse | AnthropicStreamResponse | OllamaStreamResponse | str
)


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
        role: AgentRole | str = AgentRole.BUILD,
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
        self.agent_spec: AgentSpec = agent_spec(role)
        self.role = AgentRole(self.agent_spec.kind.value)
        self._turn_abort_event = threading.Event()
        self._processes: dict[str, AsyncProcessState] = {}
        self._process_lock = threading.Lock()
        self._subagents: dict[str, SubagentRuntimeState] = {}
        self._subagent_lock = threading.Lock()
        self._debug_session_path: Path | None = None
        self._parent_session_id: str = ""
        self.process_owner_id = process_owner_id
        self.process_owner_name = process_owner_name
        self._sandbox_session: SandboxSession | None = None

    @property
    def sandbox_session(self) -> SandboxSession | None:
        return self._sandbox_session

    def init_sandbox(
        self,
        config: Mapping[str, Any] | None = None,
        status_callback: StatusCallback | None = None,
    ) -> bool:
        """Initialise and start the sandbox container if sandbox is enabled.

        Returns True when sandbox is active (either started or not needed).
        """
        cfg = self.home.load_config() if config is None else dict(config)
        if not cfg.get("sandbox_enabled"):
            return True

        from anomx.agent.helpers.sandbox import (
            SandboxSession,
            sandbox_config_from_dict,
        )

        scfg = sandbox_config_from_dict(cfg)
        project_path = self.workspace_root or self.cwd
        sandbox_hash = self._load_sandbox_hash(project_path)
        self._sandbox_session = SandboxSession(
            scfg, project_path, sandbox_hash=sandbox_hash,
        )

        if status_callback:
            status_callback("Starting Sandbox")

        self._sandbox_session.start(status_callback=status_callback)
        return True

    def _load_sandbox_hash(self, project_path: Path) -> str:
        project = self.home.project_for_path(project_path)
        if project is not None and project.sandbox_hash:
            return project.sandbox_hash
        import hashlib
        raw = str(project_path.resolve()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:6]

    def set_mode(self, mode: AgentMode) -> None:
        """Set the active command execution mode."""

        self.tool_manager.set_mode(mode)

    def abort_current_turn(
        self,
        session_path: Path | None = None,
    ) -> None:
        """Request cancellation of the active response loop."""

        self._turn_abort_event.set()

    def shutdown(self, session_path: Path | None = None) -> RuntimeCleanupResult:
        """Stop live runtime children before the CLI process exits."""

        self._turn_abort_event.set()
        self._sandbox_session = None
        processes_ended = self._end_all_process_states(session_path)
        subagents_removed = self._end_all_subagent_states(session_path)
        processes_ended = self._end_all_process_states(session_path)
        subagents_removed = self._end_all_subagent_states(session_path)
        return RuntimeCleanupResult(
            processes_ended=processes_ended,
            subagents_removed=subagents_removed,
            workers_removed=subagents_removed,
        )

    def cleanup_session_runtime_state(self, session_path: Path) -> RuntimeCleanupResult:
        """Clear stale process cards from a stored session transcript.

        Process handles are process-local. After a CLI restart, any
        running process shown by the transcript is stale even though the
        conversation history is still useful.
        """

        events = self.home.read_session_events(session_path)
        processes_ended = 0
        subagents_removed = 0
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
                    "status": "ended",
                    "statement": process.statement,
                    "output": process.output,
                    "started_at": process.started_at,
                    "finished_at": process.finished_at or "",
                    "exit_code": process.exit_code,
                    "source": process.source,
                    "owner_id": process.owner_id,
                    "owner_name": process.owner_name,
                    "pid": process.pid,
                },
            )
        for subagent in running_subagent_snapshots(events):
            subagents_removed += 1
            self.home.append_session_event(
                session_path,
                SUBAGENT_EVENT_TYPE,
                {
                    "agent_id": subagent.agent_id,
                    "kind": subagent.kind,
                    "name": subagent.name,
                    "status": "interrupted",
                    "statement": "Subagent was interrupted because the runtime restarted.",
                    "prompt": subagent.prompt,
                    "response": subagent.response,
                    "error": subagent.error,
                    "session_path": subagent.session_path,
                    "started_at": subagent.started_at,
                    "finished_at": utc_now_iso(),
                    "context_tokens": subagent.context_tokens,
                    "context_percent": subagent.context_percent,
                },
            )
        return RuntimeCleanupResult(
            processes_ended=processes_ended,
            subagents_removed=subagents_removed,
            workers_removed=subagents_removed,
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
            if self.home.full_session_logs_enabled(config):
                self._ensure_debug_meta(session_path, config)
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
    ) -> str:
        """Generate a response."""

        if self.agent_spec.kind == AgentKind.BUILD:
            session_path = parent_session_path or debug_session_path
            if session_path is None:
                session = self.home.create_session(
                    self.cwd,
                    provider=str(self.home.load_config().get("provider", "")),
                    model=str(self.home.load_config().get("model", "")),
                    mode=self.tool_manager.mode,
                )
                session_path = session.path
            self.home.append_session_event(session_path, "user_message", {"message": prompt})
        else:
            session_path = self.home.append_subagent_session_prompt(
                parent_session_path=parent_session_path or debug_session_path,
                subagent_name=self.process_owner_name or self.agent_spec.kind.value,
                subagent_id=self.process_owner_id or uuid4().hex[:8],
                subagent_kind=self.agent_spec.kind.value,
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
                final_text = response.text
                if active_callbacks.finish is not None:
                    active_callbacks.finish(final_text)
                return final_text

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
                final_text = text
                if active_callbacks.finish is not None:
                    active_callbacks.finish(final_text)
                return final_text

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
            stream_callbacks = RuntimeCallbacks(
                status=active_callbacks.status,
                delta=active_callbacks.delta,
            )
            response = self._stream_ollama_response(model, messages, stream_callbacks)
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
                final_text = text
                if active_callbacks.finish is not None:
                    active_callbacks.finish(final_text)
                return final_text

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

    def suggest_project_name(self, project_path: Path, directory_outline: str) -> str | None:
        """Suggest a compact project name from a folder path and directory outline."""

        prompt = self._project_name_prompt(project_path, directory_outline)
        config = self.home.load_config()
        provider = str(config.get("provider", ""))
        model = str(config.get("model", ""))
        if provider == "openai":
            name = self._suggest_openai_project_name(prompt, model)
            if name:
                return name
        elif provider == "anthropic":
            name = self._suggest_anthropic_project_name(prompt, model)
            if name:
                return name
        elif provider == "desy":
            name = self._suggest_desy_project_name(prompt, model)
            if name:
                return name
        elif provider == "ollama":
            name = self._suggest_ollama_project_name(prompt, model)
            if name:
                return name
        return None

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
            self._debug_log_step("openai", payload)
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
            self._debug_log_step(provider_key, payload)
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
        max_attempts = MODEL_REQUEST_RETRY_COUNT + 1
        for attempt in range(max_attempts):
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
                    or attempt >= MODEL_REQUEST_RETRY_COUNT
                ):
                    return message
                delay = self._model_request_retry_delay(attempt)
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    f"HTTP {error.code}",
                    delay,
                    attempt + 1,
                    MODEL_REQUEST_RETRY_COUNT,
                    status_callback,
                ):
                    return ""
            except (OSError, urllib.error.URLError, TimeoutError) as error:
                message = f"{provider_label} request failed: {error}"
                if attempt >= MODEL_REQUEST_RETRY_COUNT:
                    return message
                delay = self._model_request_retry_delay(attempt)
                if not self._sleep_before_model_request_retry(
                    provider_label,
                    str(error),
                    delay,
                    attempt + 1,
                    MODEL_REQUEST_RETRY_COUNT,
                    status_callback,
                ):
                    return ""
        return f"{provider_label} request failed."

    @staticmethod
    def _model_request_retry_delay(attempt: int) -> float:
        return min(
            MODEL_REQUEST_RETRY_INITIAL_DELAY_SECONDS
            * (MODEL_REQUEST_RETRY_BACKOFF_FACTOR**attempt),
            MODEL_REQUEST_RETRY_MAX_DELAY_SECONDS,
        )

    def _sleep_before_model_request_retry(
        self,
        provider_label: str,
        failure: str,
        delay_seconds: float,
        retry_number: int,
        retry_count: int,
        status_callback: StatusCallback | None,
    ) -> bool:
        self._status(status_callback, "Reconnecting")
        deadline = time.monotonic() + delay_seconds
        while True:
            if self._turn_aborted():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(MODEL_REQUEST_RETRY_SLEEP_SLICE_SECONDS, remaining))

    def _ensure_debug_meta(
        self,
        session_path: Path,
        config: Mapping[str, Any],
    ) -> None:
        """Create or update the debug session *meta.json*."""
        is_subagent = self.agent_spec.kind != AgentKind.BUILD
        session_id = (
            self._parent_session_id
            if is_subagent and self._parent_session_id
            else session_id_from_path(session_path)
        )
        project = self.home.project_for_path(self.cwd)
        meta: dict[str, Any] = {
            "session_id": session_id,
            "ai_model": str(config.get("model", "")),
            "provider": str(config.get("provider", "")),
            "mode": AgentMode.parse(config.get("agent_mode")).value,
            "location": str(self.cwd),
        }
        if project is not None:
            meta["name"] = project.name
        self.home.debug_logger.ensure_session(session_id, meta)

    def _debug_log_step(
        self,
        provider: str,
        payload: Mapping[str, Any],
        *,
        subagent_id: str | None = None,
    ) -> Path | None:
        """Write a debug step file when full session logs are enabled."""
        if not self.home.full_session_logs_enabled():
            return None
        session_path = self._debug_session_path
        if session_path is None:
            return None
        from anomx.agent.helpers.debug import session_id_from_path
        config = self.home.load_config()
        is_subagent = self.agent_spec.kind != AgentKind.BUILD
        session_id = (
            self._parent_session_id
            if is_subagent and self._parent_session_id
            else session_id_from_path(session_path)
        )
        messages = self.home.debug_logger.normalize_payload_messages(payload)
        model = str(config.get("model", ""))
        agent_subagent_id = (
            subagent_id
            or (self.process_owner_id if is_subagent else None)
        )
        return self.home.debug_logger.write_step(
            session_id,
            messages,
            model=model,
            provider=provider,
            subagent_id=agent_subagent_id,
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

        def stream_once() -> OllamaStreamResponse:
            self._debug_log_step("ollama", payload)
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            tool_calls: list[OllamaToolCall] = []
            with urllib.request.urlopen(request, timeout=120) as response:
                self._status(callbacks.status, "Thinking")
                for raw_line in response:
                    if self._turn_aborted():
                        return OllamaStreamResponse("", "", (), {"role": "assistant"})
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

        if self._turn_aborted():
            return ""
        response = self._model_request_with_retries(
            provider_key="ollama",
            provider_label="Ollama",
            env_var="",
            status_callback=callbacks.status,
            stream_once=stream_once,
        )
        if self._turn_aborted():
            return ""
        return cast(OllamaStreamResponse | str, response)

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

        if name in {"web_fetch", "webfetch"}:
            self._emit_operator_tool_statement(name, arguments, callbacks)
            return self._web_fetch_tool(arguments)
        if name in {"web_search", "websearch"}:
            self._emit_operator_tool_statement(name, arguments, callbacks)
            return self._web_search_tool(arguments)
        if name in {"read", "list", "glob", "grep"}:
            self._emit_operator_tool_statement(name, arguments, callbacks)
            return self._read_only_file_tool(name, arguments)

        if name in {"run_command", "run_cli_command", "bash"}:
            command = str(arguments.get("command", "")).strip()
            statement = str(arguments.get("statement", "")).strip()
            long_running_command: AsyncProcessState | None = None
            readonly_denial = self._readonly_command_denial(command, statement)
            if readonly_denial is not None:
                return readonly_denial

            def publish_long_running_command(process: subprocess.Popen[str]) -> str | None:
                nonlocal long_running_command
                if (
                    session_path is None
                    or long_running_command is not None
                ):
                    return None
                process_id = uuid4().hex[:8]
                source = "command"
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
                if callbacks.status is not None:
                    callbacks.status(f"Waiting:{60.0}")
                return f"Command {process_id} is still running."

            if self.cancel_event.is_set():
                return self._json_tool_result(
                    {
                        "approved": False,
                        "output": "Command was interrupted.",
                    }
                )

            if self._sandbox_session is not None and self._sandbox_session.is_running:
                # Sandbox mode: run via exec, skip tool_manager execution.
                # Authorize through tool_manager for approval flow.
                authorization = self.tool_manager._authorize_command(
                    command, statement, callbacks.approval
                )
                if isinstance(authorization, CommandResult):
                    result = authorization
                else:
                    policy = authorization
                    sandbox_output = self._sandbox_session.exec_command(command)
                    result = CommandResult(
                        sandbox_output,
                        approved=True,
                        safety=policy.safety,
                        command=policy.canonical_command,
                        reason=policy.reason,
                    )
                tool_payload = {
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
                if result.approved:
                    statement_text = statement or self._default_operator_tool_statement(name)
                    if callbacks.command is not None:
                        callbacks.command(statement_text, command, command_history_output)
                    elif callbacks.tool_message is not None:
                        callbacks.tool_message(statement_text)
                self._emit_command_system_message(callbacks, result, statement)
                return self._json_tool_result(tool_payload)

            result = self.tool_manager.run_command(
                command,
                statement or "Operator command",
                callbacks.approval,
                long_running_callback=publish_long_running_command,
            )
            tool_payload = {
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
            if result.approved:
                statement_text = statement or self._default_operator_tool_statement(name)
                if callbacks.command is not None:
                    callbacks.command(statement_text, command, command_history_output)
                elif callbacks.tool_message is not None:
                    callbacks.tool_message(statement_text)
            self._emit_command_system_message(callbacks, result, statement)
            return self._json_tool_result(tool_payload)

        if name in {
            "create_plan",
            "update_plan",
            "start_process",
            "end_process",
            "check_command_status",
            "kill_command",
            "ask_question",
            "start_subagent",
            "prompt_subagent",
            "remove_subagent",
            "start_agent",
            "prompt_agent",
            "remove_agent",
            "interrupt_agent",
        }:
            self._emit_operator_tool_statement(name, arguments, callbacks)

        if name in {"start_subagent", "start_agent"}:
            return self._start_subagent_tool(arguments, session_path, callbacks)
        if name in {"prompt_subagent", "prompt_agent"}:
            return self._prompt_subagent_tool(arguments, session_path, callbacks)
        if name in {"remove_subagent", "remove_agent", "interrupt_agent"}:
            return self._remove_subagent_tool(arguments, session_path)
        if name in {"get_subagent_info", "check_agent"}:
            return self._get_subagent_info_tool(arguments, session_path)
        if name == "create_plan":
            if not self.agent_spec.can_use_plans:
                return self._json_tool_result({"error": "This agent kind cannot create plans."})
            return self._create_plan_tool(arguments, session_path)
        if name == "update_plan":
            if not self.agent_spec.can_use_plans:
                return self._json_tool_result({"error": "This agent kind cannot update plans."})
            return self._update_plan_tool(arguments, session_path)




        if name == "start_process":
            if not self.agent_spec.can_start_processes:
                return self._json_tool_result(
                    {"error": "This agent kind cannot start async processes."}
                )
            return self._start_process_tool(arguments, session_path, callbacks)
        if name == "end_process":
            return self._end_process_tool(arguments, session_path)
        if name == "check_command_status":
            return self._check_command_status_tool(arguments)
        if name == "kill_command":
            return self._kill_command_tool(arguments, session_path, callbacks)
        if name == "ask_question":
            if not self.agent_spec.can_ask_questions:
                return self._json_tool_result(
                    {"error": "This agent kind cannot ask the user questions."}
                )
            return self._ask_question_tool(arguments, callbacks)


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
        if self._running_subagent_states():
            wait_output = self._wait_tool({}, callbacks)
            return self._subagent_continuation_prompt(wait_output), False
        if self._running_command_states():
            wait_output = self._wait_tool({}, callbacks)
            return self._command_continuation_prompt(wait_output), False

        if self.role != AgentRole.OPERATOR or not False:
            return self._plan_finish_continuation_prompt(
                session_path,
                plan_finish_attempts,
                callbacks,
            )
        wait_output = self._wait_tool({}, callbacks)
        return (None or wait_output), False

    def _plan_finish_continuation_prompt(
        self,
        session_path: Path | None,
        plan_finish_attempts: int,
        callbacks: RuntimeCallbacks | None = None,
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

        validation_message = "Validating whether the plan is finished"
        if callbacks is not None and callbacks.tool_message is not None:
            callbacks.tool_message(validation_message)
        elif callbacks is not None and callbacks.status is not None:
            callbacks.status(validation_message)
        else:
            self.home.append_session_event(
                session_path,
                "work_message",
                {
                    "message": validation_message,
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

    def _command_continuation_prompt(self, wait_output: str) -> str:
        return (
            "A long-running command tool call is still active or just finished. Continue "
            "from the command wait result below. Do not produce a final answer while a "
            "required command is still running; use wait, check_command_status, or "
            "kill_command as appropriate.\n"
            f"{wait_output}"
        )

    def _subagent_continuation_prompt(self, wait_output: str) -> str:
        return (
            "One or more subagents were active while you were about to finish. Continue "
            "from the subagent wait result below. Integrate ready subagent outputs, prompt "
            "or remove subagents if more work is needed, and do not produce a final answer "
            "while required subagent work is still running.\n"
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

        payload = {"steps": serialize_plan_steps(steps), "action": "create"}
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

        payload = {"steps": serialize_plan_steps(steps), "action": "update"}
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

    def _start_process_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "start_process requires a session."})

        if self._sandbox_session is not None:
            return self._json_tool_result(
                {
                    "approved": False,
                    "started": False,
                    "output": "start_process is not supported in sandbox mode. "
                    "Use run_command instead.",
                }
            )

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
            owner_id=self.process_owner_id,
            owner_name=self.process_owner_name,
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

    def _start_subagent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if not self.agent_spec.can_spawn_subagents:
            return self._json_tool_result({"error": "Only the build agent can start subagents."})
        if session_path is None:
            return self._json_tool_result({"error": "start_subagent requires a session."})

        kind_text = str(
            arguments.get("agent_kind") or arguments.get("kind") or "general"
        ).strip().lower()
        try:
            kind = AgentKind(kind_text)
        except ValueError:
            return self._json_tool_result(
                {
                    "error": "agent_kind must be one of: general, explore, scout.",
                    "allowed_agent_kinds": ["general", "explore", "scout"],
                }
            )
        if kind == AgentKind.BUILD:
            return self._json_tool_result({"error": "build cannot be launched as a subagent."})

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return self._json_tool_result({"error": "start_subagent requires a prompt."})

        with self._subagent_lock:
            visible_agents = [
                agent
                for agent in self._subagents.values()
                if agent.status != "removed"
            ]
            if len(visible_agents) >= SUBAGENT_MAX_CONCURRENT:
                return self._json_tool_result(
                    {
                        "error": "At most five subagents can be active at the same time.",
                        "limit": SUBAGENT_MAX_CONCURRENT,
                    }
                )

        agent_id = uuid4().hex[:8]
        name = self._subagent_display_name(
            str(arguments.get("name", "")).strip(),
            kind,
            agent_id,
        )
        statement = str(arguments.get("statement", "")).strip() or f"Starting {name}"
        state = SubagentRuntimeState(
            agent_id=agent_id,
            kind=kind,
            name=name,
            prompt=prompt,
            status="running",
            statement=statement,
            started_at=utc_now_iso(),
        )
        child_runtime = AgentRuntime(
            self.home,
            self.cwd,
            self.session_allowed_commands,
            self.session_rejected_commands,
            self.tool_manager.mode,
            role=kind.value,
            cancel_event=state.cancel_event,
            workspace_root=self.workspace_root,
            process_owner_id=agent_id,
            process_owner_name=name,
        )
        state.runtime = child_runtime
        if session_path is not None:
            child_runtime._parent_session_id = session_id_from_path(session_path)
        with self._subagent_lock:
            self._subagents[agent_id] = state

        self._publish_subagent_state(state, session_path, message=statement)
        self._start_subagent_worker(state, prompt, session_path, callbacks)
        return self._json_tool_result(
            {
                "started": True,
                "agent_id": agent_id,
                "agent_kind": kind.value,
                "name": name,
                "status": state.status,
            }
        )

    def _prompt_subagent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
        callbacks: RuntimeCallbacks,
    ) -> str:
        if not self.agent_spec.can_spawn_subagents:
            return self._json_tool_result({"error": "Only the build agent can prompt subagents."})
        if session_path is None:
            return self._json_tool_result({"error": "prompt_subagent requires a session."})
        agent_id = str(arguments.get("agent_id", "")).strip()
        prompt = str(arguments.get("prompt", "")).strip()
        if not agent_id:
            return self._json_tool_result({"error": "prompt_subagent requires an agent_id."})
        if not prompt:
            return self._json_tool_result({"error": "prompt_subagent requires a prompt."})
        with self._subagent_lock:
            state = self._subagents.get(agent_id)
            if state is None or state.status == "removed":
                return self._json_tool_result({"error": "Unknown subagent id."})
            if state.status in {"running", "working"}:
                return self._json_tool_result({"error": "Subagent is already running."})
            state.prompt = prompt
            state.status = "running"
            state.statement = (
                str(arguments.get("statement", "")).strip() or f"Prompting {state.name}"
            )
            state.response = ""
            state.error = ""
            state.finished_at = ""
            state.cancel_event.clear()
            if state.runtime is None:
                state.runtime = AgentRuntime(
                    self.home,
                    self.cwd,
                    self.session_allowed_commands,
                    self.session_rejected_commands,
                    self.tool_manager.mode,
                    role=state.kind.value,
                    cancel_event=state.cancel_event,
                    workspace_root=self.workspace_root,
                    process_owner_id=state.agent_id,
                    process_owner_name=state.name,
                )
                if session_path is not None:
                    state.runtime._parent_session_id = session_id_from_path(session_path)
        self._publish_subagent_state(state, session_path, message=state.statement)
        self._start_subagent_worker(state, prompt, session_path, callbacks)
        return self._json_tool_result(
            {
                "prompted": True,
                "agent_id": state.agent_id,
                "name": state.name,
                "status": state.status,
            }
        )

    def _remove_subagent_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        if not self.agent_spec.can_spawn_subagents:
            return self._json_tool_result({"error": "Only the build agent can remove subagents."})
        agent_id = str(arguments.get("agent_id", "")).strip()
        if not agent_id:
            return self._json_tool_result({"error": "remove_subagent requires an agent_id."})
        with self._subagent_lock:
            state = self._subagents.get(agent_id)
            if state is None:
                return self._json_tool_result({"removed": False, "error": "Unknown subagent id."})
            state.status = "removed"
            state.statement = str(arguments.get("statement", "")).strip() or "Removed subagent"
            state.finished_at = utc_now_iso()
            state.cancel_event.set()
            runtime = state.runtime
        if runtime is not None:
            with suppress(Exception):
                runtime.abort_current_turn(state.session_path)
            with suppress(Exception):
                runtime.shutdown(state.session_path)
        self._publish_subagent_state(state, session_path, message=state.statement)
        with self._subagent_lock:
            self._subagents.pop(agent_id, None)
        return self._json_tool_result({"removed": True, "agent_id": agent_id})

    def _get_subagent_info_tool(
        self,
        arguments: dict[str, Any],
        session_path: Path | None,
    ) -> str:
        if session_path is None:
            return self._json_tool_result({"error": "get_subagent_info requires a session."})
        agent_id = str(arguments.get("agent_id", "")).strip()
        if not agent_id:
            return self._json_tool_result({"error": "get_subagent_info requires an agent_id."})
        snapshot = self._subagent_snapshot(agent_id, session_path)
        if snapshot is None:
            return self._json_tool_result({"error": "Unknown subagent id."})
        return self._json_tool_result(self._subagent_snapshot_payload(snapshot))

    def _subagent_snapshot(
        self,
        agent_id: str,
        session_path: Path,
    ) -> SubagentSnapshot | None:
        for snapshot in subagent_snapshots(
            self.home.read_session_events(session_path),
            include_removed=True,
        ):
            if snapshot.agent_id == agent_id:
                return snapshot
        return None

    def _subagent_snapshot_payload(self, snapshot: SubagentSnapshot) -> dict[str, object]:
        return {
            "agent_id": snapshot.agent_id,
            "name": snapshot.name,
            "agent_kind": snapshot.kind,
            "status": snapshot.status,
            "statement": snapshot.statement,
            "prompt": snapshot.prompt,
            "response": snapshot.response,
            "error": snapshot.error,
            "session_path": snapshot.session_path,
            "context_tokens": snapshot.context_tokens,
            "context_percent": snapshot.context_percent,
            "latest_outputs": [
                {
                    "timestamp": entry.timestamp,
                    "kind": entry.kind,
                    "text": entry.text,
                }
                for entry in snapshot.history[-5:]
            ],
            "commands": list(snapshot.command_history),
        }

    def _subagent_display_name(
        self,
        requested_name: str,
        kind: AgentKind,
        agent_id: str,
    ) -> str:
        cleaned = " ".join(requested_name.split())
        if cleaned:
            return cleaned[:40]
        return f"{kind.value.title()} {agent_id[:4]}"

    def _start_subagent_worker(
        self,
        state: SubagentRuntimeState,
        prompt: str,
        parent_session_path: Path,
        parent_callbacks: RuntimeCallbacks,
    ) -> None:
        def run_subagent() -> None:
            self._run_subagent_turn(state, prompt, parent_session_path, parent_callbacks)

        worker = threading.Thread(target=run_subagent, daemon=True)
        state.worker = worker
        worker.start()

    def _run_subagent_turn(
        self,
        state: SubagentRuntimeState,
        prompt: str,
        parent_session_path: Path,
        parent_callbacks: RuntimeCallbacks,
    ) -> None:
        runtime = state.runtime
        if runtime is None:
            return

        def publish(
            statement: str = "",
            *,
            status: str = "working",
            message: str = "",
            command: dict[str, str] | None = None,
        ) -> None:
            text = statement.strip() or message.strip()
            with self._subagent_lock:
                if state.status == "removed":
                    return
                state.status = status
                if text:
                    state.statement = text
                self._refresh_subagent_context(state)
                if command is not None:
                    state.command_history.append(command)
            self._publish_subagent_state(
                state,
                parent_session_path,
                message=message or text,
                command=command,
            )

        def status_callback(message: str) -> None:
            normalized = message.strip()
            if not normalized or normalized in {"Thinking", "Loading model"}:
                return
            publish(normalized, status="working")

        def message_callback(message: str) -> None:
            self._append_subagent_agent_message(state, message, intermediate=True)
            publish(message, status="working", message=message)

        def tool_message_callback(message: str) -> None:
            self._append_subagent_work_message(state, message, role="tool")
            publish(message, status="working", message=message)

        def command_callback(statement: str, command: str, output: str) -> None:
            command_payload = {
                "statement": statement.strip(),
                "command": command.strip(),
                "output": output.strip(),
            }
            self._append_subagent_work_message(
                state,
                statement,
                role="tool",
                command=command,
            )
            publish(statement, status="working", command=command_payload)

        def delta_callback(_delta: str) -> None:
            return

        def system_message_callback(role: str, message: str) -> None:
            self._append_subagent_system_message(state, role, message)
            publish(message, status="working", message=message)

        def approval_callback(request: CommandApprovalRequest) -> ApprovalChoice:
            if parent_callbacks.approval is None:
                return ApprovalChoice.REJECT
            labelled = replace(
                request,
                agent_id=state.agent_id,
                agent_name=state.name,
            )
            return parent_callbacks.approval(labelled)

        try:
            session_path = self.home.append_subagent_session_prompt(
                parent_session_path=parent_session_path,
                subagent_name=state.name,
                subagent_id=state.agent_id,
                subagent_kind=state.kind.value,
                prompt=prompt,
            )
            state.session_path = session_path
            self._publish_subagent_state(state, parent_session_path)
            response = runtime.backend_response(
                session_path,
                callbacks=RuntimeCallbacks(
                    status=status_callback,
                    message=message_callback,
                    tool_message=tool_message_callback,
                    command=command_callback,
                    delta=delta_callback,
                    approval=approval_callback,
                    system_message=system_message_callback,
                    process=parent_callbacks.process,
                ),
                debug_session_path=session_path,
            )
            if state.cancel_event.is_set() or state.status == "removed":
                return
            state.response = response.strip()
            state.status = "ready"
            state.statement = ""
            state.finished_at = utc_now_iso()
            self._refresh_subagent_context(state)
            if state.response:
                self.home.append_session_event(
                    session_path,
                    "agent_message",
                    {"message": state.response},
                )
            self._publish_subagent_state(
                state,
                parent_session_path,
                message=state.response,
            )
        except Exception as error:  # pragma: no cover - defensive thread boundary
            state.error = f"{type(error).__name__}: {error}"
            state.status = "failed"
            state.statement = state.error
            state.finished_at = utc_now_iso()
            self._publish_subagent_state(
                state,
                parent_session_path,
                message=state.error,
            )

    def _append_subagent_agent_message(
        self,
        state: SubagentRuntimeState,
        message: str,
        *,
        intermediate: bool = False,
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        payload: dict[str, object] = {"message": message.strip()}
        if intermediate:
            payload["intermediate"] = True
        self.home.append_session_event(state.session_path, "agent_message", payload)

    def _append_subagent_work_message(
        self,
        state: SubagentRuntimeState,
        message: str,
        *,
        role: str,
        command: str = "",
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        payload: dict[str, object] = {"message": message.strip(), "role": role}
        if command:
            payload["command"] = command
        self.home.append_session_event(state.session_path, "work_message", payload)

    def _append_subagent_system_message(
        self,
        state: SubagentRuntimeState,
        role: str,
        message: str,
    ) -> None:
        if state.session_path is None or not message.strip():
            return
        self.home.append_session_event(
            state.session_path,
            "system_message",
            {"message": message.strip(), "role": role or "system"},
        )

    def _refresh_subagent_context(self, state: SubagentRuntimeState) -> None:
        if state.runtime is None or state.session_path is None:
            return
        with suppress(Exception):
            state.context_tokens = state.runtime.estimate_session_context_tokens(state.session_path)
        context_window = None
        with suppress(Exception):
            config = self.home.load_config()
            context_window = model_context_window(str(config.get("model", "")))
        state.context_percent = context_usage_percent(state.context_tokens, context_window)

    def _publish_subagent_state(
        self,
        state: SubagentRuntimeState,
        session_path: Path | None,
        *,
        message: str = "",
        command: dict[str, str] | None = None,
    ) -> None:
        if session_path is None:
            return
        payload: dict[str, object] = {
            "agent_id": state.agent_id,
            "kind": state.kind.value,
            "name": state.name,
            "status": state.status,
            "statement": state.statement,
            "prompt": state.prompt,
            "response": state.response,
            "error": state.error,
            "session_path": str(state.session_path) if state.session_path is not None else "",
            "started_at": state.started_at,
            "updated_at": utc_now_iso(),
            "finished_at": state.finished_at,
            "context_tokens": state.context_tokens,
            "context_percent": state.context_percent,
        }
        if message:
            payload["message"] = message.strip()
        if command is not None:
            payload["command"] = command
        if state.command_history:
            payload["commands"] = list(state.command_history)
        self.home.append_session_event(session_path, SUBAGENT_EVENT_TYPE, payload)

    def _running_subagent_states(self) -> tuple[SubagentRuntimeState, ...]:
        with self._subagent_lock:
            return tuple(
                state
                for state in self._subagents.values()
                if state.status in {"running", "working"}
                and state.worker is not None
                and state.worker.is_alive()
            )

    def _subagent_states(self) -> tuple[SubagentRuntimeState, ...]:
        with self._subagent_lock:
            return tuple(state for state in self._subagents.values() if state.status != "removed")

    def _end_all_subagent_states(self, session_path: Path | None = None) -> int:
        with self._subagent_lock:
            states = tuple(
                state
                for state in self._subagents.values()
                if state.status in {"running", "working"}
            )
        ended = 0
        for state in states:
            state.cancel_event.set()
            if state.runtime is not None:
                with suppress(Exception):
                    state.runtime.abort_current_turn(state.session_path)
                with suppress(Exception):
                    state.runtime.shutdown(state.session_path)
            state.status = "interrupted"
            state.statement = "Subagent was interrupted because Anomx was interrupted."
            state.finished_at = utc_now_iso()
            self._publish_subagent_state(state, session_path, message=state.statement)
            ended += 1
        return ended

    def _readonly_command_denial(self, command: str, statement: str) -> str | None:
        if not self.agent_spec.read_only:
            return None
        policy = self.tool_manager.classify(command, include_session_allowances=False)
        if policy.safety == CommandSafety.ALLOW:
            return None
        return self._json_tool_result(
            {
                "approved": False,
                "output": (
                    "This subagent is read-only. The command was denied because it is "
                    f"not classified as a read-only exploration command. Reason: {policy.reason}"
                ),
                "safety": CommandSafety.FORBIDDEN.value,
                "command": policy.canonical_command,
                "statement": statement,
            }
        )

    def _read_only_file_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "read":
            return self._read_file_tool(arguments)
        if name == "list":
            return self._list_path_tool(arguments)
        if name == "glob":
            return self._glob_tool(arguments)
        if name == "grep":
            return self._grep_tool(arguments)
        return self._json_tool_result({"error": f"Unknown read-only tool: {name}"})

    def _read_file_tool(self, arguments: dict[str, Any]) -> str:
        path_or_error = self._workspace_file_path(arguments.get("path"))
        if isinstance(path_or_error, str):
            return self._json_tool_result({"error": path_or_error})
        path = path_or_error
        if not path.is_file():
            return self._json_tool_result({"error": "Path is not a file."})
        start_line = self._positive_int(arguments.get("start_line"), 1)
        max_lines = min(self._positive_int(arguments.get("max_lines"), 200), 1_000)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            return self._json_tool_result({"error": str(error)})
        start_index = max(0, start_line - 1)
        selected = lines[start_index : start_index + max_lines]
        return self._json_tool_result(
            {
                "path": str(path),
                "start_line": start_line,
                "line_count": len(selected),
                "total_lines": len(lines),
                "content": "\n".join(selected),
            }
        )

    def _list_path_tool(self, arguments: dict[str, Any]) -> str:
        path_or_error = self._workspace_path(arguments.get("path") or ".")
        if isinstance(path_or_error, str):
            return self._json_tool_result({"error": path_or_error})
        path = path_or_error
        if not path.is_dir():
            return self._json_tool_result({"error": "Path is not a directory."})
        try:
            entries = sorted(
                path.iterdir(),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )
        except OSError as error:
            return self._json_tool_result({"error": str(error)})
        limit = min(self._positive_int(arguments.get("limit"), 200), 1_000)
        return self._json_tool_result(
            {
                "path": str(path),
                "entries": [
                    {
                        "name": entry.name,
                        "path": str(entry),
                        "kind": "directory" if entry.is_dir() else "file",
                    }
                    for entry in entries[:limit]
                ],
                "truncated": len(entries) > limit,
            }
        )

    def _glob_tool(self, arguments: dict[str, Any]) -> str:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return self._json_tool_result({"error": "glob requires a pattern."})
        root_or_error = self._workspace_path(arguments.get("path") or ".")
        if isinstance(root_or_error, str):
            return self._json_tool_result({"error": root_or_error})
        root = root_or_error
        if not root.is_dir():
            return self._json_tool_result({"error": "Glob root is not a directory."})
        limit = min(self._positive_int(arguments.get("limit"), 200), 1_000)
        matches: list[str] = []
        try:
            for match in root.glob(pattern):
                resolved = match.resolve()
                if self._path_inside_workspace(resolved):
                    matches.append(str(resolved))
                if len(matches) >= limit:
                    break
        except (OSError, ValueError) as error:
            return self._json_tool_result({"error": str(error)})
        return self._json_tool_result({"matches": matches, "truncated": len(matches) >= limit})

    def _grep_tool(self, arguments: dict[str, Any]) -> str:
        pattern = str(arguments.get("pattern", "")).strip()
        if not pattern:
            return self._json_tool_result({"error": "grep requires a pattern."})
        path_or_error = self._workspace_path(arguments.get("path") or ".")
        if isinstance(path_or_error, str):
            return self._json_tool_result({"error": path_or_error})
        root = path_or_error
        include = str(arguments.get("include", "*")).strip() or "*"
        limit = min(self._positive_int(arguments.get("limit"), 100), 1_000)
        with suppress(re.error):
            regex = re.compile(pattern)
            return self._json_tool_result(
                {
                    "matches": self._grep_regex_matches(root, regex, include, limit),
                    "pattern": pattern,
                }
            )
        needle = pattern.lower()
        return self._json_tool_result(
            {
                "matches": self._grep_literal_matches(root, needle, include, limit),
                "pattern": pattern,
            }
        )

    def _grep_regex_matches(
        self,
        root: Path,
        regex: re.Pattern[str],
        include: str,
        limit: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for path in self._iter_grep_files(root, include):
            for line_number, line in self._iter_file_lines(path):
                if regex.search(line):
                    matches.append(
                        {"path": str(path), "line": line_number, "text": line.rstrip()}
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def _grep_literal_matches(
        self,
        root: Path,
        needle: str,
        include: str,
        limit: int,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for path in self._iter_grep_files(root, include):
            for line_number, line in self._iter_file_lines(path):
                if needle in line.lower():
                    matches.append(
                        {"path": str(path), "line": line_number, "text": line.rstrip()}
                    )
                    if len(matches) >= limit:
                        return matches
        return matches

    def _iter_grep_files(self, root: Path, include: str) -> Iterable[Path]:
        if root.is_file():
            yield root
            return
        for path in root.rglob(include):
            if path.is_file() and self._path_inside_workspace(path.resolve()):
                yield path

    def _iter_file_lines(self, path: Path) -> Iterable[tuple[int, str]]:
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                yield from enumerate(handle, start=1)
        except OSError:
            return

    def _web_fetch_tool(self, arguments: dict[str, Any]) -> str:
        url = str(arguments.get("url", "")).strip()
        if not url:
            return self._json_tool_result({"error": "web_fetch requires a url."})
        if not url.startswith(("http://", "https://")):
            return self._json_tool_result({"error": "Only http and https URLs are supported."})
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AnomxAgent/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read(WEB_FETCH_MAX_CHARS + 1)
                content_type = str(response.headers.get("content-type", ""))
                status = int(getattr(response, "status", 200))
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            return self._json_tool_result({"error": str(error), "url": url})
        text = raw[:WEB_FETCH_MAX_CHARS].decode("utf-8", errors="replace")
        return self._json_tool_result(
            {
                "url": url,
                "status": status,
                "content_type": content_type,
                "truncated": len(raw) > WEB_FETCH_MAX_CHARS,
                "content": self._plain_web_text(text),
            }
        )

    def _web_search_tool(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return self._json_tool_result({"error": "web_search requires a query."})
        limit = min(self._positive_int(arguments.get("limit"), WEB_SEARCH_MAX_RESULTS), 10)
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AnomxAgent/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                html_text = response.read(80_000).decode("utf-8", errors="replace")
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            return self._json_tool_result({"error": str(error), "query": query})
        return self._json_tool_result(
            {
                "query": query,
                "results": self._duckduckgo_results(html_text, limit),
            }
        )

    def _duckduckgo_results(self, html_text: str, limit: int) -> list[dict[str, str]]:
        results: list[dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(html_text):
            href = html.unescape(match.group("href"))
            title = self._plain_web_text(match.group("title"))
            url = self._duckduckgo_result_url(href)
            if not title or not url:
                continue
            results.append({"title": title, "url": url})
            if len(results) >= limit:
                break
        return results

    def _duckduckgo_result_url(self, href: str) -> str:
        parsed = urllib.parse.urlparse(href)
        query = urllib.parse.parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return urllib.parse.unquote(uddg[0])
        return href

    def _plain_web_text(self, text: str) -> str:
        without_scripts = re.sub(
            r"<(script|style)\b.*?</\1>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        without_tags = re.sub(r"<[^>]+>", " ", without_scripts)
        return " ".join(html.unescape(without_tags).split())

    def _workspace_file_path(self, raw_path: object) -> Path | str:
        path_or_error = self._workspace_path(raw_path)
        if isinstance(path_or_error, str):
            return path_or_error
        return path_or_error

    def _workspace_path(self, raw_path: object) -> Path | str:
        raw = str(raw_path or "").strip()
        if not raw:
            return "Path is required."
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.tool_manager.current_dir / candidate
        try:
            resolved = candidate.resolve()
        except OSError as error:
            return str(error)
        if not self._path_inside_workspace(resolved):
            return f"Path is outside the trusted workspace: {raw}"
        return resolved

    def _path_inside_workspace(self, path: Path) -> bool:
        root = self.workspace_root
        return path == root or root in path.parents

    def _positive_int(self, value: object, fallback: int) -> int:
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                return fallback
        else:
            return fallback
        return parsed if parsed > 0 else fallback

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
                args=(process_state, stream, callbacks),
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

        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.exit_code = return_code
            if current.status == "running":
                current.status = "ended"
                current.finished_at = utc_now_iso()
                self._publish_process_state(current, session_path, callbacks)

    def _read_process_stream(
        self,
        process_state: AsyncProcessState,
        stream: TextIO,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        try:
            while True:
                chunk = stream.readline()
                if chunk == "":
                    break
                self._append_process_output(process_state, chunk, callbacks)
        except OSError as error:  # pragma: no cover - defensive process boundary
            self._append_process_output(process_state, str(error), callbacks)

    def _append_process_output(
        self,
        process_state: AsyncProcessState,
        chunk: str,
        callbacks: RuntimeCallbacks | None = None,
    ) -> None:
        if not chunk:
            return
        payload: dict[str, object] | None = None
        target_path: Path | None = None
        callback_state: AsyncProcessState | None = None
        with self._process_lock:
            current = self._processes.get(process_state.process_id)
            if current is None:
                return
            current.output_chunks.append(chunk)
            current.output = self._compact_process_output("".join(current.output_chunks))
            now = time.monotonic()
            if (
                current.status == "running"
                and current.session_path is not None
                and current.output != current.last_output_event_text
                and now - current.last_output_event_at >= 0.25
            ):
                current.last_output_event_at = now
                current.last_output_event_text = current.output
                target_path = current.session_path
                payload = self._process_state_payload(current)
                callback_state = current
        if payload is not None and target_path is not None:
            self.home.append_session_event(target_path, "process_event", payload)
            if (
                callbacks is not None
                and callbacks.process is not None
                and callback_state is not None
            ):
                callbacks.process(callback_state)

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

    def _append_command_event_snapshot(self, process_state: AsyncProcessState) -> None:
        if process_state.session_path is None:
            return
        self.home.append_session_event(
            process_state.session_path,
            "process_event",
            self._command_state_payload(process_state),
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
        payload = self._command_state_payload(process_state)
        self._append_command_event_snapshot(process_state)
        return self._json_tool_result(payload)

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
        self._append_command_event_snapshot(current)
        payload["waited_seconds"] = waited_seconds
        return payload

    def _wait_tool(
        self,
        arguments: dict[str, Any],
        callbacks: RuntimeCallbacks | None = None,
    ) -> str:
        del arguments
        seconds = SUBAGENT_WAIT_SECONDS
        started_at = time.monotonic()
        if not self._has_running_wait_targets():
            return self._json_tool_result(
                {
                    "waited_seconds": 0.0,

                    "commands": [
                        self._command_state_payload(command)
                        for command in self._command_states()
                    ],
                    "subagents": [
                        self._subagent_runtime_payload(subagent)
                        for subagent in self._subagent_states()
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

                "commands": [
                    self._command_state_payload(command)
                    for command in self._command_states()
                ],
                "subagents": [
                    self._subagent_runtime_payload(subagent)
                    for subagent in self._subagent_states()
                ],
            }
        )

    def _has_running_wait_targets(self) -> bool:
        return bool(self._running_command_states() or self._running_subagent_states())

    def _subagent_runtime_payload(self, state: SubagentRuntimeState) -> dict[str, object]:
        return {
            "agent_id": state.agent_id,
            "name": state.name,
            "agent_kind": state.kind.value,
            "status": state.status,
            "statement": state.statement,
            "response": state.response,
            "error": state.error,
            "session_path": str(state.session_path) if state.session_path is not None else "",
            "context_tokens": state.context_tokens,
            "context_percent": state.context_percent,
            "commands": list(state.command_history),
        }

    def _bounded_wait_seconds(self, value: object) -> float:
        try:
            seconds = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            seconds = SUBAGENT_WAIT_SECONDS
        return max(0.25, min(seconds, SUBAGENT_WAIT_SECONDS))

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
            "start_process": "Starting process",
            "end_process": "Ending process",
            "check_command_status": "Checking command",
            "kill_command": "Killing command",
            "ask_question": "Asking question",
            "remove_plan": "Removing plan",
            "finish_anyways": "Finishing anyway",
            "start_subagent": "Starting subagent",
            "prompt_subagent": "Prompting subagent",
            "remove_subagent": "Removing subagent",
            "get_subagent_info": "Checking subagent",
            "start_agent": "Starting subagent",
            "prompt_agent": "Prompting subagent",
            "remove_agent": "Removing subagent",
            "interrupt_agent": "Removing subagent",
            "check_agent": "Checking subagent",
            "web_search": "Searching web",
            "web_fetch": "Fetching web page",
            "websearch": "Searching web",
            "webfetch": "Fetching web page",
            "read": "Reading file",
            "list": "Listing directory",
            "glob": "Finding files",
            "grep": "Searching files",
            "bash": "Running command",
        }.get(tool_name, "Working")

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

    def _suggest_openai_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key("openai", "OPENAI_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(
                {
                    "model": model,
                    "instructions": self._project_name_system_prompt(),
                    "input": [{"role": "user", "content": prompt}],
                    "max_output_tokens": 16,
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
        return self._sanitize_project_name(self.extract_openai_text(data))

    def _suggest_anthropic_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key("anthropic", "ANTHROPIC_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": model,
                    "system": self._project_name_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
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
        return self._sanitize_project_name(self.extract_anthropic_text(data))

    def _suggest_desy_project_name(self, prompt: str, model: str) -> str | None:
        api_key = self._api_key("desy", "DESY_ASSISTANT_API_KEY")
        if api_key is None:
            return None

        request = urllib.request.Request(
            DESY_MESSAGES_ENDPOINT,
            data=json.dumps(
                {
                    "model": model,
                    "system": self._project_name_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
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
        return self._sanitize_project_name(self.extract_anthropic_text(data))

    def _suggest_ollama_project_name(self, prompt: str, model: str) -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self._project_name_system_prompt()},
                {"role": "user", "content": prompt},
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
        return self._sanitize_project_name(str(message.get("content", "")))

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

    def _project_name_system_prompt(self) -> str:
        return (
            "Please only return a plain text name of 2-3 words for this directory "
            "in a project style. No quotes. No trailing punctuation."
        )

    def _project_name_prompt(self, project_path: Path, directory_outline: str) -> str:
        outline = directory_outline.strip() or "- empty directory"
        return (
            f"Directory path:\n{project_path}\n\n"
            "Directory structure, first 3 levels:\n"
            f"{outline}"
        )

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

    def _sanitize_project_name(self, name: str) -> str | None:
        cleaned = " ".join(name.strip().strip("\"'`").split())
        cleaned = cleaned.rstrip(".:;,-")
        if not cleaned:
            return None
        words = [
            word.strip(" .,:;!?()[]{}\"'`")
            for word in cleaned.replace("_", " ").replace("/", " ").split()
        ]
        words = [word for word in words if word]
        if len(words) < 2:
            return None
        return " ".join(words[:3])[:48] or None

    def _instructions(self, session_path: Path | None = None) -> str:
        tools = "\n".join(f"- {tool}" for tool in self._operator_tool_descriptions())
        runtime_context = self._operator_runtime_context(session_path)
        return "\n\n".join(
            [
                self.agent_spec.prompt,
                *self._instruction_environment_sections(),
                runtime_context,
                f"Available tools:\n{tools}",
            ]
        )

    def _instruction_environment_sections(self) -> list[str]:
        sections = [self.tool_manager.mode.system_prompt_statement]
        user_name = str(self.home.load_config().get("user_name") or "").strip()
        if user_name:
            sections.append(f"User profile:\n- Name: {user_name}")
        sections.append("\n".join(self.tool_manager.workspace_prompt_lines()))
        session_policy = self.tool_manager.session_policy_prompt_lines()
        if session_policy:
            sections.append("\n".join(session_policy))
        sandbox_section = self._sandbox_instruction_section()
        if sandbox_section:
            sections.append(sandbox_section)
        custom_section = self._custom_instructions_section()
        if custom_section:
            sections.append(custom_section)
        return sections

    def _sandbox_instruction_section(self) -> str | None:
        if self._sandbox_session is None:
            return None
        cfg = self._sandbox_session.config
        return cfg.sandbox_context_prompt()

    def _custom_instructions_section(self) -> str | None:
        """Read custom instruction files and return a formatted section, or None."""
        instructions_dir = self.home.instructions_dir
        if not instructions_dir.is_dir():
            return None
        instruction_path = instructions_dir / "instruction.md"
        if not instruction_path.exists():
            return None
        content = instruction_path.read_text(encoding="utf-8").strip()
        if not content:
            return None
        return "## Custom Instructions\n\n" + content

    def _operator_tool_descriptions(self) -> tuple[str, ...]:
        if self.agent_spec.kind != AgentKind.BUILD:
            descriptions = []
            if not self.agent_spec.read_only:
                descriptions.append(
                    "run_command(statement, command): run a CLI command."
                )
            if self.agent_spec.read_only:
                descriptions.extend(
                    [
                        "read(statement, path, start_line, max_lines): read a file.",
                        "list(statement, path, limit): list a directory.",
                        "glob(statement, pattern, path, limit): find files by glob pattern.",
                        "grep(statement, pattern, path, include, limit): search text in files.",
                        (
                            "bash(statement, command): run a read-only shell command. "
                            "Write-capable commands are denied."
                        ),
                    ]
                )
            if self.agent_spec.can_start_processes:
                descriptions.extend(
                    [
                        (
                            "start_process(statement, command): start a long-running "
                            "async CLI process."
                        ),
                        "end_process(statement, process_id): end a running async CLI process.",
                    ]
                )
            if self.agent_spec.can_use_web:
                descriptions.extend(
                    [
                        "web_search(statement, query): search the web for relevant pages.",
                        "web_fetch(statement, url): fetch a web page.",
                    ]
                )
            if self._running_command_states():
                descriptions.extend(
                    [
                        "check_command_status(command_id): inspect your active command.",
                        "kill_command(command_id): kill your active command.",
                    ]
                )
            if self._running_command_states():
                descriptions.append("wait(): wait 60 seconds for active command tool calls.")
            return tuple(descriptions)

        descriptions = list(BUILD_TOOL_DESCRIPTIONS)
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
        if self._running_command_states() or self._running_subagent_states():
            descriptions.append(
                "wait(): wait 60 seconds for your active command tool calls or subagents."
            )
        return tuple(descriptions)

    def _operator_runtime_context(self, session_path: Path | None) -> str:
        if session_path is None:
            return "Runtime context:\n- No active session context."

        events = self.home.read_session_events(session_path)
        plan_steps = latest_plan_steps(events)
        processes = running_process_snapshots(events)
        subagents = subagent_snapshots(events)
        lines = ["Runtime context:"]
        if plan_steps and self.agent_spec.can_use_plans:
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
        if self.agent_spec.kind == AgentKind.BUILD:
            if subagents:
                lines.append("- Subagents:")
                for subagent in subagents:
                    latest = (
                        subagent.statement
                        or subagent.response
                        or subagent.error
                        or "No output yet"
                    )
                    context = (
                        f"{subagent.context_percent}% context"
                        if subagent.context_percent
                        else "context unknown"
                    )
                    lines.append(
                        "  "
                        f"{subagent.agent_id} · {subagent.kind} · {subagent.name} · "
                        f"status={subagent.status} · {context} · latest: {latest}"
                    )
            else:
                lines.append("- Subagents: none.")
        return "\n".join(lines)

    def _process_runtime_source_label(self, process: object) -> str:
        source = str(getattr(process, "source", "")).strip()
        if source == "command":
            return "operator command"
        return "process"

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
        if self.agent_spec.kind != AgentKind.BUILD:
            return self._subagent_tool_definitions()

        statement_description = "Persistent user-visible working message for this tool call."
        tools = [
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
            {
                "name": "start_subagent",
                "description": "Start an asynchronous subagent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_kind": {
                            "type": "string",
                            "enum": ["general", "explore", "scout"],
                            "description": "Kind of subagent to start.",
                        },
                        "name": {
                            "type": "string",
                            "description": "Short display name for the subagent.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Complete task prompt for the subagent.",
                        },
                    },
                    "required": ["statement", "agent_kind", "name", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "prompt_subagent",
                "description": "Send another prompt to an idle subagent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Subagent id.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Follow-up prompt.",
                        },
                    },
                    "required": ["statement", "agent_id", "prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "remove_subagent",
                "description": "Remove a subagent from prompt context and UI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": statement_description,
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Subagent id.",
                        },
                    },
                    "required": ["statement", "agent_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "get_subagent_info",
                "description": "Inspect the latest outputs from a subagent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "Subagent id.",
                        },
                    },
                    "required": ["agent_id"],
                    "additionalProperties": False,
                },
            },
        ]
        if self._running_command_states():
            tools.extend(self._command_control_tool_definitions())
        if self._running_command_states() or self._running_subagent_states():
            tools.append(self._wait_tool_definition("active command tool calls or subagents"))
        return tools

    def _subagent_tool_definitions(self) -> list[dict[str, Any]]:
        statement_description = "Persistent working message for this tool call."
        tools: list[dict[str, Any]] = []
        if not self.agent_spec.read_only:
            tools.append(
                {
                    "name": "run_command",
                    "description": "Run a CLI command inside the trusted workspace.",
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
                                    "A single CLI command inside the trusted workspace."
                                ),
                            },
                        },
                        "required": ["statement", "command"],
                        "additionalProperties": False,
                    },
                }
            )
        if self.agent_spec.read_only:
            tools.extend(self._read_only_tool_definitions(statement_description))
        if self.agent_spec.can_start_processes:
            tools.extend(
                [
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
                                    "description": "Long-running CLI command.",
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
                ]
            )
        if self.agent_spec.can_use_web:
            tools.extend(self._web_tool_definitions(statement_description))
        if self._running_command_states():
            tools.extend(self._command_control_tool_definitions())
            tools.append(self._wait_tool_definition("active command tool calls"))
        return tools

    def _read_only_tool_definitions(self, statement_description: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "bash",
                "description": "Run a read-only shell command inside the trusted workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "command": {
                            "type": "string",
                            "description": "A command that must be classified read-only.",
                        },
                    },
                    "required": ["statement", "command"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "read",
                "description": "Read a file inside the trusted workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "path": {"type": "string", "description": "File path to read."},
                        "start_line": {"type": "integer", "description": "One-based start line."},
                        "max_lines": {"type": "integer", "description": "Maximum lines to return."},
                    },
                    "required": ["statement", "path", "start_line", "max_lines"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list",
                "description": "List a directory inside the trusted workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "path": {"type": "string", "description": "Directory path to list."},
                        "limit": {"type": "integer", "description": "Maximum entries to return."},
                    },
                    "required": ["statement", "path", "limit"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "glob",
                "description": "Find files by glob pattern inside the trusted workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "pattern": {"type": "string", "description": "Glob pattern."},
                        "path": {"type": "string", "description": "Root path for the glob."},
                        "limit": {"type": "integer", "description": "Maximum matches."},
                    },
                    "required": ["statement", "pattern", "path", "limit"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "grep",
                "description": "Search file text inside the trusted workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "pattern": {
                            "type": "string",
                            "description": "Regex or literal search text.",
                        },
                        "path": {"type": "string", "description": "File or directory path."},
                        "include": {"type": "string", "description": "File glob filter."},
                        "limit": {"type": "integer", "description": "Maximum matches."},
                    },
                    "required": ["statement", "pattern", "path", "include", "limit"],
                    "additionalProperties": False,
                },
            },
        ]

    def _web_tool_definitions(self, statement_description: str) -> list[dict[str, Any]]:
        return [
            {
                "name": "web_search",
                "description": "Search the web for relevant pages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "query": {"type": "string", "description": "Search query."},
                        "limit": {"type": "integer", "description": "Maximum results."},
                    },
                    "required": ["statement", "query", "limit"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "web_fetch",
                "description": "Fetch a web page by URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "description": statement_description},
                        "url": {"type": "string", "description": "HTTP or HTTPS URL."},
                    },
                    "required": ["statement", "url"],
                    "additionalProperties": False,
                },
            },
        ]

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
