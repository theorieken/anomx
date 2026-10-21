"""Prompt-subagent tool."""

from __future__ import annotations

from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.delegation import append_assignment, entrypoint_prompt
from anomx.agent.helpers.utils import session_id_from_path


class PromptSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="prompt_subagent",
            description="Send another prompt to an idle subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "agent_id": {"type": "string", "description": "Subagent id."},
                    "prompt": {"type": "string", "description": "Follow-up prompt."},
                },
                ["statement", "agent_id", "prompt"],
            ),
            aliases=("prompt_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if (
            not context.runtime.agent_spec.can_spawn_subagents
            or not context.runtime.subagents_enabled
        ):
            return context.json_result(
                {"error": "Only the main agent with /subagents enabled can prompt subagents."}
            )
        if context.session_path is None:
            return context.json_result({"error": "prompt_subagent requires a session."})

        agent_id = str(arguments.get("agent_id", "")).strip()
        prompt = str(arguments.get("prompt", "")).strip()
        if not agent_id:
            return context.json_result({"error": "prompt_subagent requires an agent_id."})
        if not prompt:
            return context.json_result({"error": "prompt_subagent requires a prompt."})

        context.runtime._bind_session_home(context.session_path)
        context.runtime._restore_subagent(agent_id, context.session_path)
        with context.runtime._subagent_lock:
            state = context.runtime._subagents.get(agent_id)
            if state is None or state.status == "removed":
                return context.json_result({"error": "Unknown subagent id."})
            if state.status in {"running", "working"}:
                return context.json_result({"error": "Subagent is already running."})
            try:
                append_assignment(state, prompt)
            except (OSError, ValueError) as error:
                return context.json_result({"error": f"Cannot write subagent follow-up: {error}"})
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
                local_sandbox_session = context.runtime.local_sandbox_session
                state.runtime = context.runtime.__class__(
                    context.runtime.home,
                    state.directory or context.runtime.cwd,
                    context.runtime.session_allowed_commands,
                    context.runtime.session_rejected_commands,
                    context.runtime.tool_manager.mode,
                    agent_kind=state.kind,
                    cancel_event=state.cancel_event,
                    workspace_root=context.runtime.workspace_root,
                    process_owner_id=state.agent_id,
                    process_owner_name=state.name,
                    local_sandbox_enabled=local_sandbox_session is not None,
                    local_sandbox_home=local_sandbox_session.home
                    if local_sandbox_session is not None
                    else None,
                    local_sandbox_allow_subprocess=(
                        local_sandbox_session.config.allow_subprocess
                        if local_sandbox_session is not None
                        else False
                    ),
                    additional_instructions=context.runtime.additional_instructions,
                    mode_provider=context.runtime.mode_provider,
                    before_model_request=context.runtime.before_model_request,
                    background_api_scoped=context.runtime.background_api_scoped,
                    api_key_resolver=context.runtime.api_key_resolver,
                    background_backend_resolver=context.runtime.background_backend_resolver,
                    additional_trusted_roots=context.runtime.additional_trusted_roots,
                    path_aliases=context.runtime._legacy_path_aliases,
                    context_summarizer=context.runtime.context_summarizer,
                    context_summary_context_window=context.runtime.context_summary_context_window,
                    context_optimizer=context.runtime.context_optimizer,
                    context_optimizer_context_window=context.runtime.context_optimizer_context_window,
                )
                state.runtime._sandbox_session = context.runtime.sandbox_session
                if state.runtime._sandbox_session is not None:
                    state.runtime.tool_manager.path_aliases.update(
                        state.runtime._sandbox_session.path_aliases
                    )
                    state.runtime.tool_manager.strict_workspace = True
                state.runtime._parent_session_id = session_id_from_path(context.session_path)
        context.runtime._publish_subagent_state(
            state,
            context.session_path,
            message=state.statement,
        )
        context.runtime._start_subagent_worker(
            state,
            entrypoint_prompt(state) if state.directory is not None else prompt,
            context.session_path,
            context.callbacks,
        )
        return context.json_result(
            {
                "prompted": True,
                "agent_id": state.agent_id,
                "name": state.name,
                "status": state.status,
                "directory": str(state.directory or ""),
            }
        )
