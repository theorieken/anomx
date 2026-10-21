"""Start-subagent tool."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from anomx.agent.base.agents import AgentKind
from anomx.agent.base.subagents import SubagentRuntimeState
from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.helpers.delegation import create_subagent_directory, entrypoint_prompt
from anomx.agent.helpers.utils import session_id_from_path, utc_now_iso

SUBAGENT_MAX_CONCURRENT = 5


class StartSubagentTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="start_subagent",
            description="Start an asynchronous subagent.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "name": {
                        "type": "string",
                        "description": "Short display name for the subagent.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Complete task prompt for the subagent.",
                    },
                },
                ["statement", "name", "prompt"],
            ),
            aliases=("start_agent",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        if (
            not context.runtime.agent_spec.can_spawn_subagents
            or not context.runtime.subagents_enabled
        ):
            return context.json_result(
                {"error": "The user must enable /subagents before the main agent can delegate."}
            )
        if context.session_path is None:
            return context.json_result({"error": "start_subagent requires a session."})

        kind = AgentKind.SUB

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return context.json_result({"error": "start_subagent requires a prompt."})

        with context.runtime._subagent_lock:
            visible_agents = [
                agent for agent in context.runtime._subagents.values() if agent.status != "removed"
            ]
            if len(visible_agents) >= SUBAGENT_MAX_CONCURRENT:
                return context.json_result(
                    {
                        "error": "At most five subagents can be active at the same time.",
                        "limit": SUBAGENT_MAX_CONCURRENT,
                    }
                )

        agent_id = uuid4().hex[:8]
        name = context.runtime._subagent_display_name(
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
        context.runtime._bind_session_home(context.session_path)
        try:
            directory = create_subagent_directory(
                context.runtime.home.subagent_session_path(
                    parent_session_path=context.session_path,
                    subagent_name=name,
                    subagent_id=agent_id,
                ).parent,
                state,
                context.runtime.cwd,
            )
        except (OSError, ValueError) as error:
            return context.json_result({"error": f"Cannot prepare subagent directory: {error}"})
        local_sandbox_session = context.runtime.local_sandbox_session
        child_runtime = context.runtime.__class__(
            context.runtime.home,
            directory,
            context.runtime.session_allowed_commands,
            context.runtime.session_rejected_commands,
            context.runtime.tool_manager.mode,
            agent_kind=kind,
            cancel_event=state.cancel_event,
            workspace_root=context.runtime.workspace_root,
            process_owner_id=agent_id,
            process_owner_name=name,
            local_sandbox_enabled=local_sandbox_session is not None,
            local_sandbox_home=(
                local_sandbox_session.home if local_sandbox_session is not None else None
            ),
            local_sandbox_allow_subprocess=(
                local_sandbox_session.config.allow_subprocess
                if local_sandbox_session is not None
                else False
            ),
            additional_instructions=context.runtime.additional_instructions,
            mode_provider=context.runtime.mode_provider,
            before_model_request=context.runtime.before_model_request,
            context_summarizer=context.runtime.context_summarizer,
            context_summary_context_window=context.runtime.context_summary_context_window,
            context_optimizer=context.runtime.context_optimizer,
            context_optimizer_context_window=context.runtime.context_optimizer_context_window,
            background_api_scoped=context.runtime.background_api_scoped,
            api_key_resolver=context.runtime.api_key_resolver,
            background_backend_resolver=context.runtime.background_backend_resolver,
            additional_trusted_roots=context.runtime.additional_trusted_roots,
            path_aliases=context.runtime._legacy_path_aliases,
        )
        child_runtime._sandbox_session = context.runtime.sandbox_session
        if child_runtime._sandbox_session is not None:
            child_runtime.tool_manager.path_aliases.update(child_runtime._sandbox_session.path_aliases)
            child_runtime.tool_manager.strict_workspace = True
        state.runtime = child_runtime
        child_runtime._parent_session_id = session_id_from_path(context.session_path)
        with context.runtime._subagent_lock:
            context.runtime._subagents[agent_id] = state

        payload = context.runtime._publish_subagent_state(
            state,
            context.session_path,
            message=statement,
        )
        callback = getattr(context.callbacks, "subagent", None)
        if payload is not None and callback is not None:
            callback(payload)
        context.runtime._start_subagent_worker(
            state,
            entrypoint_prompt(state),
            context.session_path,
            context.callbacks,
        )
        return context.json_result(
            {
                "started": True,
                "agent_id": agent_id,
                "agent_kind": kind.value,
                "name": name,
                "status": state.status,
                "directory": str(directory),
                "entrypoint": str(directory / "AGENT.md"),
                "inbox": str(directory / "INBOX.md"),
                "outbox": str(directory / "OUTBOX.md"),
            }
        )
