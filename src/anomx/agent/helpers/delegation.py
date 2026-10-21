"""Durable Markdown handoffs for asynchronous agent collaboration."""

from __future__ import annotations

from pathlib import Path

from anomx.agent.base.subagents import SubagentRuntimeState
from anomx.agent.helpers.utils import utc_now_iso


def write_markdown(path: Path, text: str, *, append: bool = False) -> None:
    """Write a protocol file without following a replaced file symlink."""

    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite a symbolic link: {path}")
    with path.open("a" if append else "w", encoding="utf-8") as stream:
        stream.write(text)


def create_subagent_directory(directory: Path, state: SubagentRuntimeState, task: Path) -> Path:
    """Create a named handoff directory and an immutable initial assignment."""

    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    state.directory = directory
    write_markdown(
        directory / "AGENT.md",
        (
            f"# {state.name}\n\n"
            f"Agent ID: `{state.agent_id}`\n\n"
            f"Parent task directory: `{task}`\n\n"
            "## Assignment\n\n" + state.prompt + "\n\n"
            "## Communication protocol\n\n"
            "Start here, then read INBOX.md for the latest assignment or follow-up. "
            "Your working directory is this folder; save scratch work and artifacts here. "
            "Read or edit the parent task only when your assignment requires it, and "
            "coordinate overlapping edits through the main agent.\n\n"
            "Keep useful progress, evidence, artifact paths, and blockers in OUTBOX.md. "
            "Finish with a concise answer; the runtime appends it to OUTBOX.md, updates "
            "STATUS.md, and notifies the main agent. The main agent can retrigger you with "
            "prompt_subagent, which appends a new entry to INBOX.md. Do not modify INBOX.md "
            "or STATUS.md. You have no direct user channel and cannot spawn other agents.\n"
        ),
    )
    write_markdown(directory / "INBOX.md", "# Main agent to subagent\n")
    write_markdown(directory / "OUTBOX.md", "# Subagent to main agent\n")
    append_assignment(state, state.prompt)
    return directory


def append_assignment(state: SubagentRuntimeState, prompt: str) -> None:
    """Persist a new task before starting or retriggering a worker."""

    if state.directory is not None:
        write_markdown(
            state.directory / "INBOX.md",
            f"\n## {utc_now_iso()}\n\n{prompt}\n",
            append=True,
        )


def publish_status(state: SubagentRuntimeState) -> None:
    """Keep a small durable status document alongside the handoff files."""

    if state.directory is not None:
        write_markdown(
            state.directory / "STATUS.md",
            (
                f"# {state.name}\n\nStatus: {state.status}\n\nUpdated: {utc_now_iso()}\n\n"
                f"{state.statement or state.error or state.response}\n"
            ),
        )


def publish_result(state: SubagentRuntimeState) -> None:
    """Append a result without discarding notes written by the delegated agent."""

    if state.directory is not None:
        write_markdown(
            state.directory / "OUTBOX.md",
            (f"\n## {state.finished_at} — {state.status}\n\n{state.error or state.response}\n"),
            append=True,
        )


def entrypoint_prompt(state: SubagentRuntimeState) -> str:
    """Use local files as the source of the assignment on every invocation."""

    return (
        f"Your entry point is {state.directory}/AGENT.md. Read it and INBOX.md now, "
        "then complete the latest assignment. Keep progress and results in OUTBOX.md."
    )
