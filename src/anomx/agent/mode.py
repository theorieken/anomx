"""Agent execution modes for CLI command policy."""

from __future__ import annotations

from enum import StrEnum


class AgentMode(StrEnum):
    """User-selectable command execution mode."""

    OBSERVER = "observer"
    CONFIRM = "confirm"
    AUTONOMOUS = "autonomous"

    @classmethod
    def parse(cls, value: object, default: AgentMode | None = None) -> AgentMode:
        """Parse a stored config value into a valid agent mode."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for mode in cls:
                if mode.value == normalized:
                    return mode
        return cls.CONFIRM if default is None else default

    @property
    def label(self) -> str:
        """Return the human-readable mode label."""

        labels = {
            AgentMode.OBSERVER: "Observer Mode",
            AgentMode.CONFIRM: "Confirm Mode",
            AgentMode.AUTONOMOUS: "Autonomous Mode",
        }
        return labels[self]

    @property
    def symbol(self) -> str:
        """Return the prompt symbol for this mode."""

        symbols = {
            AgentMode.OBSERVER: "Ω",
            AgentMode.CONFIRM: "Δ",
            AgentMode.AUTONOMOUS: "Λ",
        }
        return symbols[self]

    @property
    def prompt_hint(self) -> str:
        """Return the compact prompt hint for this mode."""

        return f"{self.symbol}  {self.label} (shift+tab to cycle)"

    @property
    def system_prompt_statement(self) -> str:
        """Return the system prompt policy statement for this mode."""

        statements = {
            AgentMode.OBSERVER: (
                "Current mode: Observer Mode. You may only inspect the repository with "
                "read-only commands. Do not request commands that modify files, install "
                "packages, run tests or builds, start services, or control the host."
            ),
            AgentMode.CONFIRM: (
                "Current mode: Confirm Mode. Read-only commands may run automatically. "
                "Commands that may compute, install, execute, or modify files require "
                "user approval through the command approval UI. Do not ask for that "
                "approval in prose before calling tools. Dangerous host-control commands "
                "are blocked."
            ),
            AgentMode.AUTONOMOUS: (
                "Current mode: Autonomous Mode. You may run read, compute, install, "
                "execute, and file-modifying commands without asking. Dangerous "
                "host-control commands and commands outside the trusted workspace remain "
                "blocked."
            ),
        }
        return statements[self]

    def next(self) -> AgentMode:
        """Return the next mode in the Shift+Tab cycle."""

        order = (AgentMode.OBSERVER, AgentMode.CONFIRM, AgentMode.AUTONOMOUS)
        return order[(order.index(self) + 1) % len(order)]
