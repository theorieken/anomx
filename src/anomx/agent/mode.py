"""Agent execution modes for CLI command policy."""

from __future__ import annotations

from enum import StrEnum


class AgentMode(StrEnum):
    """User-selectable command execution mode."""

    CONFIRM = "confirm"
    AUTO = "auto"
    AUTONOMOUS = "autonomous"

    @classmethod
    def parse(cls, value: object, default: AgentMode | None = None) -> AgentMode:
        """Parse a stored config value into a valid agent mode."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            legacy_aliases = {
                "observer": cls.CONFIRM,
                "full_control": cls.AUTONOMOUS,
                "fullcontrol": cls.AUTONOMOUS,
            }
            if normalized in legacy_aliases:
                return legacy_aliases[normalized]
            for mode in cls:
                if mode.value == normalized:
                    return mode
        return cls.CONFIRM if default is None else default

    @property
    def label(self) -> str:
        """Return the human-readable mode label."""

        labels = {
            AgentMode.CONFIRM: "Confirm Mode",
            AgentMode.AUTO: "Auto Mode",
            AgentMode.AUTONOMOUS: "Autonomous Mode",
        }
        return labels[self]

    @property
    def symbol(self) -> str:
        """Return the prompt symbol for this mode."""

        symbols = {
            AgentMode.CONFIRM: "Ω",
            AgentMode.AUTO: "Λ",
            AgentMode.AUTONOMOUS: "Δ",
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
            AgentMode.CONFIRM: (
                "Current mode: Confirm Mode. Read-only commands and workspace navigation "
                "may run automatically. Commands that may compute, install, execute, "
                "or modify files require user approval through the command approval UI. "
                "Do not ask for that approval in prose before calling tools. Serious "
                "host-control commands also require approval."
            ),
            AgentMode.AUTO: (
                "Current mode: Auto Mode. Known read, compute, install, execute, and "
                "file-modifying commands may run automatically inside the trusted "
                "workspace. Unknown, structurally ambiguous, or serious host-control "
                "commands require user approval through the command approval UI."
            ),
            AgentMode.AUTONOMOUS: (
                "Current mode: Autonomous Mode. Valid commands may run automatically "
                "inside the trusted workspace. Serious host-control commands such as "
                "killall, reboot, shutdown, sudo, diskutil, mount, or systemctl still "
                "require user approval through the command approval UI."
            ),
        }
        return statements[self]

    def next(self) -> AgentMode:
        """Return the next mode in the Shift+Tab cycle."""

        order = (
            AgentMode.CONFIRM,
            AgentMode.AUTO,
            AgentMode.AUTONOMOUS,
        )
        return order[(order.index(self) + 1) % len(order)]
