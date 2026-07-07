"""Agent approval modes for CLI command policy."""

from __future__ import annotations

from enum import StrEnum


class AgentMode(StrEnum):
    """Command approval policy used by an agent."""

    CONFIRM = "confirm"
    AUTO = "auto"
    AUTONOMOUS = "autonomous"
    SANDBOX = "sandbox"

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
                "automatic": cls.AUTO,
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
            AgentMode.CONFIRM: "Standard Mode",
            AgentMode.AUTO: "Automatic Mode",
            AgentMode.AUTONOMOUS: "Autonomous Mode",
            AgentMode.SANDBOX: "Sandbox Mode",
        }
        return labels[self]

    @property
    def symbol(self) -> str:
        """Return the prompt symbol for this mode."""

        symbols = {
            AgentMode.CONFIRM: "Ω",
            AgentMode.AUTO: "Λ",
            AgentMode.AUTONOMOUS: "Δ",
            AgentMode.SANDBOX: "□",
        }
        return symbols[self]

    @property
    def prompt_hint(self) -> str:
        """Return the compact prompt hint for this mode."""

        if self == AgentMode.SANDBOX:
            return "□  Sandbox Mode (disabled in config)"
        return f"{self.symbol}  {self.label} (shift+tab to cycle)"

    @property
    def system_prompt_statement(self) -> str:
        """Return the system prompt policy statement for this mode."""

        statements = {
            AgentMode.CONFIRM: (
                "Current mode: Standard Mode. Read-only commands and workspace navigation "
                "may run automatically. Commands that may compute, install, execute, "
                "or modify files require user approval through the command approval UI. "
                "Do not ask for that approval in prose before calling tools. Serious "
                "host-control commands also require approval."
            ),
            AgentMode.AUTO: (
                "Current mode: Automatic Mode. Read-only commands may run automatically. "
                "Approval-required commands are evaluated by the command risk classifier. "
                "Low Risk commands are approved automatically. Medium or High Risk commands "
                "require user approval through the command approval UI."
            ),
            AgentMode.AUTONOMOUS: (
                "Current mode: Autonomous Mode. Valid commands may run automatically. "
                "Very severe host-control commands such as "
                "killall, reboot, shutdown, sudo, diskutil, mount, or systemctl still "
                "remain blocked by command policy."
            ),
            AgentMode.SANDBOX: (
                "Current mode: Sandbox Mode. Most commands run automatically inside "
                "the configured sandbox runtime. Version-control and serious host-control "
                "commands require user approval through the command approval UI."
            ),
        }
        return statements[self]

    @property
    def sandbox_disabled_hint(self) -> str:
        """Return prompt hint with sandbox-disabled notice."""
        return "□  Sandbox Mode (disabled in config)"
