"""Central mode policy for every Anomx agent kind."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentMode(StrEnum):
    """Operational policy applied independently of the active agent kind."""

    PLAN = "plan"
    STANDARD = "standard"
    AUTOMATIC = "automatic"
    AUTONOMOUS = "autonomous"

    @classmethod
    def parse(cls, value: object, default: AgentMode | None = None) -> AgentMode:
        """Parse current and legacy stored values into a supported mode."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            legacy_aliases = {
                "observer": cls.STANDARD,
                "confirm": cls.STANDARD,
                "auto": cls.AUTOMATIC,
                "sandbox": cls.AUTOMATIC,
                "full_control": cls.AUTONOMOUS,
                "fullcontrol": cls.AUTONOMOUS,
            }
            if normalized in legacy_aliases:
                return legacy_aliases[normalized]
            try:
                return cls(normalized)
            except ValueError:
                pass
        return cls.STANDARD if default is None else default

    @property
    def policy(self) -> AgentModePolicy:
        """Return the complete behavior and presentation policy for this mode."""

        return mode_policy(self)

    @property
    def label(self) -> str:
        return self.policy.label

    @property
    def symbol(self) -> str:
        return self.policy.symbol

    @property
    def prompt_hint(self) -> str:
        return f"{self.symbol}  {self.label} (shift+tab to cycle)"

    @property
    def system_prompt_statement(self) -> str:
        return self.policy.system_prompt_statement


@dataclass(frozen=True, slots=True)
class AgentModePolicy:
    """All runtime and UI behavior controlled by an :class:`AgentMode`."""

    label: str
    symbol: str
    ui_attr: str
    system_prompt_statement: str
    read_only: bool = False
    requires_approval_for_unremembered: bool = False
    auto_approve_risks: frozenset[str] = frozenset()
    bypass_command_policy: bool = False

    def auto_approves_risk(self, risk: object) -> bool:
        """Return whether a classified command risk may run without a prompt."""

        return str(risk or "").strip().lower() in self.auto_approve_risks


_MODE_SEQUENCE = (
    AgentMode.PLAN,
    AgentMode.STANDARD,
    AgentMode.AUTOMATIC,
    AgentMode.AUTONOMOUS,
)

_MODE_POLICIES = {
    AgentMode.PLAN: AgentModePolicy(
        label="Plan Mode",
        symbol="Π",
        ui_attr="light",
        read_only=True,
        system_prompt_statement=(
            "Current mode: Plan. Only read operations are allowed. Commands or tools "
            "that could change files, processes, platform state, or the host are unavailable."
        ),
    ),
    AgentMode.STANDARD: AgentModePolicy(
        label="Standard Mode",
        symbol="Ω",
        ui_attr="light",
        requires_approval_for_unremembered=True,
        system_prompt_statement=(
            "Current mode: Standard. Every command that is not already remembered as "
            "approved requires user approval through the command approval UI. "
            "Do not ask for that approval in prose before calling tools. Serious "
            "host-control commands also require approval."
        ),
    ),
    AgentMode.AUTOMATIC: AgentModePolicy(
        label="Automatic Mode",
        symbol="Λ",
        ui_attr="warning",
        auto_approve_risks=frozenset({"low"}),
        system_prompt_statement=(
            "Current mode: Automatic. Read-only commands may run automatically. "
            "Approval-required commands are evaluated by the command risk classifier. "
            "Low Risk commands are approved automatically. Medium or High Risk commands "
            "require user approval through the command approval UI."
        ),
    ),
    AgentMode.AUTONOMOUS: AgentModePolicy(
        label="Autonomous Mode",
        symbol="Δ",
        ui_attr="danger",
        bypass_command_policy=True,
        system_prompt_statement=(
            "Current mode: Autonomous. Commands run without command-policy restrictions or "
            "approval prompts, including host-control and sudo commands. Apply extra care."
        ),
    ),
}


def mode_policy(mode: AgentMode | str | object) -> AgentModePolicy:
    """Return the policy for a mode or stored mode value."""

    return _MODE_POLICIES[AgentMode.parse(mode)]


def next_agent_mode(mode: AgentMode | str | object) -> AgentMode:
    """Return the next mode in the canonical UI cycle."""

    current = AgentMode.parse(mode)
    return _MODE_SEQUENCE[(_MODE_SEQUENCE.index(current) + 1) % len(_MODE_SEQUENCE)]


__all__ = ["AgentMode", "AgentModePolicy", "mode_policy", "next_agent_mode"]
