"""Central mode policy for every Anomx agent kind."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AgentMode(StrEnum):
    """Operational policy applied independently of the active agent kind."""

    BACKGROUND = "background"
    RECOMMEND = "background"
    REQUEST_APPROVALS = "standard"
    STANDARD = "standard"
    AUTO_APPROVE = "automatic"
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
                "plan": cls.AUTOMATIC,
                "recommend": cls.BACKGROUND,
                "observer": cls.STANDARD,
                "confirm": cls.STANDARD,
                "request_approvals": cls.STANDARD,
                "auto_approve": cls.AUTOMATIC,
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
        return cls.AUTOMATIC if default is None else default

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
    recommendations_only: bool = False
    requires_approval_for_unremembered: bool = False
    auto_approve_risks: frozenset[str] = frozenset()
    bypass_command_policy: bool = False

    def auto_approves_risk(self, risk: object) -> bool:
        """Return whether a classified command risk may run without a prompt."""

        return str(risk or "").strip().lower() in self.auto_approve_risks


_MODE_SEQUENCE = (
    AgentMode.STANDARD,
    AgentMode.AUTOMATIC,
    AgentMode.AUTONOMOUS,
)

_CONNECTED_MODE_SEQUENCE = _MODE_SEQUENCE

_MODE_POLICIES = {
    AgentMode.RECOMMEND: AgentModePolicy(
        label="Background",
        symbol="B",
        ui_attr="accent",
        recommendations_only=True,
        auto_approve_risks=frozenset({"low"}),
        system_prompt_statement=(
            "Current mode: Background. Work unattended; ask_question and approval prompts "
            "are unavailable. Auto Approve is active: the evaluator decides whether "
            "commands are authorized and read-only without asking the user. Denied or "
            "unassessable commands do not execute. Automatic approval never expands the "
            "background access scope. Make conservative assumptions and report missing information "
            "in your final response. Use get_background_runs to inspect previous runs and "
            "avoid duplicate findings. Read operations are allowed. The only permitted "
            "persistent write by default is creating a recommendation through "
            "POST /recommendations. "
            "A platform-provided policy and scoped API credential may permit direct create, "
            "update or delete operations on specifically listed object models. Recommend all "
            "other changes. Do not run, stop, accept or reject records, "
            "or change files or processes."
        ),
    ),
    AgentMode.STANDARD: AgentModePolicy(
        label="Request Approvals",
        symbol="Ω",
        ui_attr="light",
        auto_approve_risks=frozenset({"low"}),
        system_prompt_statement=(
            "Current mode: Request Approvals. Read-only inspection and authorized, bounded "
            "reversible actions run without interruption. Ask only for consequential actions "
            "whose target or effects lack authorization. Existing authorization persists; "
            "do not ask for redundant confirmation. Approval requests are asynchronous by "
            "default: continue independent work, then use check_approval with the returned "
            "approval_id to collect the decision and execute the approved action. Never "
            "assume a pending action has executed or retry it as a new command. Set "
            "wait_for_approval=true when a critical dependency requires waiting. Do not "
            "request approval in prose before invoking the tool."
        ),
    ),
    AgentMode.AUTOMATIC: AgentModePolicy(
        label="Auto Approve",
        symbol="Λ",
        ui_attr="warning",
        auto_approve_risks=frozenset({"low"}),
        system_prompt_statement=(
            "Current mode: Auto Approve. The approval evaluator decides on the user's behalf "
            "whether actions are authorized. Never ask the user for approval, including "
            "through prose or ask_question. Routine reads and authorized reversible actions "
            "proceed. Actions the evaluator denies, or cannot assess, do not execute; "
            "report the concrete blocker and continue other useful work. Respect earlier "
            "authorization and subsequent restrictions."
        ),
    ),
    AgentMode.AUTONOMOUS: AgentModePolicy(
        label="Autonomous",
        symbol="Δ",
        ui_attr="danger",
        bypass_command_policy=True,
        system_prompt_statement=(
            "Current mode: Autonomous. Skip command classification and approval evaluation; "
            "run actions without approval prompts. Respect explicit user denials, available "
            "credentials, sandbox isolation and tool-specific access restrictions. Apply "
            "extra care to consequential actions."
        ),
    ),
}


def mode_policy(mode: AgentMode | str | object) -> AgentModePolicy:
    """Return the policy for a mode or stored mode value."""

    return _MODE_POLICIES[AgentMode.parse(mode)]


def next_agent_mode(
    mode: AgentMode | str | object,
    *,
    platform_connected: bool = False,
) -> AgentMode:
    """Return the next mode in the canonical UI cycle."""

    current = AgentMode.parse(mode)
    sequence = _CONNECTED_MODE_SEQUENCE if platform_connected else _MODE_SEQUENCE
    if current not in sequence:
        return sequence[0]
    return sequence[(sequence.index(current) + 1) % len(sequence)]


__all__ = ["AgentMode", "AgentModePolicy", "mode_policy", "next_agent_mode"]
