"""Skill loading and persistence for the Anomx CLI agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

SkillSource = Literal["builtin", "user"]

BUILTIN_SKILL_PACKAGE = "anomx.agent.skills"
STARTER_SKILL_COMMANDS = ("map-folder", "find-issues", "make-report")
_COMMAND_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Skill:
    """A prompt-backed skill that can be invoked from the slash command menu."""

    command: str
    title: str
    description: str
    body: str
    source: SkillSource
    hidden: bool = False
    path: Path | None = None

    @property
    def slash_command(self) -> str:
        """Return the command as typed in the prompt."""

        return f"/{self.command}"


def normalize_skill_command(command: str) -> str:
    """Normalize a user-entered skill command into its slash-command name."""

    stripped = command.strip().removeprefix("/").strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", stripped).strip("-_")
    return normalized[:64]


def is_valid_skill_command(command: str) -> bool:
    """Return whether a normalized command is safe to use as a skill command."""

    return bool(_COMMAND_PATTERN.fullmatch(command))


def load_builtin_skills() -> tuple[Skill, ...]:
    """Load bundled skills from package resources."""

    skill_root = files(BUILTIN_SKILL_PACKAGE)
    loaded: list[Skill] = []
    for resource in sorted(skill_root.iterdir(), key=lambda item: item.name):
        if not resource.name.endswith(".md"):
            continue
        skill = parse_skill_markdown(
            resource.read_text(encoding="utf-8"),
            default_command=resource.name.removesuffix(".md"),
            source="builtin",
            path=None,
        )
        loaded.append(skill)
    return tuple(loaded)


def load_user_skills(skills_dir: Path) -> tuple[Skill, ...]:
    """Load user-created skills from the global Anomx home directory."""

    if not skills_dir.exists():
        return ()
    loaded: list[Skill] = []
    for path in sorted(skills_dir.glob("*.md")):
        if not path.is_file():
            continue
        skill = parse_skill_markdown(
            path.read_text(encoding="utf-8"),
            default_command=path.stem,
            source="user",
            path=path,
        )
        loaded.append(skill)
    return tuple(loaded)


def write_user_skill(skills_dir: Path, skill: Skill) -> Path:
    """Persist a user-created skill as Markdown frontmatter plus instructions."""

    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / f"{skill.command}.md"
    tmp_path = path.with_suffix(".md.tmp")
    tmp_path.write_text(skill_to_markdown(skill), encoding="utf-8")
    tmp_path.replace(path)
    return path


def parse_skill_markdown(
    content: str,
    *,
    default_command: str,
    source: SkillSource,
    path: Path | None,
) -> Skill:
    """Parse a skill Markdown file with simple YAML-like frontmatter."""

    metadata, body = _split_frontmatter(content)
    command = normalize_skill_command(str(metadata.get("command", default_command)))
    if not is_valid_skill_command(command):
        command = normalize_skill_command(default_command)
    title = _metadata_text(metadata.get("title") or metadata.get("name"))
    if not title:
        title = _title_from_command(command) if source == "builtin" else command
    description = _metadata_text(metadata.get("description")) or _first_body_paragraph(body)
    hidden = _metadata_bool(metadata.get("hidden"))
    return Skill(
        command=command,
        title=title,
        description=description,
        body=body.strip(),
        source=source,
        hidden=hidden,
        path=path,
    )


def skill_to_markdown(skill: Skill) -> str:
    """Serialize a skill to the Markdown format used by Anomx."""

    frontmatter = [
        "---",
        f"command: {skill.command}",
        f"description: {_single_line(skill.description)}",
    ]
    if skill.hidden:
        frontmatter.append("hidden: true")
    frontmatter.append("---")
    return "\n".join([*frontmatter, "", skill.body.strip(), ""])


def skill_invocation_prompt(skill: Skill, arguments: str = "") -> str:
    """Return the prompt content sent to the backend for a skill invocation."""

    sections = [
        f"Use the Anomx skill /{skill.command}: {skill.title}.",
        f"Description: {skill.description}",
        "Skill instructions:",
        skill.body.strip(),
    ]
    stripped_arguments = arguments.strip()
    if stripped_arguments:
        sections.extend(["User arguments:", stripped_arguments])
    return "\n\n".join(section for section in sections if section)


def _split_frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, content

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :])
            return metadata, body
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().lower()
        if normalized_key:
            metadata[normalized_key] = value.strip().strip("\"'")
    return {}, content


def _metadata_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return _single_line(value)


def _metadata_bool(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _single_line(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def _title_from_command(command: str) -> str:
    return command.replace("-", " ").replace("_", " ").title()


def _first_body_paragraph(body: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in body.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return ""
    return _single_line(paragraphs[0])
