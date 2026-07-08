"""Skill loading and persistence for the Anomx CLI agent."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal

SkillSource = Literal["builtin", "user"]

BUILTIN_SKILL_PACKAGE = "anomx.agent.skills"
SKILL_README_NAMES = ("README.md", "readme.md")
BUILTIN_MARKER_NAME = ".anomx_builtin"
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
    system: bool = False
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


def load_builtin_skills(*, include_system: bool = False) -> tuple[Skill, ...]:
    """Load bundled skills from package resources."""

    skill_root = files(BUILTIN_SKILL_PACKAGE)
    loaded: list[Skill] = []
    for resource in sorted(skill_root.iterdir(), key=lambda item: item.name):
        skill: Skill | None = None
        if resource.is_dir():
            readme = _resource_readme(resource)
            if readme is None:
                continue
            skill = parse_skill_markdown(
                readme.read_text(encoding="utf-8"),
                default_command=resource.name,
                source="builtin",
                path=None,
            )
        elif resource.name.endswith(".md"):
            skill = parse_skill_markdown(
                resource.read_text(encoding="utf-8"),
                default_command=resource.name.removesuffix(".md"),
                source="builtin",
                path=None,
            )
        if skill is None:
            continue
        if skill.system and not include_system:
            continue
        loaded.append(skill)
    return tuple(loaded)


def load_system_skills() -> tuple[Skill, ...]:
    """Load bundled skills that are injected into system prompts."""

    return tuple(skill for skill in load_builtin_skills(include_system=True) if skill.system)


def load_user_skills(skills_dir: Path) -> tuple[Skill, ...]:
    """Load user-created skills from the global Anomx home directory."""

    if not skills_dir.exists():
        return ()
    loaded: list[Skill] = []
    seen: set[str] = set()
    for path in sorted(item for item in skills_dir.iterdir() if item.is_dir()):
        readme = _path_readme(path)
        if readme is None:
            continue
        skill = parse_skill_markdown(
            readme.read_text(encoding="utf-8"),
            default_command=path.name,
            source="user",
            path=path,
        )
        loaded.append(skill)
        seen.add(skill.command)
    for path in sorted(skills_dir.glob("*.md")):
        if not path.is_file():
            continue
        default_command = path.stem
        if normalize_skill_command(default_command) in seen:
            continue
        skill = parse_skill_markdown(
            path.read_text(encoding="utf-8"),
            default_command=default_command,
            source="user",
            path=path,
        )
        loaded.append(skill)
    return tuple(loaded)


def write_user_skill(skills_dir: Path, skill: Skill) -> Path:
    """Persist a user-created folder skill with README.md instructions."""

    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / skill.command
    path.mkdir(parents=True, exist_ok=True)
    tmp_path = path / "README.md.tmp"
    tmp_path.write_text(skill_to_markdown(skill), encoding="utf-8")
    tmp_path.replace(path / "README.md")
    return path


def sync_builtin_skills(skills_dir: Path, *, include_system: bool = False) -> None:
    """Materialize bundled folder skills under the Anomx home skills directory."""

    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_root = files(BUILTIN_SKILL_PACKAGE)
    for resource in sorted(skill_root.iterdir(), key=lambda item: item.name):
        source_readme = _resource_readme(resource) if resource.is_dir() else None
        if source_readme is None and not resource.name.endswith(".md"):
            continue
        skill = parse_skill_markdown(
            (source_readme or resource).read_text(encoding="utf-8"),
            default_command=(
                resource.name if resource.is_dir() else resource.name.removesuffix(".md")
            ),
            source="builtin",
            path=None,
        )
        if skill.system and not include_system:
            continue

        target_dir = skills_dir / skill.command
        marker_path = target_dir / BUILTIN_MARKER_NAME
        if target_dir.exists() and not marker_path.exists():
            continue
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if resource.is_dir():
            _copy_resource_tree(resource, target_dir)
        else:
            (target_dir / "README.md").write_text(
                resource.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        marker_path.write_text("synced bundled Anomx skill\n", encoding="utf-8")


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
    system = _metadata_bool(metadata.get("system"))
    return Skill(
        command=command,
        title=title,
        description=description,
        body=body.strip(),
        source=source,
        hidden=hidden,
        system=system,
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
    if skill.system:
        frontmatter.append("system: true")
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


def _resource_readme(resource: Traversable) -> Traversable | None:
    for readme_name in SKILL_README_NAMES:
        readme = resource / readme_name
        if readme.is_file():
            return readme
    return None


def _path_readme(path: Path) -> Path | None:
    for readme_name in SKILL_README_NAMES:
        readme = path / readme_name
        if readme.is_file():
            return readme
    return None


def _copy_resource_tree(resource: Traversable, target_dir: Path) -> None:
    for child in resource.iterdir():
        if child.name == "__pycache__" or child.name.endswith((".pyc", ".pyo")):
            continue
        target = target_dir / child.name
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _copy_resource_tree(child, target)
        elif child.is_file():
            target.write_bytes(child.read_bytes())


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
