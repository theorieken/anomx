"""Persistent storage for the Anomx CLI agent.

The CLI intentionally keeps its state in a small, inspectable home directory,
similar in spirit to Codex's local CLI state:

```
~/.anomx/
  config.toml
  auth.json
  skills/<command>.md
  session_index.jsonl
  sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl
```
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from anomx.agent.mode import AgentMode

DEFAULT_HOME_NAME = ".anomx"
ANOMX_HOME_ENV = "ANOMX_HOME"


@dataclass(frozen=True)
class ProviderOption:
    """AI backend option exposed by the CLI onboarding flow."""

    key: str
    label: str
    models: tuple[str, ...]
    allow_custom_model: bool = False


@dataclass(frozen=True)
class ModelMetadata:
    """Model information used for selection and context tracking."""

    model: str
    display_name: str
    context_window: int | None
    max_output_tokens: int | None


@dataclass(frozen=True)
class ThinkingIntensityOption:
    """Provider/model-specific thinking intensity selection."""

    value: str
    label: str
    detail: str


@dataclass(frozen=True)
class SessionRecord:
    """A stored CLI session transcript."""

    session_id: str
    path: Path
    created_at: str
    updated_at: str
    cwd: str
    provider: str
    model: str
    title: str


AI_PROVIDERS: tuple[ProviderOption, ...] = (
    ProviderOption("openai", "OpenAI", ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")),
    ProviderOption(
        "anthropic",
        "Anthropic",
        (
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ),
    ),
    ProviderOption(
        "desy",
        "DESY Assistant",
        ("desy-assistant", "reasoning", "coding"),
        allow_custom_model=True,
    ),
    ProviderOption(
        "ollama",
        "Ollama",
        ("qwen3.6", "qwen3-coder:30b", "qwen2.5-coder:32b"),
        allow_custom_model=True,
    ),
)

AI_PROVIDER_KEYS = tuple(provider.key for provider in AI_PROVIDERS)

THINKING_INTENSITY_AUTO = "auto"
THINKING_INTENSITY_OPTIONS: dict[str, ThinkingIntensityOption] = {
    THINKING_INTENSITY_AUTO: ThinkingIntensityOption(
        THINKING_INTENSITY_AUTO,
        "Provider default",
        "Use the backend default for this model",
    ),
    "minimal": ThinkingIntensityOption(
        "minimal",
        "Minimal",
        "Fastest OpenAI reasoning mode",
    ),
    "low": ThinkingIntensityOption(
        "low",
        "Low",
        "Prioritize speed and lower token use",
    ),
    "medium": ThinkingIntensityOption(
        "medium",
        "Medium",
        "Balance reasoning depth and latency",
    ),
    "high": ThinkingIntensityOption(
        "high",
        "High",
        "Use deeper reasoning for harder tasks",
    ),
    "xhigh": ThinkingIntensityOption(
        "xhigh",
        "Extra high",
        "Claude long-horizon agentic work",
    ),
    "max": ThinkingIntensityOption(
        "max",
        "Max",
        "Claude maximum capability mode",
    ),
}

MODEL_METADATA: dict[str, ModelMetadata] = {
    "gpt-5.5": ModelMetadata("gpt-5.5", "GPT-5.5", 1_000_000, 128_000),
    "gpt-5.4": ModelMetadata("gpt-5.4", "GPT-5.4", 1_000_000, 128_000),
    "gpt-5.4-mini": ModelMetadata("gpt-5.4-mini", "GPT-5.4 mini", 400_000, 128_000),
    "claude-opus-4-8": ModelMetadata(
        "claude-opus-4-8",
        "Claude Opus 4.8",
        1_000_000,
        128_000,
    ),
    "claude-sonnet-4-6": ModelMetadata(
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        1_000_000,
        64_000,
    ),
    "claude-haiku-4-5-20251001": ModelMetadata(
        "claude-haiku-4-5-20251001",
        "Claude Haiku 4.5",
        200_000,
        64_000,
    ),
    "qwen3.6": ModelMetadata("qwen3.6", "Qwen 3.6", None, None),
    "qwen3-coder:30b": ModelMetadata("qwen3-coder:30b", "Qwen3 Coder 30B", None, None),
    "qwen2.5-coder:32b": ModelMetadata(
        "qwen2.5-coder:32b",
        "Qwen2.5 Coder 32B",
        None,
        None,
    ),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "onboarding_complete": False,
    "provider": "openai",
    "model": "gpt-5.5",
    "thinking_intensity": THINKING_INTENSITY_AUTO,
    "agent_mode": AgentMode.CONFIRM.value,
    "require_trusted_repo": True,
    "history_persistence": "save_all",
    "last_session_id": None,
    "platform_url": None,
    "platform_last_url": None,
    "platform_last_email": None,
    "projects": {},
}

CONFIG_SCALAR_FIELDS = (
    "schema_version",
    "onboarding_complete",
    "provider",
    "model",
    "thinking_intensity",
    "agent_mode",
    "history_persistence",
    "last_session_id",
    "platform_url",
    "platform_last_url",
    "platform_last_email",
)


def default_config() -> dict[str, Any]:
    """Return a mutable default config without sharing nested values."""

    config = DEFAULT_CONFIG.copy()
    config["projects"] = {}
    return config


def resolve_anomx_home(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the Anomx home directory.

    `ANOMX_HOME` mirrors Codex-style home overrides and is useful for tests,
    isolated development runs, and future packaged desktop environments.
    """

    source = os.environ if env is None else env
    configured_home = source.get(ANOMX_HOME_ENV)
    if configured_home:
        return Path(configured_home).expanduser()
    return Path.home() / DEFAULT_HOME_NAME


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSONL events."""

    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def provider_by_key(provider_key: str) -> ProviderOption | None:
    """Return a configured provider option by key."""

    return next((provider for provider in AI_PROVIDERS if provider.key == provider_key), None)


def model_metadata(model: str) -> ModelMetadata | None:
    """Return metadata for a configured model."""

    return MODEL_METADATA.get(model)


def model_context_window(model: str) -> int | None:
    """Return the model context window in tokens when known."""

    metadata = model_metadata(model)
    return None if metadata is None else metadata.context_window


def model_detail(model: str) -> str:
    """Return a compact model detail string for menus."""

    metadata = model_metadata(model)
    if metadata is None:
        return "custom context"
    parts: list[str] = []
    if metadata.context_window is not None:
        parts.append(f"{_format_token_count(metadata.context_window)} context")
    if metadata.max_output_tokens is not None:
        parts.append(f"{_format_token_count(metadata.max_output_tokens)} max output")
    return " · ".join(parts) if parts else "custom context"


def normalize_thinking_intensity(value: object) -> str:
    """Return a known thinking intensity value, falling back to provider defaults."""

    intensity = str(value or THINKING_INTENSITY_AUTO).strip().lower()
    return intensity if intensity in THINKING_INTENSITY_OPTIONS else THINKING_INTENSITY_AUTO


def thinking_intensity_options(
    provider_key: str,
    model: str,
) -> tuple[ThinkingIntensityOption, ...]:
    """Return supported thinking intensity options for a provider/model pair."""

    if provider_key == "openai" and model.startswith("gpt-5"):
        return tuple(
            THINKING_INTENSITY_OPTIONS[value]
            for value in (THINKING_INTENSITY_AUTO, "minimal", "low", "medium", "high")
        )
    if provider_key == "anthropic":
        if model == "claude-opus-4-8":
            return tuple(
                THINKING_INTENSITY_OPTIONS[value]
                for value in (THINKING_INTENSITY_AUTO, "low", "medium", "high", "xhigh", "max")
            )
        if model in {"claude-opus-4-6", "claude-sonnet-4-6"}:
            return tuple(
                THINKING_INTENSITY_OPTIONS[value]
                for value in (THINKING_INTENSITY_AUTO, "low", "medium", "high", "max")
            )
    return ()


def thinking_intensity_supported(provider_key: str, model: str) -> bool:
    """Return whether the selected provider/model exposes thinking intensity."""

    return bool(thinking_intensity_options(provider_key, model))


def _format_token_count(tokens: int) -> str:
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    if tokens >= 1_000 and tokens % 1_000 == 0:
        return f"{tokens // 1_000}K"
    return f"{tokens:,}"


class AnomxHome:
    """Read and write the persistent Anomx CLI home directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = resolve_anomx_home() if root is None else root.expanduser()

    @property
    def config_path(self) -> Path:
        """Return the global CLI config path."""

        return self.root / "config.toml"

    @property
    def auth_path(self) -> Path:
        """Return the auth and API-key metadata path."""

        return self.root / "auth.json"

    @property
    def trusted_repos_path(self) -> Path:
        """Return the legacy trusted repository registry path."""

        return self.root / "trusted_repos.json"

    @property
    def session_index_path(self) -> Path:
        """Return the append-only session index path."""

        return self.root / "session_index.jsonl"

    @property
    def sessions_dir(self) -> Path:
        """Return the session transcript root."""

        return self.root / "sessions"

    @property
    def skills_dir(self) -> Path:
        """Return the global user-created skills directory."""

        return self.root / "skills"

    def ensure(self) -> None:
        """Create required home directories."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def has_config(self) -> bool:
        """Return whether the home directory already has a persisted config."""

        return self.config_path.exists()

    def load_config(self) -> dict[str, Any]:
        """Load the CLI config, applying defaults for missing fields."""

        config = default_config()
        config.update(self._read_toml_object(self.config_path))
        config["agent_mode"] = AgentMode.parse(config.get("agent_mode")).value
        config["thinking_intensity"] = normalize_thinking_intensity(
            config.get("thinking_intensity")
        )
        config["require_trusted_repo"] = True
        projects = config.get("projects")
        if not isinstance(projects, dict):
            config["projects"] = {}
        return config

    def save_config(self, config: Mapping[str, Any]) -> None:
        """Save the CLI config."""

        self.ensure()
        merged = default_config()
        merged.update(dict(config))
        merged["agent_mode"] = AgentMode.parse(merged.get("agent_mode")).value
        merged["thinking_intensity"] = normalize_thinking_intensity(
            merged.get("thinking_intensity")
        )
        merged["require_trusted_repo"] = True
        self._write_config_toml(self.config_path, merged)

    def load_auth(self) -> dict[str, Any]:
        """Load provider credentials metadata."""

        default_auth: dict[str, Any] = {"schema_version": 1, "api_keys": {}, "platform": {}}
        auth = default_auth.copy()
        auth.update(self._read_json_object(self.auth_path, default={}))
        api_keys = auth.get("api_keys")
        if not isinstance(api_keys, dict):
            auth["api_keys"] = {}
        platform = auth.get("platform")
        if not isinstance(platform, dict):
            auth["platform"] = {}
        return auth

    def save_auth(self, auth: Mapping[str, Any]) -> None:
        """Save provider credentials metadata with owner-only file permissions."""

        self.ensure()
        payload = {"schema_version": 1, "api_keys": {}, "platform": {}}
        payload.update(dict(auth))
        self._write_json_object(self.auth_path, payload, mode=0o600)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key for a provider."""

        auth = self.load_auth()
        api_keys = cast(dict[str, str], auth["api_keys"])
        api_keys[provider] = api_key
        self.save_auth(auth)

    def has_api_key(self, provider: str) -> bool:
        """Return whether an API key is configured for a provider."""

        auth = self.load_auth()
        api_keys = cast(dict[str, str], auth["api_keys"])
        return bool(api_keys.get(provider))

    def set_platform_connection(
        self,
        *,
        url: str,
        token: str,
        user_email: str = "",
        organization_url: str = "",
        hostname: str = "",
    ) -> None:
        """Store the connected Anomx Platform endpoint and bearer token."""

        config = self.load_config()
        config["platform_url"] = url
        self.save_config(config)

        auth = self.load_auth()
        auth["platform"] = {
            "token": token,
            "user_email": user_email,
            "organization_url": organization_url,
            "hostname": hostname,
        }
        self.save_auth(auth)

    def set_platform_form_defaults(self, *, url: str, email: str) -> None:
        """Persist the non-secret platform connection form defaults."""

        config = self.load_config()
        config["platform_last_url"] = url
        config["platform_last_email"] = email
        self.save_config(config)

    def clear_platform_connection(self) -> None:
        """Remove the connected Anomx Platform endpoint and bearer token."""

        config = self.load_config()
        config["platform_url"] = None
        self.save_config(config)

        auth = self.load_auth()
        auth["platform"] = {}
        self.save_auth(auth)

    def platform_connection(self) -> dict[str, str] | None:
        """Return the configured platform connection when a URL and token exist."""

        config = self.load_config()
        url = str(config.get("platform_url") or "").strip()
        auth = self.load_auth()
        platform = auth.get("platform")
        if not isinstance(platform, dict):
            return None
        token = str(platform.get("token") or "").strip()
        if not url or not token:
            return None
        return {
            "url": url,
            "token": token,
            "user_email": str(platform.get("user_email") or ""),
            "organization_url": str(platform.get("organization_url") or ""),
            "hostname": str(platform.get("hostname") or ""),
        }

    def has_platform_connection(self) -> bool:
        """Return whether the CLI has a platform URL and bearer token."""

        return self.platform_connection() is not None

    def clear_sessions(self, keep_session_path: Path | None = None) -> None:
        """Delete stored session history and rebuild session metadata."""

        keep_path = (
            keep_session_path.expanduser().resolve() if keep_session_path is not None else None
        )
        records = self.list_sessions(limit=None)
        remaining_records: list[SessionRecord] = []
        for record in records:
            record_path = record.path.expanduser().resolve()
            if keep_path is not None and record_path == keep_path:
                remaining_records.append(record)
                continue
            record.path.unlink(missing_ok=True)

        if self.sessions_dir.exists():
            for directory in sorted(self.sessions_dir.rglob("*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        continue
            if keep_path is None:
                shutil.rmtree(self.sessions_dir, ignore_errors=True)
        self.session_index_path.unlink(missing_ok=True)
        for record in reversed(remaining_records):
            self._append_session_index(record)

        config = self.load_config()
        config["last_session_id"] = remaining_records[0].session_id if remaining_records else None
        self.save_config(config)

    def is_repo_trusted(self, repo_path: Path) -> bool:
        """Return whether the repository path has already passed access check."""

        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        repo_key = self._repo_key(repo_path)
        repo_entry = projects.get(repo_key)
        if not isinstance(repo_entry, dict):
            return self._is_repo_trusted_in_legacy_file(repo_key)
        return repo_entry.get("trust_level") == "trusted"

    def trust_repo(self, repo_path: Path) -> None:
        """Mark a repository path as trusted."""

        self.ensure()
        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        projects[self._repo_key(repo_path)] = {
            "trust_level": "trusted",
            "trusted_at": utc_now_iso(),
        }
        self.save_config(config)

    def create_session(self, cwd: Path, provider: str, model: str) -> SessionRecord:
        """Create an empty session transcript and index entry."""

        self.ensure()
        now = utc_now_iso()
        session_id = uuid4().hex
        date_parts = datetime.now(tz=UTC).strftime("%Y/%m/%d")
        session_dir = self.sessions_dir / date_parts
        session_dir.mkdir(parents=True, exist_ok=True)
        filename_timestamp = now.removesuffix("Z").replace(":", "-")
        session_path = session_dir / f"rollout-{filename_timestamp}-{session_id}.jsonl"
        record = SessionRecord(
            session_id=session_id,
            path=session_path,
            created_at=now,
            updated_at=now,
            cwd=str(cwd.expanduser().resolve()),
            provider=provider,
            model=model,
            title="New session",
        )
        metadata = {
            "id": record.session_id,
            "session_id": record.session_id,
            "timestamp": record.created_at,
            "created_at": record.created_at,
            "cwd": record.cwd,
            "originator": "anomx_cli",
            "source": "cli",
            "model_provider": record.provider,
            "provider": record.provider,
            "model": record.model,
            "title": record.title,
        }
        self._append_jsonl(
            session_path,
            {"timestamp": now, "type": "session_meta", "payload": metadata},
        )
        self._append_session_index(record)
        config = self.load_config()
        config["last_session_id"] = record.session_id
        self.save_config(config)
        return record

    def append_session_event(
        self,
        session_path: Path,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Append an event to a stored session transcript."""

        event_payload = {"type": event_type}
        event_payload.update(dict(payload))
        self._append_jsonl(
            session_path,
            {"timestamp": utc_now_iso(), "type": "event_msg", "payload": event_payload},
        )

    def update_session_title(self, session_path: Path, title: str) -> None:
        """Update the title stored in the session metadata event."""

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        payload["title"] = title
        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def list_sessions(self, limit: int | None = 20) -> list[SessionRecord]:
        """List recently created sessions, newest first."""

        if not self.sessions_dir.exists():
            return []

        records = [
            record
            for path in self.sessions_dir.rglob("*.jsonl")
            if (record := self._read_session_record(path)) is not None
        ]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records if limit is None else records[:limit]

    def read_session_events(self, session_path: Path) -> list[dict[str, Any]]:
        """Read all JSONL events for a session transcript."""

        if not session_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with session_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    events.append(cast(dict[str, Any], json.loads(stripped)))
        return events

    def _append_session_index(self, record: SessionRecord) -> None:
        payload = {
            "timestamp": record.created_at,
            "type": "session_index",
            "payload": {
                "session_id": record.session_id,
                "path": str(record.path),
                "cwd": record.cwd,
                "provider": record.provider,
                "model": record.model,
                "title": record.title,
            },
        }
        self._append_jsonl(self.session_index_path, payload)

    def _read_session_record(self, path: Path) -> SessionRecord | None:
        events = self.read_session_events(path)
        if not events:
            return None

        first_event = events[0]
        if first_event.get("type") != "session_meta":
            return None

        metadata = cast(dict[str, Any], first_event.get("payload", {}))
        last_event = events[-1]
        title = self._session_title(metadata, events)
        return SessionRecord(
            session_id=str(metadata.get("id", metadata.get("session_id", path.stem))),
            path=path,
            created_at=str(metadata.get("created_at", first_event.get("timestamp", ""))),
            updated_at=str(last_event.get("timestamp", metadata.get("created_at", ""))),
            cwd=str(metadata.get("cwd", "")),
            provider=str(metadata.get("model_provider", metadata.get("provider", ""))),
            model=str(metadata.get("model", "")),
            title=title,
        )

    def _session_title(self, metadata: Mapping[str, Any], events: list[dict[str, Any]]) -> str:
        configured_title = metadata.get("title")
        if isinstance(configured_title, str) and configured_title != "New session":
            return configured_title

        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type not in {"user_message", "skill_invocation"}:
                continue
            message = str(payload.get("message", "")).strip()
            if message:
                return message[:60]
        return str(configured_title or "New session")

    def _is_repo_trusted_in_legacy_file(self, repo_key: str) -> bool:
        trusted = self._read_json_object(self.trusted_repos_path, default={"repos": {}})
        repos = trusted.get("repos")
        if not isinstance(repos, dict):
            return False
        repo_entry = repos.get(repo_key)
        if not isinstance(repo_entry, dict):
            return False
        return bool(repo_entry.get("trusted"))

    def _read_toml_object(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def _write_config_toml(self, path: Path, config: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for field in CONFIG_SCALAR_FIELDS:
            value = config.get(field)
            if value is None:
                continue
            lines.append(f"{field} = {self._format_toml_value(value)}")

        projects = config.get("projects")
        if isinstance(projects, dict):
            for project_path, project in sorted(projects.items()):
                if not isinstance(project, dict):
                    continue
                lines.append("")
                lines.append(f"[projects.{json.dumps(str(project_path))}]")
                for key, value in sorted(project.items()):
                    lines.append(f"{key} = {self._format_toml_value(value)}")

        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _format_toml_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value))

    def _read_json_object(self, path: Path, default: Mapping[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return dict(default)
        return cast(dict[str, Any], data)

    def _write_json_object(
        self,
        path: Path,
        payload: Mapping[str, Any],
        mode: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
        if mode is not None:
            path.chmod(mode)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")

    def _repo_key(self, repo_path: Path) -> str:
        return str(repo_path.expanduser().resolve())
