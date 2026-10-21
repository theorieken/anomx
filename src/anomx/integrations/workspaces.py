"""Check out and save a shared job workspace through the authenticated API.

``python -m anomx.integrations.workspaces mount <component> <directory>``
creates a local working copy and a revision manifest. Edit with ordinary file
and shell tools, then ``python -m anomx.integrations.workspaces save <directory>``.
Saving uses optimistic concurrency: HTTP 409 requires a reload and merge.
No credentials are written to the checkout; PlatformClient reads explicit
ANOMX_URL/ANOMX_TOKEN configuration. Mounting does not start the job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from .platform import PlatformClient, PlatformError

MANIFEST = ".anomx-workspace.json"


def component_id(reference: str) -> str:
    """Normalize a UUID or a canonical/legacy component object reference."""
    for prefix in ("jobs_job_component-", "jobs_job_work-"):
        reference = reference.removeprefix(prefix)
    return str(UUID(reference))


def validate_files(files: dict[str, str]) -> None:
    """Reject unsafe paths, binary content and oversized workspaces locally."""
    if not isinstance(files, dict) or len(files) > 256:
        raise ValueError("A workspace supports at most 256 entries.")
    size = 0
    for name, value in files.items():
        path = PurePosixPath(name)
        if (
            not name
            or len(name) > 512
            or name == MANIFEST
            or path.is_absolute()
            or "\\" in name
            or "\x00" in name
            or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        ):
            raise ValueError(f"Invalid workspace path: {name!r}.")
        if not isinstance(value, str) or "\x00" in value or name.endswith("/") and value:
            raise ValueError("Workspace entries must contain text; folders must be empty.")
        if (
            name.endswith("/")
            and name.rstrip("/") in files
            or any(str(parent) in files for parent in path.parents)
        ):
            raise ValueError("A file cannot also be a folder.")
        size += len(value.encode("utf-8"))
    if size > 2 * 1024 * 1024:
        raise ValueError("Workspace files exceed 2 MiB.")


def workspace_path(client: PlatformClient, identifier: str, *, job: bool = False) -> str:
    """Resolve the workspace route for legacy or versioned API roots."""
    prefix = "/jobs" if getattr(client, "url", "").rstrip("/").endswith("/v1") else ""
    resource = "jobs" if job else "job-works"
    return f"{prefix}/{resource}/{identifier}/workspace"


def mount(
    reference: str, directory: str | Path, *, client: PlatformClient | None = None
) -> dict[str, Any]:
    """Materialize a fresh working copy without replacing existing local files."""
    client = client or PlatformClient.from_env()
    is_job = reference.startswith("jobs_job-")
    identifier = str(UUID(reference.removeprefix("jobs_job-"))) if is_job else component_id(reference)
    target = Path(directory).expanduser()
    if target.is_symlink() or target.exists() and any(target.iterdir()):
        raise ValueError("Choose a new or empty directory for the workspace.")
    response = client.request(workspace_path(client, identifier, job=is_job), method="GET")
    if not isinstance(response, dict) or not isinstance(response.get("files"), dict):
        raise PlatformError("Invalid workspace response.")
    files = response["files"]
    validate_files(files)
    target.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = target / name
        if name.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    manifest = {
        "job" if is_job else "component": identifier,
        "revision": response["revision"],
        "entrypoint": response["entrypoint"],
    }
    (target / MANIFEST).write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")
    return manifest


def save(directory: str | Path, *, client: PlatformClient | None = None) -> dict[str, Any]:
    """Save the complete working copy if the remote revision still matches."""
    client = client or PlatformClient.from_env()
    target = Path(directory).expanduser()
    if target.is_symlink() or (target / MANIFEST).is_symlink():
        raise ValueError("A workspace and its manifest must not be symbolic links.")
    manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("entrypoint"), str)
        or not isinstance(manifest.get("revision"), int)
        or isinstance(manifest.get("revision"), bool)
    ):
        raise ValueError("Invalid workspace manifest.")
    is_job = "job" in manifest
    identifier = str(UUID(manifest["job"])) if is_job else component_id(manifest["component"])
    files = {}
    for path in target.rglob("*"):
        relative = path.relative_to(target).as_posix()
        if relative == MANIFEST or any(
            part in {"__pycache__", ".git", ".venv"} for part in path.relative_to(target).parts
        ):
            continue
        if path.is_symlink():
            raise ValueError("Workspace symbolic links are not supported.")
        if path.is_file():
            if path.stat().st_size > 2 * 1024 * 1024:
                raise ValueError("Workspace file exceeds 2 MiB.")
            files[relative] = path.read_text(encoding="utf-8")
        elif path.is_dir() and not any(path.iterdir()):
            files[relative + "/"] = ""
    validate_files(files)
    response = client.request(
        workspace_path(client, identifier, job=is_job),
        {"files": files, "entrypoint": manifest["entrypoint"], "revision": manifest["revision"]},
        method="PUT",
    )
    if not isinstance(response, dict) or "revision" not in response:
        raise PlatformError("Invalid workspace save response.")
    manifest.update(revision=response["revision"], entrypoint=response["entrypoint"])
    (target / MANIFEST).write_text(json.dumps(manifest, indent=4) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entry point usable by the Anomx agent's ordinary terminal tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checkout = commands.add_parser("mount")
    checkout.add_argument("reference", help="jobs_job-UUID or a legacy component reference")
    checkout.add_argument("directory")
    push = commands.add_parser("save")
    push.add_argument("directory")
    args = parser.parse_args()
    result = (
        mount(args.reference, args.directory) if args.command == "mount" else save(args.directory)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
