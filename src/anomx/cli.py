"""Command line entry points for Anomx."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from anomx import __version__

AI_PROVIDER_KEYS = ("desy", "blablador", "anthropic", "openai", "ollama", "kimi")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="anomx",
        description="Anomx system intelligence agent.",
        epilog='Pass a request, e.g. anomx "list folders", to get a matching shell command.',
    )
    parser.add_argument(
        "request",
        nargs="*",
        help="Describe a task to get a shell command for it instead of opening the interface.",
    )
    parser.add_argument("--version", action="version", version=f"anomx {__version__}")
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Override the Anomx home directory (defaults to ANOMX_HOME or ~/.anomx).",
    )
    parser.add_argument(
        "--print-home",
        action="store_true",
        help="Print the resolved Anomx home directory and exit.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in the interactive shell.",
    )
    parser.add_argument(
        "--shell-init",
        choices=("zsh", "bash"),
        default=None,
        help="Print a shell function that puts suggested commands on the prompt line.",
    )
    parser.add_argument(
        "--provider",
        choices=AI_PROVIDER_KEYS,
        default=None,
        help="Start with a specific AI backend.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Start with a specific model.",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Start with the local Ollama backend.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    home_path = args.home if args.home is not None else _resolve_anomx_home()
    if args.print_home:
        print(home_path)
        return 0
    if args.shell_init:
        from anomx.agent.command_mode import shell_init_script

        print(shell_init_script(args.shell_init), end="")
        return 0
    request = " ".join(args.request).strip()
    if request:
        return _run_command_mode(args, home_path, request)
    return _run_interface(args, home_path)


def _run_command_mode(args: argparse.Namespace, home_path: Path, request: str) -> int:
    """Translate one request into a shell command with the selected model."""

    from anomx.agent.command_mode import run_command_mode
    from anomx.agent.store import AnomxHome

    return run_command_mode(
        AnomxHome(home_path),
        request,
        provider=_startup_provider(args.provider, args.ollama),
        model=_startup_model(args.model),
        use_color=not args.no_color,
    )


def _run_interface(args: argparse.Namespace, home_path: Path) -> int:
    """Import and run the full interface after the lightweight entry phase."""

    from anomx.agent import AnomxCliApp, AnomxHome

    provider = _startup_provider(args.provider, args.ollama)
    model = _startup_model(args.model)
    app = AnomxCliApp(
        home=AnomxHome(home_path),
        startup_provider=provider,
        startup_model=model,
        use_color=not args.no_color,
        isolate_runtime=True,
    )
    return int(app.run())


def _resolve_anomx_home() -> Path:
    configured = os.environ.get("ANOMX_HOME")
    return Path(configured).expanduser() if configured else Path.home() / ".anomx"


def _startup_provider(provider: str | None, ollama: bool) -> str | None:
    if ollama:
        return "ollama"
    if provider is not None:
        return provider
    configured_provider = os.environ.get("ANOMX_PROVIDER")
    if configured_provider:
        return configured_provider
    if os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA_MODEL_ID"):
        return "ollama"
    return None


def _startup_model(model: str | None) -> str | None:
    return (
        model
        or os.environ.get("ANOMX_MODEL")
        or os.environ.get("OLLAMA_MODEL")
        or os.environ.get("OLLAMA_MODEL_ID")
    )
