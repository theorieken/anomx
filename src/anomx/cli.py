"""Command line entry points for Anomx."""

from __future__ import annotations

import argparse

from anomx import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="anomx", description="Anomx time-series anomaly toolkit.")
    parser.add_argument("--version", action="version", version=f"anomx {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
