"""Describe known file operations without executing commands or changing their policy."""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from anomx.agent.helpers.tool_manager import CliToolManager


def file_activity_statement(
    tool_name: str,
    arguments: dict[str, Any],
    manager: CliToolManager,
    brain_dir: Path,
) -> str | None:
    """Prefer file targets and memory actions over generic model-authored labels.

    Only recognized literal operations are described. Arbitrary programs and
    dynamic paths keep their existing tool statement; this is not a safety check.
    """
    if tool_name == "memorize":
        return "Updating Memories"
    accesses: list[tuple[str, str, Path]] = []
    if tool_name in {"read", "list", "glob", "grep"}:
        accesses.append(("read", str(arguments.get("path") or "."), manager.current_dir))
    elif tool_name in {"run_command", "run_cli_command", "bash", "start_process"}:
        try:
            accesses.extend(_command_files(str(arguments.get("command") or ""), manager))
        except (OSError, ValueError, RuntimeError, RecursionError):
            return None

    writes: list[str] = []
    reads_memory = False
    writes_memory = False
    for access, raw, cwd in accesses:
        if not raw or any(char in raw for char in ("$", "`", "\n", "\r", "\x00")):
            continue
        try:
            path = Path(raw).expanduser()
            path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        except (OSError, ValueError, RuntimeError):
            continue
        if path.is_relative_to("/dev"):
            continue
        memory = path.is_relative_to(brain_dir.resolve()) or path.is_relative_to("/anomx/brain")
        if access == "write":
            if memory:
                writes_memory = True
            else:
                label = (
                    str(path.relative_to(manager.current_dir))
                    if path.is_relative_to(
                        manager.current_dir,
                    )
                    else str(path)
                )
                if label not in writes:
                    writes.append(label)
        elif memory:
            reads_memory = True
    if writes:
        label = f"Writing {', '.join(writes[:3])}"
        if len(writes) > 3:
            label += f" (+{len(writes) - 3} more)"
        return f"{label} · Updating Memories" if writes_memory else label
    if writes_memory:
        return "Updating Memories"
    return "Retrieving Memories" if reads_memory else None


def _heredoc_bodies(command: str, manager: CliToolManager) -> dict[str, str]:
    bodies: dict[str, str] = {}
    lines = iter(command.splitlines())
    for line in lines:
        for delimiter, strip_tabs in manager._heredoc_delimiters(line):
            body: list[str] = []
            for content in lines:
                content = content.lstrip("\t") if strip_tabs else content
                if content == delimiter:
                    break
                body.append(content)
            bodies[delimiter] = "\n".join(body)
    return bodies


def _command_files(
    command: str,
    manager: CliToolManager,
    cwd: Path | None = None,
    depth: int = 0,
) -> Iterator[tuple[str, str, Path]]:
    if depth > 2:
        return
    cwd = cwd or manager.current_dir
    bodies = _heredoc_bodies(command, manager)
    source = manager._strip_heredoc_bodies(command)
    for segment in manager._shell_segments(
        source,
        split_operators=frozenset(
            {
                ";",
                "&&",
                "||",
                "\n",
                "|",
            }
        ),
    ):
        redirects = [
            span for span in manager._shell_operator_spans(segment) if span[2] in {"<", ">"}
        ]
        for start, end, operator in redirects:
            if segment[start:end] == "<<" or segment[end:].startswith(("<", "&")):
                continue
            lexer = shlex.shlex(segment[end:], posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            try:
                target = next(lexer)
            except (StopIteration, ValueError):
                continue
            if target not in {"&", "-"}:
                yield ("write" if operator == ">" else "read", target, cwd)
        try:
            parts = shlex.split(segment[: redirects[0][0]] if redirects else segment)
        except ValueError:
            continue
        if not parts:
            continue
        executable = Path(parts[0]).name
        args = parts[1:]
        if executable == "cd" and len(args) == 1:
            cwd = (cwd / Path(args[0]).expanduser()).resolve()
        elif executable in {"sh", "bash", "zsh"} and len(args) >= 2 and args[0] in {"-c", "-lc"}:
            yield from _command_files(args[1], manager, cwd, depth + 1)
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
            script = args[args.index("-c") + 1] if "-c" in args and args[-1] != "-c" else ""
            if not script:
                script = next(
                    (bodies.get(key, "") for key, _ in manager._heredoc_delimiters(segment)), ""
                )
            for access, path in _python_files(script):
                yield access, path, cwd
        elif executable == "apply_patch":
            patch = (
                args[0]
                if args
                else next(
                    (bodies.get(key, "") for key, _ in manager._heredoc_delimiters(segment)), ""
                )
            )
            for path in re.findall(
                r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)$",
                patch,
                re.M,
            ):
                yield "write", path, cwd
        elif executable in {"tee", "touch"}:
            for path in args:
                if not path.startswith("-"):
                    yield "write", path, cwd
        elif executable in {"cp", "mv", "install"} and len(args) >= 2:
            yield "write", args[-1], cwd
        elif executable == "sed" and any(
            arg.startswith(("--in-place", "-i")) for arg in args
        ):
            # BSD sed takes an empty backup suffix as a separate argument.
            # Script files are reads, not write targets of the in-place edit.
            edit_args = []
            index = 0
            while index < len(args):
                arg = args[index]
                if arg == "-i" and args[index + 1 : index + 2] == [""]:
                    index += 2
                    continue
                if arg in {"-f", "--file"}:
                    index += 2
                    edit_args.extend(["-e", ""])
                    continue
                if arg.startswith(("--file=", "-f")):
                    edit_args.extend(["-e", ""])
                else:
                    edit_args.append(arg)
                index += 1
            for path in manager._sed_path_candidate_arguments(edit_args):
                yield "write", path, cwd
        elif executable in {"cat", "head", "tail", "less", "ls", "find", "rg", "grep", "sed", "wc"}:
            for path in manager._path_candidate_arguments(parts):
                yield "read", path, cwd


def _python_files(source: str) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(source)
        reader = _PythonFiles()
        reader.visit(tree)
        return reader.accesses
    except (SyntaxError, ValueError, RecursionError):
        return []


class _PythonFiles(ast.NodeVisitor):
    """Extract literal paths from common pathlib/open operations; never evaluate code."""

    def __init__(self) -> None:
        self.names: dict[str, str | None] = {}
        self.accesses: list[tuple[str, str]] = []

    def path(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = self.path(node.left), self.path(node.right)
            if left is not None and right is not None:
                return str(Path(left) / right)
            return None
        if isinstance(node, ast.Call):
            name = (
                node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            )
            if name in {"Path", "PosixPath"} and node.args:
                paths = [self.path(arg) for arg in node.args]
                if all(path is not None for path in paths):
                    return str(Path(*[path for path in paths if path is not None]))
        return None

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.names[target.id] = self.path(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        path = None
        access = "read"
        if name == "open":
            path = self.path(node.func.value) if isinstance(node.func, ast.Attribute) else None
            mode_index = 0 if path is not None else 1
            if path is None and node.args:
                path = self.path(node.args[0])
            mode = node.args[mode_index] if len(node.args) > mode_index else ast.Constant("r")
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), mode)
            mode_text = self.path(mode)
            if mode_text is None:
                path = None
            elif any(flag in mode_text for flag in "wax+"):
                access = "write"
        elif isinstance(node.func, ast.Attribute) and name in {
            "read_text",
            "read_bytes",
            "write_text",
            "write_bytes",
            "touch",
        }:
            path = self.path(node.func.value)
            access = "read" if name.startswith("read_") else "write"
        if path is not None:
            self.accesses.append((access, path))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.AST) -> None:
        pass

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
