"""Tool execution exceptions."""

from __future__ import annotations

from anomx.agent.base.exceptions import BaseException


class ToolExecutionError(BaseException):
    """Raised when a tool cannot complete its requested operation."""


class ToolNotFoundError(ToolExecutionError):
    """Raised when a model calls a tool unavailable to the active agent."""
