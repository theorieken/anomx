"""Interactive request value objects shared by runtime and tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionOption:
    """A user-selectable option for an operator question."""

    label: str
    value: str
    description: str = ""


@dataclass(frozen=True)
class QuestionRequest:
    """Interactive question request shown by the CLI."""

    question: str
    kind: str
    options: tuple[QuestionOption, ...] = ()
    placeholder: str = ""
    default: str = ""
    allow_custom: bool = False


@dataclass(frozen=True)
class QuestionResponse:
    """Answer returned from an interactive CLI question."""

    answered: bool
    answer: str = ""
    selected_label: str = ""
    kind: str = ""
    cancelled: bool = False
