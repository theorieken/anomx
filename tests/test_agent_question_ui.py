"""Question layout regressions: wrapping, paging, and exclusive answer input."""

import curses
from dataclasses import replace

import pytest

from anomx.agent import AnomxHome
from anomx.agent.app import AnomxCliApp
from anomx.agent.helpers.terminal import CODE_END, CODE_START, EMPHASIS_MARKERS
from anomx.agent.runtime import QuestionOption, QuestionRequest
from anomx.agent.ui import BottomPanel, MenuChoice


class QuestionWindow:
    def __init__(self, keys=(), height=28, width=80):
        self.keys = iter(keys)
        self.height = height
        self.width = width
        self.writes = []
        self.frames = []
        self.delay_modes = []
        self.attributes = []

    def getmaxyx(self):
        return self.height, self.width

    def get_wch(self):
        try:
            return next(self.keys)
        except StopIteration as error:
            raise curses.error() from error

    def nodelay(self, flag):
        self.delay_modes.append(flag)

    def erase(self):
        self.writes = []

    def addnstr(self, y, x, text, n, attr=0):
        assert 0 <= y < self.height
        self.writes.append((y, x, text[:n]))
        self.attributes.append((text[:n], attr))

    def refresh(self):
        self.frames.append(tuple(self.writes))


@pytest.fixture
def question_app(tmp_path):
    home = AnomxHome(tmp_path / "home")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = home.create_session(repo, provider="openai", model="gpt-5.5")
    return AnomxCliApp(home=home, cwd=repo, use_color=False), session


@pytest.mark.parametrize("width", [40, 80, 157])
def test_text_question_wraps_the_complete_label(question_app, width):
    app, _ = question_app
    window = QuestionWindow(height=42, width=width)
    command = "ssh -M -S /tmp/" + "long-socket-name-" * 5 + "end"
    question = "Please open the connection before continuing.\n\n" + command

    offset, maximum, _ = app._draw_text_popover(
        window, "Question", question, "ok", 2, "Esc Cancel · Enter Submit",
        replaces_prompt=True,
    )

    assert offset == maximum == 0
    heading_y = next(y for y, _, text in window.writes if text == "Question")
    input_y = next(y for y, _, text in window.writes if text == "ok")
    lines = [text for y, x, text in window.writes if heading_y < y < input_y and x == 4]
    paragraph_end = next(y for y, _, text in window.writes if "continuing." in text)
    command_start = next(y for y, _, text in window.writes if text.startswith("ssh"))
    assert command_start == paragraph_end + 2
    assert "".join("".join(lines).split()) == "".join(question.split())
    assert all(len(line) <= width - 8 for line in lines)


def test_text_question_pages_without_losing_answer_or_showing_prompt(question_app, monkeypatch):
    app, session = question_app
    question = "\n".join(f"Question line {index:02}" for index in range(45))
    window = QuestionWindow(keys=["x", *([curses.KEY_NPAGE] * 8), "\n"], height=22)
    prompts = []
    monkeypatch.setattr(app, "_draw_prompt_bar", lambda *_args, **_kwargs: prompts.append(True))

    response = app._request_question(
        window, session, QuestionRequest(question=question, kind="text", default="ok"),
    )

    shown = {text for frame in window.frames for _, _, text in frame}
    assert all(line in shown for line in question.splitlines())
    assert response.answer == "okx"
    assert prompts == []
    assert window.delay_modes == [False, True]
    assert any("PgUp/PgDn" in text for text in shown)


def test_select_question_pages_and_restores_normal_prompt(question_app, monkeypatch):
    app, session = question_app
    question = "\n".join(f"Question line {index:02}" for index in range(45))
    window = QuestionWindow(keys=[*([curses.KEY_NPAGE] * 20), curses.KEY_DOWN, "\n"], height=22)
    prompts = []
    monkeypatch.setattr(app, "_draw_prompt_bar", lambda *_args, **_kwargs: prompts.append(True))

    response = app._request_question(
        window, session, QuestionRequest(
            question=question, kind="select",
            options=(QuestionOption("First", "first"), QuestionOption("Second", "second")),
        ),
    )

    shown = {text for frame in window.frames for _, _, text in frame}
    assert all(line in shown for line in question.splitlines())
    assert response.answer == "second"
    assert response.selected_label == "Second"
    assert prompts == []
    app._draw_session(window, session, [], "draft", 5, 0)
    assert prompts == [True]


def test_question_panel_keeps_choices_inside_resized_terminal(question_app):
    app, _ = question_app
    window = QuestionWindow(height=22, width=40)
    panel = BottomPanel(
        "Question", "\n".join(f"Line {index}" for index in range(60)),
        tuple(MenuChoice(f"Option {index}", str(index)) for index in range(20)),
        selected=19, subtitle_max_lines=0, subtitle_scroll=1000, replaces_prompt=True,
    )

    for height in (22, 36, 18):
        window.height = height
        viewport = app._bottom_panel_viewport(window, panel)
        assert viewport.subtitle_lines[-1] == "Line 59"
        assert 19 in viewport.visible_indices
        assert viewport.choice_y + len(viewport.visible_indices) + 2 <= height - 1
        app._draw_bottom_panel(window, panel)

    # A newly opened question always starts with its beginning.
    initial = app._bottom_panel_viewport(window, replace(panel, subtitle_scroll=0))
    assert initial.subtitle_lines[0] == "Line 0"


@pytest.mark.parametrize("key", ["\x1b", "\x03"])
def test_text_question_cancel_keeps_response_unanswered(question_app, key):
    app, session = question_app
    window = QuestionWindow(keys=[key])

    response = app._request_question(
        window, session, QuestionRequest(question="Continue?", kind="text"),
    )

    assert response.cancelled
    assert not response.answered
    assert window.delay_modes[0] is False
    assert window.delay_modes[-1] is True


@pytest.mark.parametrize("kind", ["text", "select"])
@pytest.mark.parametrize("mouse_format", ["curses", "sgr", "sgr_fragmented"])
def test_questions_scroll_with_mouse_without_changing_answers(
    question_app, monkeypatch, kind, mouse_format,
):
    app, session = question_app
    question = "\n".join(f"## Section {index:02}" for index in range(65))
    wheel_states = [curses.BUTTON5_PRESSED] * 30 + [curses.BUTTON4_PRESSED]
    events = iter(wheel_states)
    monkeypatch.setattr(curses, "getmouse", lambda: (0, 8, 10, 0, next(events)))
    if mouse_format == "curses":
        keys = [curses.KEY_MOUSE] * len(wheel_states)
    else:
        keys = ["\x1b[<65;9;11M"] * 30 + ["\x1b[<64;9;11M"]
        if mouse_format == "sgr_fragmented":
            keys = list("".join(keys))
    if kind == "select":
        keys = [curses.KEY_DOWN, *keys]
    keys += ["\n"]
    window = QuestionWindow(keys=keys, height=22)

    response = app._request_question(
        window, session, QuestionRequest(
            question=question, kind=kind, default="typed answer",
            options=(QuestionOption("First", "first"), QuestionOption("Second", "second")),
        ),
    )

    shown = {text for frame in window.frames for _, _, text in frame}
    assert all(f"Section {index:02}" in shown for index in range(65))
    assert response.answer == ("second" if kind == "select" else "typed answer")
    headings = [text for _, _, text in window.frames[-1] if text.startswith("Question ·")]
    assert len(headings) == 1 and "/65" in headings[0] and "Scroll" in headings[0]


@pytest.mark.parametrize("kind", ["text", "select"])
def test_question_markdown_renders_styles_tables_and_literal_code(question_app, monkeypatch, kind):
    app, _ = question_app
    window = QuestionWindow(height=50, width=80)
    monkeypatch.setattr(app, "_attr", lambda name: curses.A_BOLD if name == "bold" else 0)
    question = (
        "# Question title\n\n"
        "**Bold example** and *italic example* and ***both*** and `a * 2`.\n\n"
        "| Column A | Column B |\n| --- | --- |\n| `value_a` | value_b |\n\n"
        "> A quotation\n\n"
        "```python\nresult = a ** 2\n```\n\n"
        r"Escaped \*literal\* and [Link](https://example.com)"
    )

    if kind == "text":
        app._draw_text_popover(
            window, "Question", question, "", 0, "Enter Submit", replaces_prompt=True,
        )
    else:
        app._draw_bottom_panel(window, BottomPanel(
            "Question", question, (MenuChoice("Yes", "yes"),), 0,
            replaces_prompt=True, subtitle_max_lines=0,
        ))

    shown = "\n".join(text for _, _, text in window.writes)
    assert "Question title" in shown and "# Question title" not in shown
    assert "a * 2" in shown and "result = a ** 2" in shown
    assert "*literal*" in shown and "│ A quotation" in shown
    assert "Link (https://example.com)" in shown
    assert "┌" in shown and "value_a" in shown and "value_b" in shown
    assert "**Bold example**" not in shown and "```" not in shown
    assert any(text == "Bold example" and attr & curses.A_BOLD for text, attr in window.attributes)
    italic = getattr(curses, "A_ITALIC", curses.A_UNDERLINE)
    assert any(text == "italic example" and attr & italic for text, attr in window.attributes)
    markers = (CODE_START, CODE_END, *EMPHASIS_MARKERS, *EMPHASIS_MARKERS.values())
    assert not any(marker in shown for marker in markers)


def test_long_question_inline_styles_and_words_survive_wrapping(question_app):
    app, _ = question_app
    literal = "very_long_name_" * 20
    rendered = app._question_markdown_lines(f"**{literal}** and `{literal}`", 32)
    window = QuestionWindow(height=len(rendered) + 1, width=40)

    for row, line in enumerate(rendered):
        app._draw_question_markdown_line(window, row, 0, line, 32)

    shown = "".join(text for _, _, text in window.writes)
    assert "".join(shown.split()) == literal + "and" + literal
    assert all(len(text) <= 32 for _, _, text in window.writes)
