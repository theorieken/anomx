"""Initial onboarding flow."""

from __future__ import annotations

import curses
import textwrap
import time

from anomx.agent.helpers.sandbox import container_runtime_status
from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
    MenuChoice,
)


class OnboardingViewMixin:
    """Initial onboarding flow."""

    def _run_onboarding(self, stdscr: CursesWindow) -> bool:
        """Walk forward and back without losing the user's setup choices."""
        self.state = AgentState.ONBOARDING
        self._onboarding_cancelled = False
        self._onboarding_selections = {}
        execution, provider, model, thinking, user_name = "host", None, "", "auto", ""
        stage = 0
        try:
            while stage < 6:
                self._onboarding_step = (0, 1, 1, 2, 2, 3)[stage]
                if stage == 0:
                    value = self._select_onboarding_execution(stdscr)
                    if value is not None:
                        execution = value
                elif stage == 1:
                    value = self._select_provider(stdscr)
                    if value is not None:
                        provider = value
                elif stage == 2:
                    if not provider.requires_api_key:
                        stage += 1
                        continue
                    value = self._prompt_text(
                        stdscr, title=f"Connect to {provider.label}",
                        label="API key", mask=True, optional=True,
                    )
                    if value:
                        self.home.set_api_key(provider.key, value)
                        provider = self._provider_with_discovered_models(provider, refresh=True)
                elif stage == 3:
                    value = self._select_model(stdscr, provider)
                    if value is not None:
                        model = value
                elif stage == 4:
                    value = self._select_thinking_intensity(stdscr, provider, model)
                    if value is not None:
                        thinking = value
                else:
                    value = self._prompt_text(
                        stdscr, title="Make yourself at home", label="Your name",
                        optional=False, default=user_name,
                    )
                    if value:
                        user_name = value.strip()
                    elif value == "":
                        value = None
                if self._onboarding_cancelled:
                    return False
                if value is None:
                    if stage == 0:
                        return False
                    stage -= 1
                    if stage == 2 and not provider.requires_api_key:
                        stage = 1
                    # Reasoning may be implicit; going back from the name returns to models.
                    if stage == 4:
                        stage = 3
                    continue
                stage += 1

            if not provider.requires_api_key:
                self.home.set_backend_connected(provider.key, True)
            config = self.home.load_config()
            config.update(
                onboarding_complete=True, provider=provider.key, model=model,
                thinking_intensity=thinking, user_name=user_name,
                sandbox_enabled=execution != "host",
            )
            if execution != "host":
                config.update(sandbox_system=execution, sandbox_method="mount")
            self.home.save_config(config)
            self._animate_onboarding_complete(stdscr, user_name, execution)
            return True
        finally:
            self._onboarding_step = None
            self._onboarding_selections = {}

    def _onboarding_execution_choices(self) -> tuple[MenuChoice, ...]:
        choices = [MenuChoice("On this computer", "host", "Use your installed tools directly.")]
        for runtime in ("docker", "podman"):
            status = container_runtime_status(runtime)
            detail = (
                "Ready · isolated tools, persistent project files."
                if status.ready else status.guidance
            )
            choices.append(MenuChoice(
                f"{runtime.title()} runtime" if status.installed else
                f"{runtime.title()} · not installed",
                runtime, detail, selectable=status.installed,
            ))
        return tuple(choices)

    def _select_onboarding_execution(self, stdscr: CursesWindow) -> str | None:
        return self._menu(
            stdscr, "Where would you like to work?",
            "Choose your default. You can change it later in /config → Manage Sandbox.",
            self._onboarding_execution_choices(),
        )

    def _draw_onboarding_panel(
        self, stdscr: CursesWindow, *, title: str, subtitle: str,
        choices: tuple[MenuChoice, ...] = (), selected: int = 0,
        input_value: str = "", input_cursor: int = 0, show_input_cursor: bool = False,
        **_ignored: object,
    ) -> tuple[int, int, int, int]:
        """A compact, responsive setup card with vertically separated option details."""
        height, width = stdscr.getmaxyx()
        stdscr.erase()
        left = max(2, (width - 76) // 2)
        span = max(1, min(76, width - left * 2))
        top = max(1, min(4, (height - 22) // 2))
        step = getattr(self, "_onboarding_step", 0) or 0
        labels = ("Runtime", "Provider", "Model", "You")
        self._add(stdscr, top, left, "Anomx.", span, self._attr("accent") | curses.A_BOLD)
        self._add(stdscr, top + 2, left, f"SETUP  ·  {step + 1} of 4", span, self._attr("light"))
        nav = "  →  ".join(f"[{label}]" if i == step else label for i, label in enumerate(labels))
        self._add(stdscr, top + 3, left, nav, span, self._attr("accent"))
        y = top + 5
        self._add(stdscr, y, left, title, span, self._attr("bold"))
        y += 2
        if show_input_cursor:
            # The field label belongs beside the field, not in a distant screen header.
            self._add(stdscr, y, left, subtitle, span, self._attr("bold"))
            y += 1
            self._add(stdscr, y, left, "─" * span, span, self._attr("muted"))
            y += 1
            offset = max(0, input_cursor - max(1, span - 4))
            visible = input_value[offset:offset + max(1, span - 3)]
            self._add(stdscr, y, left, "› ", 2, self._attr("accent"))
            self._add(stdscr, y, left + 2, visible, span - 2)
            self._paint_visual_cursor(stdscr, y, left + 2, visible, input_cursor - offset)
            y += 1
            self._add(stdscr, y, left, "─" * span, span, self._attr("muted"))
            hint = (
                "Saved on this computer. Leave blank to keep a saved key or use your environment."
                if subtitle == "API key" else "This is how Anomx will address you."
                if subtitle == "Your name" else "Enter a value, then press Enter to continue."
            )
            for line in textwrap.wrap(hint, span):
                y += 1
                self._add(stdscr, y, left, line, span, self._attr("light"))
        else:
            for line in textwrap.wrap(subtitle, span):
                self._add(stdscr, y, left, line, span, self._attr("light"))
                y += 1
            y += 1
            # Reserve a full description line per choice, even on narrow terminals.
            capacity = max(1, (height - y - 4) // 3)
            offset = min(max(0, selected - capacity + 1), max(0, len(choices) - capacity))
            for index in range(offset, min(len(choices), offset + capacity)):
                choice = choices[index]
                attr = self._attr("muted") if not choice.selectable else (
                    self._attr("accent") if index == selected else curses.A_NORMAL
                )
                self._add(stdscr, y, left, "›" if index == selected else " ", 1, attr)
                self._add(stdscr, y, left + 2, choice.label, span - 2, attr)
                detail = textwrap.shorten(choice.detail, max(4, span - 2), placeholder="…")
                self._add(stdscr, y + 1, left + 2, detail, span - 2, self._attr("light"))
                y += 3
            if len(choices) > capacity:
                self._add(stdscr, height - 3, left,
                          f"{selected + 1} / {len(choices)} · ↑↓ to see more", span,
                          self._attr("light"))
        footer = "Enter Continue   Esc Back   Ctrl+C Exit"
        if step == 0:
            footer = "↑↓ Choose   Enter Continue   Esc Exit"
        self._add(stdscr, height - 2, left, footer, span, self._attr("light"))
        stdscr.refresh()
        return top, left, span, max(1, y - top)

    def _animate_onboarding_complete(
        self, stdscr: CursesWindow, name: str, execution: str,
    ) -> None:
        """Brief transition after settings have been saved, without pretending to load."""
        height, width = stdscr.getmaxyx()
        left = max(2, (width - 60) // 2)
        span = max(1, min(60, width - left * 2))
        y = max(1, height // 2 - 3)
        for frame in range(13):
            stdscr.erase()
            self._add(stdscr, y, left, "Anomx.", span, self._attr("accent") | curses.A_BOLD)
            self._add(stdscr, y + 2, left, "━" * (frame * min(36, span) // 12), span,
                      self._attr("accent"))
            self._add(stdscr, y + 4, left, f"You're ready, {name}.", span, self._attr("bold"))
            location = "This computer" if execution == "host" else "Secure Agent Runtime"
            self._add(stdscr, y + 5, left, f"Settings saved · {location}", span,
                      self._attr("light"))
            stdscr.refresh()
            time.sleep(0.055)
