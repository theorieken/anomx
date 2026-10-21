"""Initial onboarding flow."""

from __future__ import annotations

import shutil
import textwrap

from anomx.agent.ui.models import (
    AgentState,
    CursesWindow,
    MenuChoice,
)


class OnboardingViewMixin:
    """Initial onboarding flow."""

    def _run_onboarding(self, stdscr: CursesWindow) -> bool:
        self.state = AgentState.ONBOARDING
        execution = self._select_onboarding_execution(stdscr)
        if execution is None:
            return False
        provider = self._select_provider(stdscr)
        if provider is None:
            return False

        if provider.requires_api_key:
            api_key = self._prompt_text(
                stdscr,
                title=provider.label,
                label="API key (saved locally; leave blank to use your environment)",
                mask=True,
                optional=True,
            )
            if api_key:
                self.home.set_api_key(provider.key, api_key)
                provider = self._provider_with_discovered_models(provider, refresh=True)

        model = self._select_model(stdscr, provider)
        if model is None:
            return False

        thinking_intensity = self._select_thinking_intensity(stdscr, provider, model)
        if thinking_intensity is None:
            return False

        user_name = self._prompt_text(
            stdscr,
            title="Make Anomx Yours",
            label="What should Anomx call you?",
            optional=False,
        )
        if not user_name:
            return False

        if not provider.requires_api_key:
            self.home.set_backend_connected(provider.key, True)

        config = self.home.load_config()
        config["onboarding_complete"] = True
        config["provider"] = provider.key
        config["model"] = model
        config["thinking_intensity"] = thinking_intensity
        config["user_name"] = user_name.strip()
        config["sandbox_enabled"] = execution != "host"
        if execution != "host":
            config["sandbox_system"] = execution
            config["sandbox_method"] = "mount"
        self.home.save_config(config)
        return True

    def _onboarding_execution_choices(self) -> tuple[MenuChoice, ...]:
        """Keep unavailable engines visible but non-selectable."""
        choices = [
            MenuChoice(
                "On this host", "host", "Use this project and your installed tools directly",
            ),
        ]
        for runtime in ("docker", "podman"):
            available = shutil.which(runtime) is not None
            label = f"Secure Agent Runtime · {runtime.title()}"
            choices.append(MenuChoice(
                label if available else f"{label} — not installed",
                runtime,
                "Containerized tools; project files and user memory stay persistent",
                selectable=available,
            ))
        return tuple(choices)

    def _select_onboarding_execution(self, stdscr: CursesWindow) -> str | None:
        _, width = stdscr.getmaxyx()
        help_text = (
            "Host mode uses your installed tools directly. A container isolates tools and mounts "
            "your project, so its files and user memories stay persistent. Start Docker or Podman "
            "before using a runtime. Change this later in /config → Manage Sandbox."
        )
        help_rows = tuple(
            MenuChoice(line, "", selectable=False)
            for line in textwrap.wrap(help_text, width=max(24, width - 12))
        )
        return self._menu(
            stdscr,
            "Welcome to Anomx",
            "Choose where Anomx should work by default",
            (
                *self._onboarding_execution_choices(),
                MenuChoice("", "", selectable=False),
                *help_rows,
            ),
        )
