# Agent terminal UI

Read the [harness guide](../AGENTS.md). `views/` owns screens such as session,
project, config and onboarding; `components/` owns reusable prompt bars, message
rendering, popups and status bars. `models.py` contains UI state contracts;
activity, constants and wave modules support presentation.

Keep runtime execution, platform access and approval policy outside rendering.
Use existing callbacks and state models. Preserve keyboard/focus behavior,
terminal width/cell calculations, resize handling and noninteractive command mode.
Keep updates bounded and avoid blocking the input loop. Share activity/status
presentation instead of duplicating animation and color logic in every view.

Use concise product text and established prompts. Do not expose internal
reasoning or turn failures into successful-looking completion messages.
