# Changelog

## 0.2.88

- Stream CLI responses immediately, including unfinished lines, and fold completed activity as the response arrives.
- Reuse compatible busy sandboxes, defer image replacement while work is active, and isolate chats that need different mounts.
- Simplify sandbox startup messaging and keep explicit container cleanup scoped to managed Anomx containers.
- Clarify shared job checkouts and frozen run workspace documentation.

## 0.2.87

- Modernize training and job skills around native Dataset, TimeSeries, models, transforms, metrics and scorers, with a tested LSTM template and remote compute as the default.
- Distinguish agent-shell dependencies from compute-runtime capabilities and expose safe interpreter/package metadata.
- Support selected and downsampled dataset views through hosted Work callbacks, with authorization-scoped caching and versioned source selections.
- Journal tool starts before execution so interrupted calls are identified as having unknown outcomes and reconciled before retrying side effects.
- Refresh the curated DESY, JSC, Anthropic, OpenAI and Moonshot model catalogs, filtering retired models against live provider discovery.
- Restore background startup update prompts and add `/update` for an explicit version check, installation and restart.

## 0.2.86

- Simplify CLI navigation with a cursor dot, typing-to-prompt focus, primary-colored active session titles, and animated activity waves.
- Use compact two-line headers with model names, launch paths, and sandbox indicators.
- Add a persistent context display choice between percentages and token counts, available by clicking the context indicator or in Design settings.
- Organize configuration with breadcrumbs and inline Sandbox and Debug Mode settings; keep Commands as a settings subpage.
- Remove the legacy Manage Memories screens while retaining brain-folder notes.
- Keep the personal `tmp.py` experiment outside version control and release artifacts.

## 0.2.85

- Add native time-series containers, processing, metrics, forecasting, scoring and detection APIs, with upstream attribution and optional ecosystem integrations.
- Add portable local experiment execution, model adapters and result artifacts.
- Checkpoint remote dataset pages under `~/.anomx/cache/datasets`, resume interrupted transfers, and share cached pages across preview limits and batch sizes.
- Add lazy time-window selections and server-side downsampling with explicit capability checks and versioned query caches.
- Separate snapshot reuse from content validation, protect cache journals against concurrent writes, and preserve valid prefixes after interruption or corruption.
- Document the local experiment workflow and default its example to the complete selected source.

## 0.2.84

- Correct note-sync notification labels so agent activity no longer displays the internal `system` role.
- Let subagents share their parent's note synchronization instead of repeating it independently.
- Simplify the slash menu to `/goal` and `/subagents`; active modes can still be disabled with their chips.

## 0.2.82

- Reuse one persistent `anomx-agent` container, starting it only when the first message is submitted.
- Keep each chat in `/anomx/sessions/<chat-id>/` with its own task, temporary, sub-agent, and transcript files.
- Keep host projects unmounted by default; provide a bounded launch-directory listing as context and require explicit task-specific approval through `mount_folder` before granting read/write folder access.
- Preserve session files across idle runtime image and mount updates, and refuse replacement while commands or background processes are active.

## 0.2.81

- Store agent notes as ordinary Markdown files, migrate legacy notes safely, and merge connected CLI/platform notes with numbered conflict copies.
- Use quieter notes guidance and place removable goal/sub-agent controls alongside the CLI mode selector.

## 0.2.80

- Redesign CLI onboarding with a compact step indicator, separate descriptions, labelled inputs, Back navigation, and a completion animation.
- Check container engine readiness, start Docker Desktop on macOS when needed, and show actionable setup errors without tracebacks.
- Build and reuse a prepared Anomx runtime image with Python, Git, Node.js, and analysis tools; upgrade it automatically with the installed CLI while retaining mounted files.

## 0.2.79

- Organize agent storage into shared brain/skills and per-chat task, temporary, and sub-agent directories.
- Migrate legacy CLI homes automatically on startup, with concurrent-launch protection and recovery after interruption.
- Add opt-in `/subagents` with Markdown handoffs and `/goal` with explicit goal completion.
- Add host/container onboarding choices, runtime-aware labels, and responsive line-by-line CLI output.

## 0.2.78

- Document compact, paginated Work history and explicit full Run-detail retrieval in the built-in agent skills.
- Distinguish persisted Job workspace UI state from executable Work definitions and their access requirements.

## 0.2.76

- Add `WorkContext.artifact()` to retain run outputs with SHA-256 checksums and bounded JSON metadata.
- Support host-managed artifact persistence and standalone content-addressed copies, with explicit file and metadata limits.

## 0.2.43

- Forward provider reasoning and OpenAI reasoning summaries as separate thought events.
- Preserve thoughts in subagent work history and retain signed/redacted Anthropic blocks for tool continuation.
- Keep structured reasoning and unfinished thinking blocks out of normal assistant text.
- Start agent turns with a brief text response and provide occasional progress updates during longer work.

## 0.2.8

- **Major codebase restructuring**: Tools and backends moved into dedicated classes for better object-oriented design.
- **Agent UI restructuring**: UI class refactored into a mixin-based architecture; subagent views now include a proper "back" navigation.
- **"@" referencing feature**: Added support for mentioning files and directories via "@" for agent context.
- **Paste support**: Added paste with pre/post handling.
- **Agent steering**: Implemented agent steering capability for better workflow control.
- **Global approvals**: Config now uses global approvals instead of session-scoped ones; improved /config command management.
- **Debug mode upgrade**: Full logs are now stored in `~/.anomx/debug` by default.
- **Removed deprecated shortcuts**: Cleaned up unused deprecated shortcut commands.
- **Various fixes**: Improved automatic restart logic, reconnection handling, and general stability.

## 0.2.6

- Agent sandbox updates: podman/docker modes, copy/mount project file modes, container stop/remove modes.
- Agent UI and store improvements for sub-process handling, session output, and formatting.
- Enhanced agent CLI with better visualization and on-brand UI.
- Various bug fixes and stability improvements.

## 0.1.0

- Initial package scaffold for datasets, scorers, detectors, models, and optional Darts integration.
