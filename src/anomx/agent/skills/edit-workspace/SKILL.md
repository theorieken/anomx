---
name: edit-workspace
description: Mount, edit and save the multi-file Python workspace of an Anomx job component. Use for AI-assisted coding, adding modules or folders, debugging work code and changing a component entrypoint.
metadata:
    title: Edit a job workspace
    keywords: IDE, files, folders, Python, job, component, code, mount
---

# Edit a job component workspace

Use the focused `jobs_job_component-...` object when available. For a focused job,
read its `works` and choose the component matching the user's request. DAQ has
configuration; only compute components have executable workspaces.

Use the terminal with the runtime's existing `ANOMX_URL` and `ANOMX_TOKEN`:

```bash
python -m anomx.integrations.workspaces mount jobs_job_component-UUID work/component
```

This creates a local working copy, not an operating-system mount. Inspect the
files, then use ordinary file editing tools. Files and folders are relative to
the workspace. The `.anomx-workspace.json` manifest records the component,
entrypoint and base revision; keep it and never reset the revision to bypass a
conflict. Change its `entrypoint` only when the user needs a different Python
entrypoint. It must name an existing `.py` file.

Save the reviewed changes back into the IDE:

```bash
python -m anomx.integrations.workspaces save work/component
```

Saving uses the complete file tree; removing a local file removes it remotely.
An HTTP 409 means another edit won: mount into a second directory, compare and
merge the user's changes, then save from that newer checkout. Do not overwrite
or discard either version automatically. On other API errors, report the error
and keep the working copy. An active component must be stopped before editing;
ask before stopping a running acquisition or another component outside scope.

The worker snapshots all files for each execution. Local modules are importable;
relative data-file paths should be based on `Path(__file__).parent`. `work`,
`inputs`, `context` and `result` are available in the entrypoint. Imported modules
can use `anomx.log(...)` to write to the active project/run, with
`anomx.log({"loss": loss, "accuracy": accuracy}, step=epoch)` for one shared step.
Read `write-work` for compute dispatch and package APIs.

Opening the assistant or saving files does not authorize running them. Start a
run only when requested; verify its persisted status and output before claiming
successful execution. A successful workspace save is confirmed by the returned
revision, not by local file edits alone.
