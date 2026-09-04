---
command: manage-jobs
title: Manage Jobs
description: Inspect, configure, run, and diagnose Anomx jobs through auditable platform APIs.
hidden: true
system: true
---

# Manage Anomx Jobs

Read `GET /jobs/jobs/build-options` and the relevant component catalogs before
creating or restructuring a job. Useful catalogs include `/jobs/models`,
`/jobs/algorithms`, `/jobs/scorers`, `/jobs/detectors`, and `/jobs/components`.
Retrieve the current job before patching it and preserve bindings or configuration not
named by the user.

Use `POST /jobs/jobs/<id>/run` and `/stop` only when the active mode and the user's
request authorize execution. Diagnose outcomes through `/jobs/job-runs`,
`/jobs/findings`, and `/jobs/model-artifacts`; retain run IDs, component versions,
timestamps, configuration snapshots, metrics, and errors in conclusions. After any
write or action, retrieve the job or run again and verify the resulting state.
