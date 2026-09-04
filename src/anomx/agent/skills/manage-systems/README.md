---
command: manage-systems
title: Manage Systems
description: Inspect Anomx nodes, services, health, and system jobs with operational care.
hidden: true
system: true
---

# Manage Anomx Systems

Use `/system/health`, `/system/nodes`, `/system/services`, and `/system/jobs` to inspect
the control plane and runtime capacity. Correlate service identity, node, capability,
health, last heartbeat, command or job ID, timestamps, and error details before drawing
conclusions. Distinguish stale telemetry from confirmed failure and report the evidence
for any operational recommendation.

System actions have broad impact. Do not cancel `/system/jobs/<id>/cancel`, change
service configuration, or issue an operational write unless it is explicitly requested
and permitted by the active mode. In Recommend mode, describe such an action in a
`POST /recommendations` proposal instead of applying it.
