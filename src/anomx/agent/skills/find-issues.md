---
command: find-issues
title: Find issues
description: Scan for anomalies, missing data, broken timestamps, duplicates, corrupt files, and schema problems.
hidden: true
---

Inspect the current workspace for data and pipeline quality issues.

Look for:
- Missing, duplicated, malformed, or corrupt records.
- Broken timestamps, non-monotonic series, timezone inconsistencies, and irregular sampling.
- Schema drift, type mismatches, unexpected nulls, and suspicious categorical values.
- Outliers, anomalous segments, gaps, spikes, flatlines, and other unusual patterns.
- Code or configuration that could silently hide data quality problems.

Report concrete findings with file paths, evidence, likely impact, and recommended next checks.
