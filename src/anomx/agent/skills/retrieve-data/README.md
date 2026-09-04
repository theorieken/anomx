---
command: retrieve-data
title: Retrieve Data
description: Discover Anomx data sources and retrieve bounded, inspectable channel data.
hidden: true
system: true
---

# Retrieve Anomx Data

Use the dedicated data tools first. `search_anomx_data_channels` discovers live
channels from leading identifier segments and returns continuation hints;
`get_anomx_data_channel_history` retrieves a bounded history window for a concrete
`data_channel-...` reference. Use `use_anomx_api` for endpoints not covered by those
tools.

Useful reads include:

- `GET /data/datasets`
- `GET /data/channels`
- `GET /data/channels/live-search?query=<leading-prefix>&limit=<n>`
- `GET /data/channels/live-hints?query=<leading-prefix>&limit=<n>`
- `GET /data/channels/<object-reference>/value`
- `GET /data/channels/<object-reference>/history?range=1h&max_points=100`

The API tool returns a bounded parsed response directly and writes the complete JSON
payload to the response path it reports. Analyze the returned JSON directly; a missing
shell or Python tool is not a blocker. If the preview is truncated, use `read` on the
reported response path and continue in chunks.

Start with narrow queries and explicit limits. Follow pagination metadata (`count`,
`next`, `previous`, `limit`, and `offset`) rather than assuming the first page is
complete. Preserve timestamps, units, quality/status fields, channel identifiers, and
object references when comparing values. State when data is sampled, aggregated,
missing, stale, or outside the requested time range.
