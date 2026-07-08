---
command: use-anomx-api
title: Use Anomx API
description: Internal platform API instructions for connected Anomx agents.
hidden: true
system: true
---

# Connected Anomx Platform API

This agent is connected to an Anomx Platform instance. Use the `use_anomx_api`
tool or the helper file at `~/.anomx/skills/use-anomx-api/api.py` whenever the
task involves platform data, projects, pages, files, jobs, channels, object
search, system state, or user-owned platform records.

The runtime exports these environment variables when a platform is connected:

- `ANOMX_PLATFORM_API_URL`: Base API URL, usually ending in `/api/v1`.
- `ANOMX_PLATFORM_URL`: Same base URL for compatibility.
- `ANOMX_PLATFORM_API_KEY`: Bearer token for platform API calls.
- `ANOMX_PLATFORM_TOKEN`: Same bearer token for compatibility.
- `ANOMX_API_KEY`: Same bearer token for compatibility.
- `ANOMX_RESPONSES_DIR`: Directory for raw API responses, normally `~/.anomx/responses`.

Prefer the `use_anomx_api` tool for simple calls because it automatically stores
the response body as JSON and returns only metadata plus the response file path.
For custom local scripts, import or execute `api.py`; it reads the environment
variables above and writes raw payloads into the responses directory.

Examples:

```bash
python ~/.anomx/skills/use-anomx-api/api.py GET /objects --query '{"query":"xfel","limit":10}'
python ~/.anomx/skills/use-anomx-api/api.py GET /data/channels --query '{"search":"temperature"}'
python ~/.anomx/skills/use-anomx-api/api.py POST /folders --body '{"name":"Analysis","description":""}'
```

The helper output is intentionally short: HTTP status, response length,
detected result count, and the JSON file path. Read that file when the payload
matters.

Important platform endpoints:

- `GET /objects`: Unified object search. Useful query params include `query`,
  `model_reference`, `limit`, and `offset`.
- `GET /account`, `GET/PATCH/DELETE /account/profile`,
  `PATCH /account/preferences`, `PATCH /account/password`:
  user account and preferences.
- `GET/PATCH /account/organization`, `GET /account/workspace`,
  `GET/POST /account/tokens`: account organization, workspace summary, and API
  tokens.
- `GET /account/trash`, `POST /account/trash/<resource_type>/<object_id>/restore`:
  trash.
- `GET/POST /folders`, `GET/PATCH/DELETE /folders/<id>`: folders/projects.
- `GET/POST /pages`, `GET/PATCH/DELETE /pages/<id>`: pages and dashboards.
- `POST /pages/<id>/update-component-positions`: reorder page components.
- `GET/POST /files`, `GET/PATCH/DELETE /files/<id>`, `POST /files/upload`:
  files and uploads.
- `GET/POST /integrations`, `GET /integrations/connector-catalog`:
  integrations and connector metadata.
- `GET /data/datasets`, `GET /data/channels`, `GET /data/channels/overview`,
  `GET /data/channels/live-hints`, `GET /data/channels/live-search`:
  data catalog and live channel discovery.
- `GET /data/channels/<id>/history`, `GET /data/channels/<id>/value`:
  channel time series and latest value.
- `GET /jobs/jobs/build-options`, `GET/POST /jobs/jobs`,
  `GET/PATCH/DELETE /jobs/jobs/<id>`: job configuration and orchestration
  objects.
- `POST /jobs/jobs/<id>/run`, `POST /jobs/jobs/<id>/stop`,
  `POST /jobs/jobs/<id>/archive`, `POST /jobs/jobs/<id>/restore`: job actions.
- `GET /jobs/models/featured`, `GET /jobs/models`, `GET /jobs/algorithms`,
  `GET /jobs/scorers`, `GET /jobs/detectors`, `GET /jobs/components`:
  component catalogs.
- `GET /jobs/findings`, `GET /jobs/model-artifacts`, `GET /jobs/job-runs`:
  run outputs.
- `GET /agents/chats`, `GET /agents/turns`, `GET /agents/runs`,
  `GET/PATCH /agents/settings/me`: agent state.
- `POST /agents/turns/<id>/approval`, `POST /agents/turns/<id>/question`:
  human-in-the-loop agent responses.
- `GET /system/health`, `GET /system/nodes`, `GET /system/services`,
  `GET /system/jobs`, `POST /system/jobs/<id>/cancel`: operator system state.
- `GET /openapi.json`: full OpenAPI schema for exact request and response shapes.

When creating or updating records, inspect the relevant schema first through
`/openapi.json` or by retrieving a similar object. Keep writes scoped to the
user request and report the response file path in your final summary when it
contains important details.
