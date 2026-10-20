# System and relationship contracts

Paths are relative to the connected platform API base. Inspect its current
schema or a retrieved object if a required field/endpoint is missing. These are
the same domain operations used by the manual system editor.

## System role is independent of kind

`POST /systems` and `PATCH /systems/<system-id>` accept:

- `role`: `system` or `aggregate`, default `system`.
- `kind`: `unknown`, `physical`, `logical`, or `hybrid`.
- Supported descriptive fields including `name`, `description`,
  `classification`, `external_ref`, `tags`, `properties`, and `spatial`.

Example of an **explicitly requested** collection:

```json
{
  "name": "Timing controllers",
  "role": "aggregate",
  "kind": "logical",
  "description": "Organisational collection of the documented timing controllers."
}
```

Do not copy this example as a real system name or invent an external reference.
A zero channel count does not imply aggregate. Current read permissions,
discovery coverage, and incomplete channel mapping can all produce zero.
System detail, explorer and graph responses expose role; graph metadata exposes
direct channel counts. Count only what the response actually covers.

`parent` is accepted only when creating a System. For existing Systems use
`part_of` Connections; names and JSON properties do not create hierarchy.
Read `GET /systems/<system-id>/parents?offset=0&limit=100` to retrieve actual
hierarchy connection IDs before an authorised correction. Its `outgoing`
edges go from the selected child to parents; `incoming` edges come from child
Systems to the selected parent. The endpoint returns both directions despite
its name, using the same paging and permission fields as the influence endpoint.

## Relationship creation and direction

`POST /connections` accepts object references returned by the API:

```json
{
  "kind": "influences",
  "source_object_reference": "systems_system-<influencing-system-id>",
  "target_object_reference": "systems_system-<affected-system-id>",
  "label": "Short description of the evidenced influence"
}
```

Both influence endpoints must be Systems, with no self-link. Feedback cycles
between distinct Systems are allowed. Source is the influencing system and
target the affected system. Do not infer causality from correlation or graph
layout. Record the actual evidence or the user's explicit declaration in a
supported note/description; a label is not a proof.

For hierarchy, use `kind=part_of` with child as source and parent as target;
self-links and hierarchy cycles are invalid. For measurements, use
`kind=observes` with `data_channel-<id>` as source and the measured System as
target. Reusing the same source, target and kind is deduplicated by the server.

`GET /systems/<system-id>/channels?offset=0&limit=100` lists direct `observes`
connections using the paging fields below. `incoming` contains Channel → System
edges; `outgoing` is empty. Sources contain Channel records and targets contain
System records, both with object references and permissions. This is not a
recursive descendant-channel listing. Read this endpoint before changing an
existing observation so that you retain its actual connection ID.
Channel write access controls `can_write` and `can_delete` for these links;
System write access alone does not grant it.

## Read, edit and delete influences

`GET /systems/<system-id>/influences?offset=0&limit=100` returns `incoming`,
`outgoing`, `incoming_total`, `outgoing_total`, `incoming_has_more`,
`outgoing_has_more`, `offset`, and `limit`. The maximum limit is 200.
Advance by the returned `limit` while either direction has more records;
do not assume one short direction means both directions are exhausted.

An influence edge contains `id`, `kind`, `label`, `source`, `target`,
`can_write`, and `can_delete`. Endpoint objects carry their `_anomx` references
and permissions. Incoming means the selected System is the target; outgoing
means it is the source. Preserve the connection ID, not just the endpoint IDs.

- `PATCH /connections/<connection-id>` changes supported fields, such as
  `label` or a specifically authorised endpoint correction.
- `DELETE /connections/<connection-id>` removes that particular relation.
- Reread the affected influence list to verify the result.

Creation requires accessible endpoints and write access to the source;
mutation rights are also reflected by the returned edge permission flags.
Do not assume write access to the affected system permits editing every
incoming influence. Background runtime policy may impose additional
restrictions. A successful proposal is not a successful relationship change.

There is no generic `GET /connections` collection/detail read. Use the
System influence, parent, and channel endpoints for edge IDs, `/systems/explore` for
hierarchy and channel verification, and actual IDs retained from authorised writes.
Graph responses may include typed non-layout influence edges and
`influence_context_ids`; these are relationship context rather than additional
hierarchy parents or proof of complete catalog coverage.
