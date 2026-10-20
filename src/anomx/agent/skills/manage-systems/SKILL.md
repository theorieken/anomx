---
name: manage-systems
description: Discover systems and aggregates, build verified hierarchies, assign channels, and maintain evidenced directed influences.
metadata:
    title: Manage Systems
    hidden: true
    system: true
---

# Manage Anomx Systems

Use this skill to discover domain systems, organise a hierarchy, assign measured
channels, and maintain directed influence relationships. `/systems` contains domain
objects; `/system/nodes` and `/system/services` describe compute infrastructure.
Read `retrieve-data` and `use-anomx-api` for discovery and API conventions, and
[system and relationship contracts](references/contracts.md) before a write.
Manual editing and agent editing use the same API and permission checks.

## Distinguish the objects and relationships

`role` and `kind` answer different questions:

- `role=system` identifies a meaningful functional or physical unit with its own
  behaviour. Its measurements may be undiscovered, inaccessible, or absent from
  the current catalog.
- `role=aggregate` identifies an explicitly intended organisational collection
  of systems. Create this role because grouping is intended, never because
  `direct_channel_counts` is zero or a search returned no channels.
- `kind` independently describes `physical`, `logical`, `hybrid`, or `unknown`.
  Do not use kind as a replacement for role.

Existing records default to `role=system`; this default is not evidence that each
record has already been semantically reviewed. Preserve known roles and human
choices unless the user authorises a correction. A controller can remain a system
before its channels are mapped. An explicitly requested collection of controllers
can be an aggregate without changing the roles of its members.

Keep three relationship meanings separate:

| Kind | Source → target | Meaning |
|---|---|---|
| `part_of` | child System → parent System | Containment or membership in the hierarchy |
| `observes` | data Channel → measured System | What the signal measures |
| `influences` | influencing System → affected System | A directed, evidenced or explicitly user-declared influence |

`kind=part_of` forbids self-links and hierarchy cycles. `influences` also forbids
self-links, but feedback cycles between different systems are valid. Do not
invert an influence to match a layout direction or convert it into containment.
An influence is not proven by a name, proximity, shared parent, or correlation.
Record the source and uncertainty of its evidence; an explicit user-declared
relation is a declaration, not an experimentally established causal effect.
Use notes or recommendations for hypotheses that have not been authorised as
relationships. `contains` is a folder relation, not system hierarchy.

## Discover hierarchy, channels, then influences

1. **Bound the task.** Identify the requested system, branch, channel, or explicit
   collection and its writable scope. For recurring discovery, inspect one page
   of `get_background_runs` once and retain the selected channel and continuation
   state. Use catalog requests with explicit limits and stable `ordering=id`;
   vary the selected page/channel according to the task. An empty page ends a
   traversal; an endpoint failure does not mean the catalog is empty.
2. **Reuse the existing hierarchy.** Search `/systems?query=<name>` and, when
   established, the exact `external_ref`. Read `/systems/explore` for roots or
   `/systems/explore?focus=<system-uuid>&offset=0&limit=9` for a branch. Inspect
   returned parents, children, roles and descriptions before creating anything.
   Before correcting an existing hierarchy edge, read
   `/systems/<id>/parents?offset=0&limit=100` for its actual connection ID:
   outgoing edges point to parents, incoming edges come from children.
   A graph view may help investigate context, but its visual layout is not an
   authoritative direction for a relationship.
3. **Create only meaningful missing nodes.** Inspect identifiers, metadata,
   documentation and existing connections. Reuse evidenced ancestors rather
   than making one node for every identifier segment, channel, or recurring
   discovery pass. Use a stable, evidenced `external_ref` when available.
   Choose aggregate only for explicit grouping intent. Keep uncertain identity
   or hierarchy proposals as notes instead of expanding a speculative tree.
4. **Associate channels.** Inspect the channel's identity, description, units
   and existing associations. Read `/systems/<id>/channels?offset=0&limit=100`
   for actual observation-edge IDs: incoming edges run from Channels to this
   System; outgoing is empty. This endpoint covers direct `observes` links,
   not descendant channels or every possible channel relationship.
   Attach `observes` to the system it measures and
   verify the affected branch. A channel may be relevant to several analyses;
   do not delete an existing association merely because another system is
   related. A channel's catalog identity, live value, history and recording
   state remain distinct. Classification does not start acquisition.
5. **Review directed influences.** Read the focus system's incoming and outgoing
   relationships via `GET /systems/<id>/influences?offset=0&limit=100`, following
   its paging flags as needed. Check the concrete source and affected endpoint,
   existing edge, supplied evidence, and permissions. Create or correct only
   the relationship justified by the task. Preserve legitimate feedback paths;
   hierarchy cycle rules do not apply to influences.
6. **Verify and retain progress.** Read the returned object after each mutation
   and reread the affected branch or influence list. Keep the actual connection
   IDs for later authorised corrections. Record reused/created systems, role
   decisions, mapped channels, direction and evidence of influence links,
   unresolved hypotheses, blocked writes, and the next bounded action.

## Write through the shared API

Create a System with `POST /systems`; update it with `PATCH /systems/<id>`.
Use `role` explicitly when its meaning is established. `parent` is a
creation-only convenience field: existing hierarchy changes use actual
`part_of` connections. Both hierarchy and influence writes use
`POST /connections` with `source_object_reference`, `target_object_reference`,
and `kind`; influence labels use the supported `label` field.
Retrieve references rather than inventing them. The server deduplicates an
identical source/target/kind.

Use connection IDs returned by the influence, hierarchy, or channel list for
`PATCH /connections/<connection-id>` and `DELETE /connections/<connection-id>`.
The collection does not support `GET /connections`; do not fabricate a generic
connection-list query. Check the endpoint's `can_write`/`can_delete` flags and
honour server denials. A missing old-platform endpoint or unsupported role is
a capability mismatch: report it, rather than silently reverting to a
different relation or inferring roles from channel counts.
An observation link is controlled by write access to its source Channel;
write access to the measured System alone does not permit changing the link.

Read existing JSON properties before PATCHing; replacing that field replaces
the JSON object. Preserve unrelated fields and `human_overrides`. Do not
overwrite protected human choices based solely on inferred metadata. Save
scientific evidence in supported descriptions/properties or linked notes;
`provenance` and `human_overrides` may be read-only API fields. Use `use-icons`
when available to validate icons and preserve existing valid user choices.

This skill supplies a procedure, not extra permissions. Background tasks must
obey their configured create/update/delete permissions for both Systems and
Connections. Use recommendations for denied writes and distinguish a proposal
from a completed mutation. Follow `manage-jobs` for authorised acquisition
changes and `inspect-platform` for infrastructure inspection. Organising domain
systems does not authorise worker restarts or compute-capacity changes.
