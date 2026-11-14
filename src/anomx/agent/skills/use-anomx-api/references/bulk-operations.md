# Bounded bulk operations

Inspect the connected platform's `/openapi.json` before relying on a bulk route.
The installed skill can be newer than the server. There is no universal bulk
endpoint for every object type. Never invent `/bulk-create`, send arrays to a
single-object endpoint, or bypass the API with SQL.

## Connections

`POST /connections/bulk` accepts one operation and 1–200 items. Each request is
atomic: if an item fails, none of that request's changes are committed. Error
`index` values are zero-based when a particular item can be identified. Success
returns `operation`, `count`, and ordered `items` with actual connection IDs,
source/target references, kind, and state. These are compact receipts rather than
expanded objects or a complete catalog count.

Create uses the same fields as `POST /connections`:

```json
{
  "operation": "create",
  "items": [
    {
      "kind": "observes",
      "source_object_reference": "data_channel-<verified-channel-uuid>",
      "target_object_reference": "systems_system-<verified-system-uuid>"
    }
  ]
}
```

Plain Channel → System observations containing exactly those three fields use
set-based permission checks and insertion. The natural key is source/target/kind;
duplicates and retries reuse the same edge and leave existing labels unchanged.
`state=present` means the edge exists after the request, not that it was newly
created. Use this shape for large verified channel mappings. Labels or other
fields use normal per-item validation. Bulk create never changes an existing
edge: if supplied fields differ, use the explicit update operation instead.

Update items contain `id` and the desired PATCH fields. Delete items contain only
`id`. Obtain real IDs from mutation receipts or the System `channels`, `parents`,
and `influences` endpoints; there is no `GET /connections` collection. Each update
or deletion must be writable, and IDs cannot repeat within a batch. Source-write,
sharing, hierarchy-cycle, and immutable provenance rules still apply. Background
permissions check the requested create/update/delete operation independently.

```json
{"operation":"update","items":[{"id":"<connection-uuid>","label":"Verified label"}]}
```

```json
{"operation":"delete","items":[{"id":"<connection-uuid>"}]}
```

There is no dry-run flag. Validate intended relationships from read-only evidence
before submitting. A lost response is an unknown outcome: simple observations
are retry-safe; reconcile update/delete results before repeating them. A batch
that mixes valid and inaccessible IDs fails as a whole, without revealing which
inaccessible records exist.

## Large mapping workflow

1. Establish the requested scope and direction using `manage-systems`. Reuse the
   existing catalog and verified systems. Do not launch live rediscovery for an
   already-indexed catalog: discovering new channels can schedule analysis work.
2. Save a deterministic mapping manifest with channel reference, target reference,
   mapping evidence and status. Deduplicate pairs. A naming prefix alone is not
   proof unless its meaning has been verified for the selected facility.
3. Submit one small batch, inspect its status and receipts, then read a sample
   back from the relevant System endpoint. Continue in batches of at most 200,
   with one request in flight initially. Keep checkpoints outside the prompt.
4. Increase concurrency only when measured throughput and server health justify
   it. On 429/503, honor `Retry-After`, reduce load and use bounded retries. On
   validation/permission failures, repair the manifest or report the blocker;
   do not repeatedly submit the same failed batch.
5. Report confirmed mappings, reused edges, unresolved items, and the checkpoint.
   Estimate remaining time from representative successful batches, separating
   API latency from queue backlog. If bulk is absent, report that capability
   limit instead of silently starting a multi-day single-POST fallback.
