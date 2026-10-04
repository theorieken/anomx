# Agent context management

`maximum_context_tokens` bounds input context. A known model window also reserves
space for output. The CLI and platform display that effective maximum. Legacy
`context_compression_target_percent` settings are ignored.

## Two reduction levels

1. **Tool-result reduction** uses the easy-work model on one result at a time.
   Calls, arguments, call IDs, message roles and result ordering remain intact.
   The model receives the original and recent user requests, a bounded work
   outline, prior working memory, and the selected call's arguments and result.
   JSON remains JSON: objects retain their keys and types, lists retain selected
   items in source order, and scalar evidence is copied exactly. Long text fields
   can retain original lines. Invalid, empty or insufficiently smaller reductions
   are rejected; `KEEP` retains the original. Results never become assistant
   prose or a synthetic summary envelope. Large arrays are processed in valid
   JSON envelopes and reassembled without changing the outer structure.
2. **History summarization** starts at 48 context messages, independently of
   token utilization. It replaces an old prefix with first-person working memory
   while retaining at least 24 recent messages. The boundary must be durable and
   must not split an assistant's tool-call/result group. Each summary incorporates
   the previous summary and new history, retaining goals, decisions, verified
   outcomes, references, constraints and unfinished work. It must not imitate
   tool-call syntax. The summary is a separate system message (or a separate
   system text block for Messages APIs), explicitly labelled historical memory.

## Evaluation and limits

A new user message, 24 additional entries, or crossing 50%, 65%, 80%, 90% or 99%
utilization triggers evaluation. A follow-up can reduce older results of at least
2,048 tokens even below 50% utilization. Under pressure, results of at least
2,048 tokens are candidates. Immediately returned results qualify at 32,768
tokens regardless of utilization, or above the smaller of 8,192 tokens and one
eighth of capacity when projected utilization reaches 50%. Projection includes
provider input/output usage and the current batch's results.

The optimizer targets at most one third of a historical result, capped at 4,096
tokens. Accepted reductions save at least 25%. Model context budgets and request
counts bound background work. Final output never starts an optimizer request.

Token pressure alone does not turn a short conversation into a history summary.
At 99%, or after a provider rejects its context, a safe reduction is required
before retrying. If tool reduction and eligible history summarization cannot
make room, the runtime returns a typed failure and preserves the evidence.
It never silently drops recent messages or repeats an oversized request.

## Persistence and provider replay

Full results are saved before optimization. Version 2 `context_optimization`
events replace only individual result bodies by stable storage ID. They never
merge or delete results. Legacy merged-digest replacements are ignored so their
original tool evidence can be replayed safely. Existing history-summary boundaries
remain readable. The raw session log is not rewritten.

All backends rebuild native calls and results: Responses function calls/outputs,
Chat Completions assistant calls/tool messages, Messages tool-use/result blocks,
and Ollama calls/tool messages. Provider reasoning is retained through in-turn
rebuilds where available. Local storage and optimization metadata are excluded
from provider wire messages. New persisted executions retain provider call IDs;
old executions receive stable replay IDs. A context reset does not execute tools.

## Activity display

One activity ID moves from running `check` to running optimization and then a
terminal result. The platform updates the same turn in place. Completed checks
and attempts without an accepted reduction disappear before tool-group creation,
so they do not leave a row or split adjacent tool groups. Accepted optimization
remains as “Context optimized”; subsequent tools form the next group. Errors
remain visible. The CLI also omits completed no-op checks.

The platform and CLI both use the easy-work model for these operations. Package
and platform changes must be released together; a platform deployment must update
its pinned/vendored package. This rework requires no database migration.
