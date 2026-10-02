# Agent context management

`maximum_context_tokens` is the only context-size setting. It bounds input
context; a known model window also reserves room for model output. The CLI and
platform usage display use this effective maximum. Old
`context_compression_target_percent` configuration values are ignored and removed
on save. Historical summary records can still contain that field's equivalent
`target_percent`, recording the automatically chosen budget.

## Reduction levels

1. **Tool results:** results above the smaller of 8,192 tokens or one eighth of
   the maximum are candidates for the medium-work model only when the projected
   next request reaches 50% of the effective maximum. A large result alone does
   not trigger optimization in a small context. Projection includes provider
   input/output usage and all results in the current tool batch. Full results are saved
   first. The model can return `KEEP`; only a smaller, nonempty digest is accepted.
   Digests reference the original session event.
2. **Tool blocks:** adjacent completed tool records can be reduced together by
   the medium-work model. User and assistant text form boundaries. Digests replace
   only backend-visible context; original transcript events remain unchanged.
   Block reduction also requires at least 50% utilization (or forced recovery).
3. **History compression:** the easy-work model merges older history into a
   rolling summary injected with the next request. New user prompts are retained
   at turn boundaries. During a tool loop, the summary includes completed tool
   actions before rebuilding the provider request chain.

## Evaluation and headroom

The runtime evaluates opportunities at each user follow-up, after 24 additional
context entries, and when crossing 50%, 65%, 80%, 90%, and 99% utilization.
Follow-ups and message-count checkpoints below 50% are evaluations without an
AI reduction request. Delivering final output never starts another optimizer call.
These deterministic gates bound background work; the optimizer decides which
tool evidence can be reduced without losing necessary detail. Tool blocks need
at least the smaller of 2,048 tokens or one sixteenth of the maximum to qualify.

At 80%, the runtime attempts optimization and then history compression if still
needed. At 99%, or following a provider context-window rejection, reduction is
mandatory before another request. Failed mandatory compression produces a typed
error instead of sending the same oversized request again. Optional failures
retain the original context and retry at a later evaluation threshold.

History compression aims to retain 50–65% of capacity. The target reserves at
least 35% headroom and expands that reserve for recent growth, up to 50%.
At least 48 remaining messages and 50% utilization also trigger history
compression. A summary is committed only when it lowers the estimated context
below the selected budget. This hysteresis creates room for continued work
between reductions.

## Persistence and presentation

`context_optimization` events record backend replacements by stable message ID;
`context_compression` records the summary and transcript boundary. The raw log
is never rewritten. Provider-local tool messages are flattened before resetting
a request chain, so completed tools are not executed again by the runtime.

`RuntimeCallbacks.context_activity` and matching persisted events report running,
completed, or failed activities, their level, and token counts. CLI and platform
render optimization and compression separately from tool groups. In the platform,
context activities stay inside the overall collapsible work section, display only
their label with the Crop02 icon, and do not create another duration header. Repeated
updates to an activity share its ID. Context usage is refreshed after reduction.

Token counts are estimates until provider usage is available. Model selection
uses `background_medium_work_model` for tool evidence and
`background_easy_work_model` for history; the platform supplies equivalent
callbacks using its system integrations. These selections also propagate to
child runtimes.

The platform update includes migration `0061_remove_context_compression_target`.
Deploy the package and platform changes together and apply that migration through
the normal deployment workflow.
