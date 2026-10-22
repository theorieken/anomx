# Data mapping, bounded reads and defensible forecasting

## Inspect once, select before loading

Resolve the Dataset, schema, source kind, target units and timestamp meaning with
bounded API reads. Use `create-dataset` when mapping is missing and inspect only
needed files/columns/array keys. Keep the requested range separate from a small
probe range. Avoid full-schema dumps and repeated source-code inspection after a
contract has been confirmed. Metadata availability is not evidence of readable rows.

For a time-series Dataset, the current package selection API is:

```python
from anomx import Dataset

selected = Dataset(inputs["dataset_reference"], progress=False).select(
    start=inputs["start"], end=inputs["end"]
)
# Only when the task justifies aggregating to these physical time bins:
selected = selected.downsample("10min", aggregation="mean")
probe = selected.to_dataframe(max_rows=100, batch_size=100, progress=False)
```

`select` is lazy, includes `start` and excludes `end`; timezone-free bounds mean
UTC. Explicit bounds replace the corresponding bounds on the view. `downsample`
accepts a fixed frequency and `mean`, `first`, `last`, `min`, `max`, `sum` or `count`.
Time filtering precedes aggregation into UTC-epoch bins. Empty bins are omitted;
no values are filled. Left-edge bin labels can precede the selected start.
Aggregation reduces temporal resolution and may suppress anomalies; preserve its
method/rate in provenance. It cannot substitute for knowing source semantics.

Remote selection/aggregation requires the host's dataset selection capability.
`NotImplementedError` is a capability mismatch; do not silently fetch all history
and filter locally. Confirm the connected host contract or choose a bounded
supported selection explicitly. Native mounted `DCacheDataset` can read a resolved
manifest on compute when that integration supports it; do not invent globs or
mount paths. Use its source selection/version, not a second untracked data path.

`max_rows` caps output after selection/aggregation. It is a prefix, not a random
sample, training split, time interval, or proof that the entire range was read.
For a bounded training procedure, load at most `budget + 1` and fail if oversized,
or deliberately consume all batches within explicit time/byte/row limits.
`iter_batches(batch_size=...)` bounds transferred batches (1–10,000 rows), not the
full memory footprint if code appends all batches. Close an iterator if stopping
early. A tiny API response limit does not guarantee cheap server aggregation over
a huge source. Narrow expensive source windows before lengthening timeouts.

## Resumable cache and version evidence

Caching is built into `Dataset`; do not reimplement paging or invent `.cache()`.
Use constructor `cache=True` (default), a writable path, or `False`. Default
location is `ANOMX_CACHE_DIR` or `~/.anomx/cache/datasets` on the executing machine.
It is not automatically shared between the agent shell and remote compute.
`iter_batches()` and `to_dataframe()` checkpoint pages and resume the same
selection, source version and cursor. A later larger prefix can reuse saved pages.
Inspect `dataset.cache_info` and the final `describe()`/frame selection metadata.

After interrupted transfer, retry the same selection with its cache intact.
Do not add `refresh=True`, clear the cache or recreate temporary environments as a
routine retry: refresh explicitly restarts the download. If a cursor expires,
record that error and deliberately refresh. Completed-cache reuse requires a
matching content token or explicit TTL. `cache_ttl=0` is the default;
`cache_ttl=None` opts into snapshot reuse until refresh/version change. Without a
content validator, version-pinned resumed pages are a local snapshot, not proof
that upstream bytes stayed unchanged. Every new load rechecks metadata/permissions.

Direct platform reads must use the resolved Dataset `version` and returned
cursor; prefer the package client, which handles these contracts. An omitted
version or repeated failed range query is not a reason to bypass the dataset.
A fixed Dataset pins references/configuration/fingerprints, not immutable copies
of all external bytes. Automatic datasets resolve a current selection per run;
preserve the actual resolved version and coverage, not only the dataset name.

## Native series, timing and shape

```python
# After bounding and verifying this selection, on compute:
series = selected.to_timeseries(value_columns=["DETUNING"], progress=False)
```

`to_timeseries()` creates native `anomx.TimeSeries`. It materializes the selection;
unlike `to_dataframe`, it has no `max_rows` argument. Convert an already bounded
frame via `Dataset.from_dataframe(frame, kind="time_series").to_timeseries()`
when enforcing a row budget. Timestamps become a UTC-normalized, timezone-free
index; timezone information remains in metadata. `frequency=...` resamples after
loading, whereas `downsample(...)` pushes aggregation to the host before transfer.
Both change data semantics; neither silently justifies interpolation.

No fill is the default. `fill="ffill"` or `"interpolate"` is an explicit decision;
interpolation can use future observations and must not cross split/segment edges
in causal evaluation. Native forecasters require a regular finite series. Detect
gaps and model separate coherent segments or document a justified missing-data
policy. Do not silently drop missing bins and renumber time. Duplicate quantized
timestamps need real event ordering, not invented subsecond offsets.

Keep `time_series`, ordered `sequence`, and `independent_samples` distinct.
`sample_axis=-1` denotes source rows on the last array dimension, not necessarily
the forecasting axis. Separate event order from intra-waveform samples. Choose
explicit, scientifically meaningful reductions/components for tensor-valued rows;
never average unrelated channels merely because their shapes match.

## Splits and baseline

Split by chronology or verified independent recording/system/subject before
learning normalization, imputation, feature selection or model parameters. Never
random-split overlapping windows. Keep each window within its segment; report row
and forecast/window counts after exclusions. A large tensor can contain few
independent events. Reduce complexity or acquire more data for undersized splits.

Select architecture/hyperparameters/checkpoints on validation and evaluate the
final test after selection. Fit scalers on training only; invert predictions
before reporting physical-unit RMSE/MAE. For each metric, compare the same targets,
horizon, forecast origins and units against a meaningful baseline. Rolling-origin
forecasting may consume already-observed test history; distinguish it from one
open-loop forecast across the entire test span. Never compare normalized training
MSE with physical-unit baseline MSE.

Inside Work, `Dataset(reference)` uses the active context. Outside Work, use the
authenticated `use-anomx-api` helper for bounded inspection or explicitly configure
`PlatformClient` according to that skill. Bare `Dataset(reference)` otherwise
expects `ANOMX_URL`/`ANOMX_TOKEN`. Do not print environments, persist secrets in
inputs, or interpret missing agent-shell connection variables as a compute failure.
