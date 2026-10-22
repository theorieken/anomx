# Local experiments

The scientific core runs without an Anomx platform or Darts installation.
`TimeSeries` owns regular temporal observations; `Dataset` owns local tables or
remote source references. Models, transforms, metrics, and detection components
operate on these containers, while `LocalExecutor` handles local evaluation and
portable result artifacts.

```bash
python -m pip install -e '.[visualization]'
export ANOMX_TOKEN='your token'
python tmp.py
```

The repository's `tmp.py` is one deliberately editable example file using the
real Detuning October 2026 platform dataset. It compares persistence, 60-observation
seasonality, and fitted autoregression, then demonstrates PCA reconstruction,
training-calibrated thresholding, and two-dimensional representations. Add
`--include-torch` with `anomx[torch]` installed to train a small neural forecaster.
`--show` opens Matplotlib figures; all configurations save PNG, CSV and JSON under
`artifacts/detuning-local/` by default. Nothing is uploaded to the platform.

The example loads all rows in the selected dataset or time window by default,
potentially millions of raw samples. It resamples to one-second means and selects
the longest complete interval. It reports how many bins were retained, leaves
the raw data unchanged, and never fills outages silently. Use `--max-rows N` for
an explicit preview limit and `--frequency 10min` for a different resolution.
The baseline seasonal period is a chosen parameter, not an inferred physical period. Forecasts use a chronological test
suffix. Scaling and model fitting use training data only. Reconstruction windows
are created separately on either side of the split, so windows cannot overlap
across the train/test boundary. Threshold exceedances are model outputs, not
independently verified physical anomaly labels.

## Dataset loading and caching

```python
from contextlib import closing
from anomx import Dataset, PlatformClient

client = PlatformClient.from_env()
dataset = Dataset(
    anomx_reference="data_dataset-...",
    client=client,
    cache_ttl=None,  # explicitly reuse the local snapshot until refresh
)
frame = dataset.to_dataframe(max_rows=50_000)
# A later larger read reuses saved pages and continues at the saved cursor.
more = dataset.to_dataframe(max_rows=100_000)
with closing(dataset.iter_batches(5000)) as batches:
    for batch in batches:
        process(batch)
        # Stopping here still retains the entire received page on disk.
```

Each API page is atomically saved **before** it is yielded, together with its
source identity, checksum and next cursor. Restarting Python or interrupting a
download keeps all committed pages. The next call replays those pages and then
continues downloading. Changing `max_rows` or batch size does not create a second
copy. A prefix ending inside a received page retains its unused tail. Empty API
pages that advance the cursor are also checkpointed. Only `next_cursor=None`
marks completion; a small or empty page does not imply the end of the source.

The default directory is `~/.anomx/cache/datasets`. Each source/query has a
`<hash>.parts/` directory containing a small manifest and numbered JSON pages.
This avoids rewriting a huge table after the download. Existing completed JSON
table caches remain readable. No pickle is loaded. Files have owner-only
permissions and are scoped by source, selection, API origin and authorization
hash; the token itself is never saved. Concurrent readers of the same journal
receive a clear lock conflict rather than overwriting each other's progress.
Always close an iterator that stops early, as in the example above.

Set `ANOMX_CACHE_DIR`, pass `cache=Path(...)`, or disable persistence with
`cache=False`. `clear_cache()` removes the selection's saved pages;
`to_dataframe(refresh=True)` or `iter_batches(refresh=True)` starts again.
`dataset.cache_info` reports `hit`, `miss`, `resumed`, or `disabled`, plus
`cached_rows`, `downloaded_rows`, and the journal path. Corrupt pages discard
only the damaged suffix, preserving the preceding valid pages. A checkpoint I/O
failure stops an active transfer immediately; after fixing the storage problem,
re-run to resume the saved prefix.

Every load checks current source metadata and permissions before replaying any
page. A changed source version or content token invalidates the saved download.
Channels currently lack a reliable content revision: backfills and corrections
may leave their selection version unchanged. There are three **completed-cache**
policies when no content token is available:

- `cache_ttl=0` (library default): re-read completed downloads.
- `cache_ttl=300`: allow five minutes of stale reuse, measured from the first
  page, not from completion.
- `cache_ttl=None`: explicitly reuse a version-pinned local snapshot until
  refresh or a source version change. This is the example script's default.

Matching **incomplete downloads** resume regardless of TTL. Without a content
validator this cannot guarantee a consistent point-in-time view of a mutable
upstream source. Use `refresh=True` when you want to incorporate corrections.
Older servers expire cursors after 24 hours; a rejected cursor preserves saved
pages and requires an explicit refresh to restart. The companion backend change
removes that fixed expiry while retaining source, signature and access checks;
re-resolving the same source version makes a later resume possible. Source resolution must
succeed even for snapshot reuse; this is not an offline authorization bypass.
For intentional offline use, `--snapshot FILE` loads a local Parquet snapshot.

### Time windows and server aggregation

```python
window = dataset.select(
    start="2026-10-01T00:00:00Z",
    end="2026-10-01T06:00:00Z",
)
frame = window.to_dataframe()
minute_means = window.downsample("1min", aggregation="mean").to_dataframe()
# Output rows, after filtering and aggregation:
preview = window.downsample("10s").to_dataframe(max_rows=1000)
```

These methods return lazy independent views; they leave the original dataset
unchanged. Bounds are `[start, end)` and timezone-free inputs mean UTC. Filtering
happens before aggregation. Supported reducers are `mean`, `first`, `last`,
`min`, `max`, `sum`, and `count`, over fixed intervals of at least 10 ms. Bins
are anchored to the UTC epoch and labelled at their left edge; the first label
may precede `start`. Empty bins are omitted, absent channel values are missing,
and no values are forward-filled. Different queries have separate journals.

The new platform API supports raw channel datasets. Recipes with a preconfigured
resampling frequency and file-backed datasets reject query selections explicitly.
Filtering and aggregation run on the server, reducing transferred observations;
bounded source scanning still takes time. The SDK checks capability confirmation
so an older host cannot silently ignore query parameters. **The companion platform changes need deployment before these
remote methods work.** Local `Dataset.from_dataframe(...).select(...).downsample(...)`
works immediately without the platform.

The example supports `--start`, `--end`, and `--server-downsample`. For example,
after deploying the platform extension:

```bash
python tmp.py --start 2026-10-01T00:00:00Z --end 2026-10-01T06:00:00Z \
    --server-downsample --frequency 10s --all-rows
```

Without `--server-downsample`, `--frequency` remains local resampling after
loading the selected raw observations. `to_timeseries(frequency="10min")` also
resamples locally; it does not request server aggregation. It creates a regular
TimeSeries and leaves missing bins as NaN unless `fill="ffill"` or
`fill="interpolate"` is explicitly chosen. Aware timestamps normalize to UTC,
numeric timestamps honor the host's `time_unit`, and native regular TimeSeries
indices use timezone-free UTC with `source_timezone` retained in metadata.

## Local execution

```python
from anomx import LocalExecutor, LinearRegressionModel
from anomx.data.processing.transformers import Scaler

result = LocalExecutor().forecast(
    LinearRegressionModel(lags=24, ridge=1.0),
    series,
    horizon=144,
    transformer=Scaler(),
)
print(result.metrics)
axes = result.plot(history=100)
result.save("artifacts/forecast", plot=True)
```

`LocalResult` retains training/test observations, predictions, metrics, timing,
and split metadata. `save` writes observations, predictions and a JSON receipt;
model serialization remains an explicit model-level decision. `verbose=False`
silences executor receipts and `progress=False` silences dataset tqdm output.
`fit_predict` handles reconstruction and representation models on explicit
training/test arrays or DataFrames. It does not equate embedding coordinates
with reconstruction errors.

## Extending the baseline

`GenericModel` accepts estimators, callables, and optional `torch.nn.Module`
objects, with explicit `fit_fn`/`predict_fn` adapters. `GenericScorer` and
`GenericDetector` adapt custom score and decision implementations. See
[models and detection](models-and-detection.md) for signatures and examples.
`Mapper`/`InvertibleMapper` provide callable transforms; fittable and invertible
base contracts are available in `data.processing.transformers`.

Forecasting, reconstruction, and representation remain distinct model families.
`DatasetKind` already distinguishes `time_series`, `sequence`, and
`independent_samples`; `SequenceView`, `TabularView`, `ObservationSet`, and
`ModelSignature` describe shape/ordering/capabilities. They are the extension
points for future sampled and generic sequence containers. This baseline does
not pretend that regular temporal resampling or forecasting is meaningful for
those future types.

The native TimeSeries, processing, metrics and shared utilities were adapted
from Darts 0.48.1, with source commit and Apache license retained in
`THIRD_PARTY_NOTICES.md`. Optional `integrations.darts` remains a separate bridge
for existing external models; native operations do not import it.
