# Public API

| Domain | Public entry points |
| --- | --- |
| Shared contracts | `anomx.base.BaseComponent`, capabilities, `ModelSignature`, `DataStructure`, `iter_component_classes` |
| Data | `anomx.data.Dataset`, `TimeSeries`, `TimeSeriesDataset`, `ObservationSet`, `MountedDataset`, `DCacheDataset` |
| Data access | `anomx.data.base.DatasetClient`, `BaseConnector`, `use_dataset_client`; implementations in `anomx.data.sources` |
| Views | `anomx.data.views.TabularView`, `SequenceView`, `WindowView`, `TimeSeriesView`, `GraphView`, `TimeSeriesBatch` |
| Processing | `anomx.data.processing.Transform`, `ProcessingPipeline`, `Resample`, native `Pipeline`, `transformers`, `encoders`, `dtw` |
| Models | `anomx.models.Model`, `NormalityModel`, native forecasting/reconstruction/representation models, `PyTorchModel`, `ONNXModel` |
| Metrics | `anomx.metrics` point, probabilistic, scaled, interval and anomaly metrics |
| Model outputs | `anomx.models.InferenceOutput`, `InferenceKind` |
| Detection | `anomx.detection.Scorer`, `Detector`, `AnomalyPipeline`, `AnomalyScores`, `AnomalyEvents` |
| Scoring | `anomx.detection.scorers.AbsoluteErrorScorer`, `ZScoreScorer` |
| Decisions | `anomx.detection.detectors.ThresholdDetector`, `QuantileThresholdDetector` |
| Execution | `anomx.execution.LocalExecutor`, `LocalResult`, `JobDefinition`, `JobRunner`, `Run`, `init`, `log`, `current_run`, `WorkContext`, `WorkGraph` |
| Run persistence | `anomx.execution.base.RunBackend`, `anomx.execution.backends.JsonlBackend`, `CallbackBackend` |
| Platform integration | `anomx.integrations.platform.PlatformClient`, `anomx.integrations.platform.tracking.PlatformBackend` |
| Darts integration | `anomx.integrations.darts.Dataset`, `DartsModel`, `DartsForecastingModel`; Darts-native scorers/detectors in its submodules |

`anomx.Dataset`, `anomx.Model`, `anomx.Run`, `anomx.init`, and `anomx.log` are
convenience exports of those same objects. The generic `Model` requires inference
only. `NormalityModel` adds fitting/persistence, and `DartsModel` provides the
optional Darts forecasting contract. Use explicit `.to_darts()` conversion when
passing the generic dataset to Darts libraries.

The library discovers component classes through `iter_component_classes`.
The platform owns `discover_component_payloads` in
`anomx_platform.helpers.utils.component_catalog`, which serializes those classes
for its catalog and UI.

## Platform HTTP client

```python
import os
from getpass import getpass

from anomx.data import Dataset
from anomx.integrations.platform import PlatformClient

client = PlatformClient(
    url="https://anomalies.msktools.desy.de",
    token=os.environ.get("ANOMX_TOKEN") or getpass("Anomx API token: "),
)
print(client.is_alive())
dataset = Dataset(anomx_reference="data_dataset-<uuid>", client=client)
print(dataset.describe())
```

`PlatformClient` owns authentication, URL construction, timeouts, JSON encoding,
and HTTP errors. It exposes `get`, `post`, `put`, `patch`, `delete`, `head`, and
the general `request(path, payload=None, method="POST", params=None)` method.
For example, `client.get("/datasets", params={"page": 2})` reads a resource, and
`client.post(path, payload)` submits JSON. Query parameters are URL-encoded; list
values repeat the key. JSON responses are decoded, and empty responses return
`None`. Requests do not follow redirects or retry mutations automatically.

A site origin is normalized to `/api`. An explicit API URL such as `/api/v1`
or a reverse-proxy prefix is preserved. All request paths start with `/` and
are appended to this API base. Resource callers own any route differences:
for example, datasets use `/datasets` under `/api` and `/data/datasets` under
`/api/v1`. `from_env()` reads `ANOMX_URL` and `ANOMX_TOKEN` with the same rules.

`is_alive()` checks `GET /system/health` for `status: ok` and returns `False` on
HTTP, connection, timeout, or invalid-response failures. Set `health_path=` for
a different health route. Liveness does not establish access to a dataset.
Other requests raise `PlatformError`, exposing `status_code`, `method`, and the
requested `url`. Error messages include the method and URL without query values,
credentials, or response bodies; the `url` attribute retains the query for debugging.
Socket timeouts raise the more specific `PlatformTimeoutError` (a `PlatformError`)
and report the configured limit. Set `timeout=60` on the client when a slower
service needs more time; this does not change server-side pagination or limits.

Dataset routes and payloads live in `anomx.data.sources.PlatformDatasetClient`,
which adapts the generic HTTP client to the data domain's `DatasetClient`
contract. `Dataset` accepts the transport directly and selects the adapter.
It owns schema validation, version pinning, pagination, and DataFrame conversion.
Work callbacks still implement the same data contract without HTTP. Run and
workspace endpoints likewise stay in their respective integration classes.
`PlatformClient` has no `describe_dataset`, `read_dataset`, or `load_channels`
methods; call the relevant dataset class instead.

See [Architecture](architecture.md) for dependency rules and the migration map,
[datasets](datasets.md) for versioned access, and [runs](runs.md) for tracking.
