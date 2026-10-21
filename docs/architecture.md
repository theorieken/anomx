# Package architecture

Anomx supplies reusable Python building blocks for anomaly-detection work, usable
from a local script without an agent or platform deployment.

```text
anomx/
  agent/                 CLI agent and its independent application runtime
  base/                  component metadata, capabilities, signatures, discovery
  data/
    base/                source/client, transform, and observation contracts
    dataset.py           local or versioned remote Dataset facade
    sources/             filesystem, mounted/dCache, DAQ, and platform dataset access
    views/               observation sets and structured materializations
    processing/          preprocessing composition and reusable transforms
  models/
    base/                inference, normality-training, and classifier contracts
    forecasting/         forecasting implementations
    reconstruction/      reconstruction implementations
    representation/      representation implementations
    results.py           typed model outputs
    torch.py             optional PyTorch forecasting wrapper
    onnx.py              optional ONNX inference
  detection/
    base/                score and decision contracts
    scorers/             continuous anomaly scores
    detectors/           thresholds and decisions
    pipeline.py          model -> scorer -> detector composition
    results.py           scores, decisions, and events
  execution/
    base/                execution and event-persistence contracts
    jobs.py              portable node definitions
    runner.py            synchronous job execution
    runs.py              lifecycle, metrics, and active-run binding
    backends/            local JSONL and callback event sinks
    work/                host callbacks, graphs, and artifact helpers
  integrations/
    darts/               Darts-native objects and adapters
    platform/            optional HTTP access and event transport
```

## Dependencies and responsibilities

- Data code depends on data contracts and explicit clients; it does not import
  models, detection, execution, or the agent. A scoped `DatasetClient` lets a
  host supply access without reversing that dependency.
- `PlatformClient` is a generic authenticated JSON transport with HTTP verbs and
  a health check. Dataset endpoints live in `data.sources.PlatformDatasetClient`;
  the Dataset facade selects that adapter for an HTTP client. The same data
  contract supports Work callbacks. Run and workspace endpoint behavior stays
  with their consumers rather than accumulating on the HTTP client.
- Models consume data and produce outputs. The generic `Model` needs only
  `predict`; trainable `NormalityModel` adds training and persistence. Capability
  mixins describe forecasting, reconstruction, representation, and other outputs.
- Detection composes data, model outputs, scoring, and decisions. Native scorers
  consume frames; Darts scorers consume Darts objects and remain explicit adapters.
- Execution composes reusable work and tracks its lifecycle. Scheduling,
  distributed dispatch, authorization, database persistence, and catalog/UI
  payloads belong to `anomx-platform`.
- Integrations adapt third-party contracts. Scientific core imports and native
  discovery do not load Darts, Torch, ONNX, Django, or the agent. Install the
  appropriate optional extra when using an integration.
- The agent remains a separate application within the distribution. The
  scientific package does not depend on its runtime.

`base/` consistently holds contracts within each domain. `BaseComponent` carries
shared metadata for discoverable algorithms, while datasets, clients, transforms,
and runs use contracts appropriate to their own behavior. No universal object
superclass is required.

`WorkGraph` and `JobDefinition` remain distinct portable graph formats. Their node
types and execution semantics differ; grouping them under execution does not
silently convert one into the other. `WorkContext.dataset()` currently retains
its Darts channel helper; generic dataset work uses `Dataset` directly.

## Processing

```python
from anomx.data.processing import ProcessingPipeline, Resample

# Materialization is explicit; resampling requires complete time intervals.
pipeline = ProcessingPipeline([Resample("1s", method="mean")])
processed = pipeline.fit_transform(training_frame)
validation_processed = pipeline.transform(validation_frame)
```

Transforms receive DataFrames and preserve their index and metadata where the
operation permits. `fit` learns from training observations only; inference calls
`transform` without refitting. `ProcessingPipeline.transform_batches` delegates
to each step's batch method, so temporal transforms can carry state between
batches. It never loads a whole dataset implicitly. `Resample` rejects batch
iteration because aggregating partial time bins independently would be wrong.
The existing `TimeSeriesDataset.resample()` delegates to the same implementation.

## Development migration

This is a source-level development migration. Old domain directories are removed;
there are no compatibility wrappers or saved-job/database migrations.

| Previous location | Current location |
| --- | --- |
| `components.base` shared metadata | `base` |
| `components.base` model/scorer/detector contracts | `models.base`, `detection.base` |
| `components.models` | `models` |
| `components.detection`, `components.pipeline` | `detection` |
| `components.algorithms`, `runner`, `runs`, `work` | `execution` and `execution.work` |
| `datasets` Darts subclass | `integrations.darts.datasets` |
| `scorers`, `detectors` Darts subclasses | `integrations.darts.scorers`, `integrations.darts.detectors` |
| `models.Model` Darts base | `integrations.darts.DartsModel`; `models.Model` is now generic |
| `data.connectors`, `data.mounted`, `data.daq` | `data.sources`; connector base in `data.base` |
| `data.base.views`, `data.base.observation_set`, `data.sequences` | `data.views` |
| `data.remote` platform export helper | `integrations.platform.datasets` |
| `runs.PlatformBackend` | `integrations.platform.tracking.PlatformBackend` |
| `discover_component_payloads` | platform `helpers/utils/component_catalog.py` |

Known platform consumers, package examples, and agent guidance use these new
paths. Install the matching package source/version when running the updated
platform. This refactor does not publish a release or deploy either application.
