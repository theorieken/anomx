# Darts foundations and flexible work

Install `pip install 'anomx[ml]'` for Darts, PyTorch, and ONNX. The base install
remains lightweight. `anomx[darts]` provides Darts without neural dependencies;
`anomx[onnx]` provides portable inference and artifact validation.

## Dataset, Model, Scorer, Detector

`anomx.Dataset` is a lightweight table or versioned platform source, supporting
`time_series`, `sequence` and `independent_samples`. See [versioned datasets](datasets.md).
Use `.to_darts()` for Darts transformations and catalog models; `PyTorchModel`
accepts the new Dataset directly. Code that needs the former Darts subclass can
use `from anomx import DartsDataset`, or the unchanged `anomx.datasets.Dataset`.
That class retains Darts operations such as `split_after` and component selection.

`anomx.models.Model` extends Darts `GlobalForecastingModel`.
`anomx.scorers.Scorer` extends Darts `AnomalyScorer`, and
`anomx.detectors.Detector` extends Darts `Detector`. The scorer and detector
modules export Darts' common implementations, plus an absolute-error scorer and
upper-quantile detector. `anomx.models` exposes the main Darts forecasting models
on demand. Their native Darts options and limitations apply.

Existing `anomx.components`, `anomx.data.TimeSeriesDataset`, and the legacy
`anomx.AnomxDataset` platform-export container retain their earlier API. New work
uses `Dataset` and the new `models`, `scorers`, and `detectors` namespaces. Existing top-level Dataset
users needing the full Darts API should switch to `DartsDataset`.

```python
from anomx import Dataset, PlatformClient

client = PlatformClient(url="https://anomx.example/api", token="your-token")
series = Dataset.from_channels(
    ["data_channel-12345678-1234-5678-1234-567812345678"],
    client=client,
    start="2026-01-01T00:00:00Z",
    end="2026-01-01T01:00:00Z",
    frequency="1s",
)
```

`ANOMX_URL` and `ANOMX_TOKEN` can configure the connection instead. The URL must
include the platform API prefix. Access is authorized by the platform. Object
references use `data_channel-<uuid>` or `data_recorded_channel-<uuid>`. External
channel references, such as DOOCS property paths, resolve to accessible recorded
channels through the platform. The package neither imports Django nor needs the CLI
agent's state. A custom `loader(channels=..., start=..., end=..., frequency=...)`
can return a wide pandas DataFrame with a `timestamp` column or DatetimeIndex.
Channel metadata records the original references; timezone-aware values become
UTC before Darts receives them. The generic Dataset preserves source values. Darts conversion and model
validation enforce temporal and numeric requirements; custom loaders must choose
preprocessing explicitly. `work.dataset()` retains the older Darts channel API. The platform loader aligns recorded numeric samples using their last
value, forward-fills gaps, and discards leading incomplete rows. With `frequency`,
it first resamples each channel to the last value in each fixed interval. The
platform limits requests to 50,000 samples and intervals of at least 10 ms.
Choose a frequency appropriate to the recorded signal before training.

## A tiny PyTorch model

```python
import numpy as np
from anomx import Dataset, PyTorchModel

series = Dataset.from_values(np.sin(np.arange(128, dtype=np.float32) / 8))
model = PyTorchModel(input_chunk_length=12, output_chunk_length=1, n_epochs=5)
model.fit(series)
forecast = model.predict(10)
artifact = model.export_onnx("model.onnx")
```

The default network is one linear layer. Supply `module=your_torch_module` for a
custom network. Inputs have shape `(batch, input_window, channels)` and outputs
`(batch, output_window, channels)`. Multiple training series must have identical
ordered component names. Training uses MSE and Adam by default, records average
epoch losses in `training_history`, and accepts `on_epoch=callback` for KPI
streaming. The model accepts deterministic targets; choose an appropriate Darts
catalog model for covariates or probabilistic forecasting.

ONNX export validates the graph and writes a neighboring JSON manifest with the
tensor contract and training history. The graph is self-contained, has a dynamic
batch dimension, and includes network weights; dataset normalization is not
implicit and must be represented in your model or separately versioned. Use
`ONNXModel(path).predict(windows)` for inference without loading pickled Python
objects. ONNX is an inference artifact, not a resumable optimizer checkpoint.

## Work context

The task worker injects `work`, `inputs` (also `context`), and a writable `result` into Python work.
`work.compute(code, inputs=..., target=...)` submits a block to the selected
compute worker or integration. The Anomx task worker waits for the block and
returns its `result`, so following graph blocks can use it. Other execution hosts
may return a submission reference; the callback defines that behavior.

```python
result = work.compute('''
import numpy as np
from anomx import Dataset, PyTorchModel

series = Dataset.from_values(np.sin(np.arange(128, dtype=np.float32) / 8))
# For recorded platform data: series = work.dataset(["data_channel-12345678-1234-5678-1234-567812345678"])
model = PyTorchModel(input_chunk_length=12, n_epochs=5, random_state=42)
model.fit(series, on_epoch=lambda row: work.metric("loss", row["loss"], int(row["epoch"])))
artifact = model.export_onnx("model.onnx")
result = work.publish_model(
    artifact, name="Simple sine forecast",
    metrics={"loss": model.training_history[-1]["loss"]},
    metadata={"purpose": "flexible-work smoke example"},
)
work.log("Training completed")
''')
```

Other host interfaces are `work.notify(title, message, severity)` and
`work.finding(title, score=..., message=..., **evidence)`; `work.detection` remains
a compatibility alias. Findings can include ISO 8601 event timestamps, channel
references, model/data versions and bounded observed/expected evidence. The host
persists them with Job/WorkRun provenance and pending human feedback. Model publication delegates
durable storage, ownership, versioning, and MLflow synchronization to the host.
An unavailable compute/publication callback raises rather than pretending to
persist anything. Standalone contexts collect logs, metrics, and findings in
`work.events` for inspection.

Use the model reference returned by publication to load a stored version on a
compute worker. The host checks access and returns the ONNX graph with a SHA-256
checksum; `load_model()` verifies it before starting inference. This interface
accepts self-contained artifacts up to 64 MiB and requires `anomx[onnx]`:

```python
result = work.compute('''
model = work.load_model(inputs["model_reference"])
result = model.run(inputs["tensors"])
''', inputs={
    "model_reference": "models_model-12345678-1234-5678-1234-567812345678",
    "tensors": {"observations": [[[0.0]] * 12]},
})
```

Use the tensor names and dimensions shown in the model's structure. `run()`
supports multiple named input and output tensors, converts inputs to the graph's
dtypes, and returns named JSON-compatible outputs. Standalone scripts can also
call `ONNXModel(path_or_bytes).run(tensors)`. Hosts provide a `model_artifact`
callback receiving `reference` and returning `artifact_base64` and
`checksum_sha256`.

Host adapters construct `WorkContext(callbacks={...}, data_loader=..., inputs=...,
job_id=..., run_id=...)`. Callback names match method names and receive matching
keyword arguments. Activate it with `with work.activate():` so helper modules can
call `get_work_context()` without global process state. This API grants no
sandboxing: the execution host is responsible for isolating untrusted Python.

Run `python examples/flexible_job_training.py` for the standalone training,
scoring, detection, export, and ONNX inference example.

Reference APIs: [Darts forecasting](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html),
[Darts anomaly detection](https://unit8co.github.io/darts/generated_api/darts.ad.html),
[ONNX checker](https://onnx.ai/onnx/api/checker.html).

## Graph execution

`WorkGraph` and `WorkStep` define a deterministic acyclic graph. A host receives
one Python or inference step at a time and dispatches its selected runtime.
`task` means the orchestrating task worker, `default` uses the work's configured
compute target, and `service:<id>` or `integration:<id>` select an authorized
platform resource. Inference currently executes Python and retains a distinct
kind for future model-session optimization.

Use `work.run_graph(graph)` with the editor's nodes and edges representation.
Every executable step must connect from the single start node to the single end
node. Each step sees original job inputs, `inputs["previous"]` (one predecessor's
result, or a map at a join), and `inputs["results"]` (completed results by step ID).
Steps run in topological order, fail immediately on errors, and share the run's
deadline and cancellation state. `work.compute()` inherits the current inputs
unless an explicit replacement mapping is supplied.

The platform's transport preserves NumPy arrays and pandas tables without pickle.
Arrays and tables are limited to 16 MiB each; return stored artifact references
for larger outputs. Plain JSON values are also supported. Arbitrary Python
objects and non-finite JSON scalar values are rejected, rather than stringified.
