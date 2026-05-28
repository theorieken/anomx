# anomx

`anomx` is the core Python library for the Anomx platform: time-series datasets,
forecasting models, anomaly scorers, and anomaly detectors built for interpretable,
production-oriented workflows.

Anomx is a data intelligence platform focused on anomaly detection and predictive
insights in complex, data-driven systems. The platform handles orchestration,
workers, storage, connectors, audit trails, and human-in-the-loop feedback. This
repository contains the reusable modeling layer that those systems compose.

## Goals

- Provide clean primitives for time-series datasets, scorers, detectors, and models.
- Support both batch and online anomaly detection workflows.
- Keep outputs interpretable: scores, thresholds, timestamps, labels, and metadata.
- Stay modular enough to power the Anomx platform without coupling the library to
  the platform's storage or orchestration layer.
- Integrate with the broader time-series ecosystem, especially Darts, through
  optional adapters.

## Installation

The package is structured for PyPI distribution:

```bash
pip install anomx
```

For local development:

```bash
git clone https://github.com/anomx/anomx.git
cd anomx
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "anomx[darts]"     # Darts forecasting model adapters
pip install "anomx[docs]"      # Documentation tooling
pip install "anomx[platform]"  # Platform-facing IO dependencies
pip install "anomx[release]"   # Build and PyPI publishing tooling
```

## Package Layout

```text
src/anomx/
  datasets/      TimeSeriesDataset, metadata, dataset loaders, transforms
  scorers/       Anomaly score contracts and scoring implementations
  detectors/     Batch and online anomaly detector contracts and implementations
  models/        Forecasting model contracts and baseline forecasters
  integrations/  Optional adapters, including Darts
```

## Quick Start

```python
from anomx.datasets import make_sine_anomaly_dataset
from anomx.detectors import MovingAverageDetector
from anomx.models import NaiveSeasonalModel

dataset = make_sine_anomaly_dataset()

detector = MovingAverageDetector(window=24, threshold=3.0)
result = detector.fit_predict(dataset)
print(result.to_dataframe().tail())

model = NaiveSeasonalModel(season_length=24).fit(dataset)
forecast = model.predict(12)
print(forecast.to_dataframe())
```

## Darts Integration

The default install stays lightweight. To use Darts models, install the optional
extra and wrap any compatible Darts forecasting model:

```python
from darts.models import ExponentialSmoothing

from anomx.datasets import make_sine_anomaly_dataset
from anomx.integrations import DartsForecastingModel

dataset = make_sine_anomaly_dataset()
model = DartsForecastingModel(ExponentialSmoothing())
forecast = model.fit(dataset).predict(24)
```

The intent is to let Anomx expose Darts-backed forecasting and anomaly workflows
without making Darts a required dependency for every user.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src/anomx
python -m pip install -e ".[release]"
python -m build
```

## Platform Context

Anomx prioritizes:

- Signal over noise: surface only relevant insights.
- Clarity over complexity: outputs should be interpretable.
- Actionability: every insight should support a decision.

The platform is designed around modular pipelines, real-time and batch workers,
heterogeneous data sources, and versioned entities such as datasets, datasources,
channels, jobs, runs, findings, and model artifacts. This library is the focused
modeling and time-series package used by that larger system.

## Status

This repository is in pre-alpha scaffold stage. The public API should be treated
as provisional until the first stable release.
