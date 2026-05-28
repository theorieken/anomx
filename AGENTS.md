# AGENTS.md

This repository contains the open-source Python package that backs Anomx, a
data intelligence platform for time series, forecasting, anomaly detection, and
human-in-the-loop operational workflows.

## Project Intent

The package should stay focused on reusable library primitives:

- `datasets`: time series containers, dataset loaders, splits, transforms, and metadata.
- `scorers`: functions and classes that turn observations, residuals, forecasts, or model
  outputs into anomaly scores.
- `detectors`: online and batch anomaly detectors that produce actionable anomaly labels
  and detection results.
- `models`: forecasting, representation, and predictive models.
- `integrations`: optional adapters for third-party ecosystems such as Darts.

The platform itself owns orchestration, persistence, workers, user feedback loops,
audit trails, and product-specific APIs. This library should expose the clean core
that the platform can compose.

## Engineering Guidelines

- Prefer small, typed, composable abstractions over platform-specific coupling.
- Keep the default install lightweight. Put heavy ML dependencies behind extras such as
  `anomx[darts]`.
- Treat batch and online use cases as first-class design constraints.
- Favor interpretable outputs: anomaly scores, thresholds, labels, timestamps, and
  metadata should be easy to inspect.
- Match the existing public API style before adding new patterns.
- Add or update tests for behavior changes.

## Local Commands

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src/anomx
python -m pip install -e ".[release]"
python -m build
```

## Compatibility

The package targets Python 3.10+ and uses a `src/` layout. Public APIs should be
documented in docstrings and exported deliberately through package `__init__.py` files.
