# Anomx

Anomx is a Python library for time-series datasets, forecasting models, anomaly
scorers, and anomaly detectors.

The library is designed to be the modeling core for the Anomx platform while
remaining useful as a standalone package.

## Main Concepts

- `TimeSeriesDataset`: a typed wrapper around timestamp-indexed data and metadata.
- `AnomalyScorer`: converts observations or residuals into anomaly scores.
- `AnomalyDetector`: converts datasets into scores and anomaly labels.
- `ForecastingModel`: produces future values for a time horizon.
- `DartsForecastingModel`: optional adapter for Darts models.
