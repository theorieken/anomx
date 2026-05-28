# API Outline

## Datasets

- `anomx.datasets.TimeSeriesDataset`
- `anomx.datasets.ChannelMetadata`
- `anomx.datasets.make_sine_anomaly_dataset`

## Scorers

- `anomx.scorers.AnomalyScorer`
- `anomx.scorers.ZScoreScorer`
- `anomx.scorers.ThresholdScorer`

## Detectors

- `anomx.detectors.AnomalyDetector`
- `anomx.detectors.DetectionResult`
- `anomx.detectors.MovingAverageDetector`

## Models

- `anomx.models.ForecastingModel`
- `anomx.models.Forecast`
- `anomx.models.NaiveSeasonalModel`

## Integrations

- `anomx.integrations.DartsForecastingModel`
- `anomx.integrations.is_darts_available`
