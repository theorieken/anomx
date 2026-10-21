"""Anomx public API. Optional and scientific dependencies are loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.82"

_EXPORTS = {
    "AbsoluteErrorScorer": "anomx.detection.scorers.absolute_error",
    "AdaptiveThresholdDetector": "anomx.detection.base.detectors",
    "AnomalyEvent": "anomx.detection.results",
    "AnomalyEvents": "anomx.detection.results",
    "AnomalyPipeline": "anomx.detection.pipeline",
    "AnomalyScores": "anomx.detection.results",
    "AnomxDataset": "anomx.data",
    "BaseAlgorithm": "anomx.execution.base.algorithm",
    "BaseComponent": "anomx.base.component",
    "BaseConnector": "anomx.data",
    "BoundaryEstimating": "anomx.base.capabilities",
    "BoundaryScorer": "anomx.detection.base.scorers",
    "Capability": "anomx.base.capabilities",
    "ChangePointDetector": "anomx.detection.base.detectors",
    "Classifier": "anomx.models.base.classifiers",
    "ComponentCapabilities": "anomx.base.capabilities",
    "CompositeScorer": "anomx.detection.base.scorers",
    "ConstantBaselineModel": "anomx.models.forecasting.constant",
    "DCacheDataset": "anomx.data",
    "DartsDataset": "anomx.integrations.darts.datasets",
    "DartsNaiveSeasonalModel": "anomx.models.forecasting.darts_naive_seasonal",
    "DataCharacteristics": "anomx.data",
    "DataStructure": "anomx.base.signature",
    "Dataset": "anomx.data",
    "DatasetAdapter": "anomx.data",
    "DatasetSourceChanged": "anomx.data",
    "Detector": "anomx.detection.base.detectors",
    "DirectDataScorer": "anomx.detection.base.scorers",
    "Distributional": "anomx.base.capabilities",
    "EventAggregationDetector": "anomx.detection.base.detectors",
    "Forecast": "anomx.models.base.normality",
    "GraphView": "anomx.data",
    "Hierarchy": "anomx.data",
    "InferenceKind": "anomx.models.results",
    "InferenceOutput": "anomx.models.results",
    "IsolationForestModel": "anomx.models.representation.isolation_forest",
    "JobDefinition": "anomx.execution",
    "JobNode": "anomx.execution",
    "JobNodeType": "anomx.execution",
    "JobRunResult": "anomx.execution",
    "JobRunner": "anomx.execution",
    "LikelihoodScorer": "anomx.detection.base.scorers",
    "LocalFSConnector": "anomx.data",
    "Model": "anomx.models.base",
    "ModelSignature": "anomx.base.signature",
    "MountedDataset": "anomx.data",
    "MountedSource": "anomx.data",
    "NormalityModel": "anomx.models.base.normality",
    "ONNXModel": "anomx.models",
    "ObservationSet": "anomx.data",
    "PcaReconstructionModel": "anomx.models.reconstruction.pca",
    "PipelineOrchestrator": "anomx.execution.pipeline",
    "PipelineValidationError": "anomx.detection.pipeline",
    "PlatformClient": "anomx.integrations.platform",
    "Predictive": "anomx.base.capabilities",
    "PyTorchModel": "anomx.models",
    "QuantileThresholdDetector": "anomx.detection.detectors.quantile",
    "ReconstructionScorer": "anomx.detection.base.scorers",
    "Reconstructive": "anomx.base.capabilities",
    "RecordsAdapter": "anomx.data",
    "RelationSet": "anomx.data",
    "RepresentationScorer": "anomx.detection.base.scorers",
    "Representational": "anomx.base.capabilities",
    "ResidualScorer": "anomx.detection.base.scorers",
    "RollingWindowForecastModel": "anomx.models.forecasting.rolling_window",
    "Run": "anomx.execution.runs",
    "ScoreLevel": "anomx.detection.results",
    "Scorer": "anomx.detection.base.scorers",
    "SequenceView": "anomx.data",
    "StaticThresholdDetector": "anomx.detection.base.detectors",
    "StatisticalDetector": "anomx.detection.base.detectors",
    "TabularView": "anomx.data",
    "ThresholdDetector": "anomx.detection.detectors.threshold",
    "TimeSeriesBatch": "anomx.data",
    "TimeSeriesBatchAdapter": "anomx.data",
    "TimeSeriesDataset": "anomx.data",
    "TimeSeriesView": "anomx.data",
    "TorchAutoencoderModel": "anomx.models.reconstruction.torch_autoencoder",
    "WindowView": "anomx.data",
    "WorkContext": "anomx.execution.work",
    "ZScoreScorer": "anomx.detection.scorers.zscore",
    "current_run": "anomx.execution.runs",
    "finish": "anomx.execution.runs",
    "get_work_context": "anomx.execution.work",
    "init": "anomx.execution.runs",
    "iter_component_classes": "anomx.base.discovery",
    "log": "anomx.execution.runs",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
