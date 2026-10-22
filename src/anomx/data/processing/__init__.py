"""Reusable preprocessing for native time series and dataset frames.

``Pipeline`` composes TimeSeries transformers, while ``ProcessingPipeline``
composes DataFrame ``Transform`` steps for materialized data and batch streams.
Transformers, covariate encoders and dynamic time warping live in their named
subpackages. The numerical baseline is derived from Unit8 Darts under Apache 2.0;
its implementations are owned by Anomx and do not import the Darts package.
"""

from typing import TYPE_CHECKING

from anomx._series_utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from anomx.data.base.transform import Transform as Transform
    from anomx.data.processing.dataframe import ProcessingPipeline as ProcessingPipeline
    from anomx.data.processing.pipeline import Pipeline as Pipeline
    from anomx.data.processing.resampling import Resample as Resample

_LAZY_IMPORTS = {
    "Pipeline": "anomx.data.processing.pipeline",
    "ProcessingPipeline": "anomx.data.processing.dataframe",
    "Resample": "anomx.data.processing.resampling",
    "Transform": "anomx.data.base.transform",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
