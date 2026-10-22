# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""
Metrics
=======

Regression Metrics
------------------

For deterministic forecasts (point predictions with `num_samples == 1`), probabilistic forecasts (`num_samples > 1`),
and quantile forecasts. For probabilistic and quantile forecasts, use parameter `q` to define the quantile(s) to
compute the deterministic metrics on:

- Aggregated over time:
    Absolute metrics:
        - :func:`MERR <anomx.metrics.metrics.merr>`: Mean Error
        - :func:`MAE <anomx.metrics.metrics.mae>`: Mean Absolute Error
        - :func:`MSE <anomx.metrics.metrics.mse>`: Mean Squared Error
        - :func:`RMSE <anomx.metrics.metrics.rmse>`: Root Mean Squared Error
        - :func:`RMSLE <anomx.metrics.metrics.rmsle>`: Root Mean Squared Log Error

    Relative metrics:
        - :func:`MASE <anomx.metrics.metrics.mase>`: Mean Absolute Scaled Error
        - :func:`MSSE <anomx.metrics.metrics.msse>`: Mean Squared Scaled Error
        - :func:`RMSSE <anomx.metrics.metrics.rmsse>`: Root Mean Squared Scaled Error
        - :func:`MAPE <anomx.metrics.metrics.mape>`: Mean Absolute Percentage Error
        - :func:`wMAPE <anomx.metrics.metrics.wmape>`: weighted Mean Absolute Percentage Error
        - :func:`sMAPE <anomx.metrics.metrics.smape>`: symmetric Mean Absolute Percentage Error
        - :func:`OPE <anomx.metrics.metrics.ope>`: Overall Percentage Error
        - :func:`MARRE <anomx.metrics.metrics.marre>`: Mean Absolute Ranged Relative Error

    Other metrics:
        - :func:`R2 <anomx.metrics.metrics.r2_score>`: Coefficient of Determination
        - :func:`CV <anomx.metrics.metrics.coefficient_of_variation>`: Coefficient of Variation
        - :func:`AUTC <anomx.metrics.metrics.autc>`: Area Under Tolerance Curve

- Per time step:
    Absolute metrics:
        - :func:`ERR <anomx.metrics.metrics.err>`: Error
        - :func:`AE <anomx.metrics.metrics.ae>`: Absolute Error
        - :func:`SE <anomx.metrics.metrics.se>`: Squared Error
        - :func:`SLE <anomx.metrics.metrics.sle>`: Squared Log Error

    Relative metrics:
        - :func:`ASE <anomx.metrics.metrics.ase>`: Absolute Scaled Error
        - :func:`SSE <anomx.metrics.metrics.sse>`: Squared Scaled Error
        - :func:`APE <anomx.metrics.metrics.ape>`: Absolute Percentage Error
        - :func:`sAPE <anomx.metrics.metrics.sape>`: symmetric Absolute Percentage Error
        - :func:`ARRE <anomx.metrics.metrics.arre>`: Absolute Ranged Relative Error

For probabilistic forecasts (storchastic predictions with `num_samples >> 1`) and quantile forecasts:

- Aggregated over time:
    Probabilistic metrics:
        - :func:`MCRPS <anomx.metrics.metrics.mcrps>`: Mean Continuous Ranked Probability Score

    Quantile metrics:
        - :func:`MQL <anomx.metrics.metrics.mql>`: Mean Quantile Loss
        - :func:`QR <anomx.metrics.metrics.qr>`: Quantile Risk

    Quantile interval metrics:
        - :func:`MIW <anomx.metrics.metrics.miw>`: Mean Interval Width
        - :func:`MWS <anomx.metrics.metrics.miws>`: Mean Interval Winkler Score
        - :func:`MIC <anomx.metrics.metrics.mic>`: Mean Interval Coverage
        - :func:`MINCS_QR <anomx.metrics.metrics.mincs_qr>`: Mean Interval Non-Conformity Score for Quantile Regression

- Per time step:
    Probabilistic metrics:
        - :func:`CRPS <anomx.metrics.metrics.crps>`: Continuous Ranked Probability Score

    Quantile metrics:
        - :func:`QL <anomx.metrics.metrics.ql>`: Quantile Loss

    Quantile interval metrics:
        - :func:`IW <anomx.metrics.metrics.iw>`: Interval Width
        - :func:`WS <anomx.metrics.metrics.iws>`: Interval Winkler Score
        - :func:`IC <anomx.metrics.metrics.ic>`: Interval Coverage
        - :func:`INCS_QR <anomx.metrics.metrics.incs_qr>`: Interval Non-Conformity Score for Quantile Regression

Classification Metrics
----------------------

For deterministic forecasts (point predictions with `num_samples == 1`), probabilistic forecasts (`num_samples > 1`),
and class label probability forecasts. Most metrics allow to extract the scores for specific labels with parameter
`labels`.

- Aggregated over time:
    - :func:`Accuracy <anomx.metrics.metrics.accuracy>`: Accuracy Score
    - :func:`Precision <anomx.metrics.metrics.precision>`: Precision Score
    - :func:`Recall <anomx.metrics.metrics.recall>`: Recall Score
    - :func:`F1 <anomx.metrics.metrics.f1>`: F1 Score
    - :func:`CM <anomx.metrics.metrics.confusion_matrix>`: Confusion Matrix

Dynamic Time Warping Metrics
----------------------------

For Dynamic Time Warping (DTW) (aggregated over time):

- :func:`DTW <anomx.metrics.metrics.dtw_metric>`: Dynamic Time Warping Metric



"""

import importlib
from typing import TYPE_CHECKING

from anomx._series_utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from anomx.metrics.metrics import accuracy as accuracy
    from anomx.metrics.metrics import ae as ae
    from anomx.metrics.metrics import ape as ape
    from anomx.metrics.metrics import arre as arre
    from anomx.metrics.metrics import ase as ase
    from anomx.metrics.metrics import autc as autc
    from anomx.metrics.metrics import (
        coefficient_of_variation as coefficient_of_variation,
    )
    from anomx.metrics.metrics import confusion_matrix as confusion_matrix
    from anomx.metrics.metrics import crps as crps
    from anomx.metrics.metrics import dtw_metric as dtw_metric
    from anomx.metrics.metrics import err as err
    from anomx.metrics.metrics import f1 as f1
    from anomx.metrics.metrics import ic as ic
    from anomx.metrics.metrics import incs_qr as incs_qr
    from anomx.metrics.metrics import iw as iw
    from anomx.metrics.metrics import iws as iws
    from anomx.metrics.metrics import mae as mae
    from anomx.metrics.metrics import mape as mape
    from anomx.metrics.metrics import marre as marre
    from anomx.metrics.metrics import mase as mase
    from anomx.metrics.metrics import mcrps as mcrps
    from anomx.metrics.metrics import merr as merr
    from anomx.metrics.metrics import mic as mic
    from anomx.metrics.metrics import mincs_qr as mincs_qr
    from anomx.metrics.metrics import miw as miw
    from anomx.metrics.metrics import miws as miws
    from anomx.metrics.metrics import mql as mql
    from anomx.metrics.metrics import mse as mse
    from anomx.metrics.metrics import msse as msse
    from anomx.metrics.metrics import ope as ope
    from anomx.metrics.metrics import precision as precision
    from anomx.metrics.metrics import ql as ql
    from anomx.metrics.metrics import qr as qr
    from anomx.metrics.metrics import r2_score as r2_score
    from anomx.metrics.metrics import recall as recall
    from anomx.metrics.metrics import rmse as rmse
    from anomx.metrics.metrics import rmsle as rmsle
    from anomx.metrics.metrics import rmsse as rmsse
    from anomx.metrics.metrics import sape as sape
    from anomx.metrics.metrics import se as se
    from anomx.metrics.metrics import sle as sle
    from anomx.metrics.metrics import smape as smape
    from anomx.metrics.metrics import sse as sse
    from anomx.metrics.metrics import wmape as wmape
    from anomx.metrics.plotting import plot_tolerance_curve as plot_tolerance_curve

_LAZY_IMPORTS: dict[str, str] = {
    "plot_tolerance_curve": "anomx.metrics.plotting",
    "accuracy": "anomx.metrics.metrics",
    "ae": "anomx.metrics.metrics",
    "crps": "anomx.metrics.metrics",
    "ape": "anomx.metrics.metrics",
    "arre": "anomx.metrics.metrics",
    "ase": "anomx.metrics.metrics",
    "autc": "anomx.metrics.metrics",
    "coefficient_of_variation": "anomx.metrics.metrics",
    "confusion_matrix": "anomx.metrics.metrics",
    "dtw_metric": "anomx.metrics.metrics",
    "err": "anomx.metrics.metrics",
    "f1": "anomx.metrics.metrics",
    "ic": "anomx.metrics.metrics",
    "incs_qr": "anomx.metrics.metrics",
    "iw": "anomx.metrics.metrics",
    "iws": "anomx.metrics.metrics",
    "mae": "anomx.metrics.metrics",
    "mape": "anomx.metrics.metrics",
    "mcrps": "anomx.metrics.metrics",
    "marre": "anomx.metrics.metrics",
    "mase": "anomx.metrics.metrics",
    "merr": "anomx.metrics.metrics",
    "mic": "anomx.metrics.metrics",
    "mincs_qr": "anomx.metrics.metrics",
    "miw": "anomx.metrics.metrics",
    "miws": "anomx.metrics.metrics",
    "mql": "anomx.metrics.metrics",
    "mse": "anomx.metrics.metrics",
    "msse": "anomx.metrics.metrics",
    "ope": "anomx.metrics.metrics",
    "precision": "anomx.metrics.metrics",
    "ql": "anomx.metrics.metrics",
    "qr": "anomx.metrics.metrics",
    "r2_score": "anomx.metrics.metrics",
    "recall": "anomx.metrics.metrics",
    "rmse": "anomx.metrics.metrics",
    "rmsle": "anomx.metrics.metrics",
    "rmsse": "anomx.metrics.metrics",
    "sape": "anomx.metrics.metrics",
    "se": "anomx.metrics.metrics",
    "sle": "anomx.metrics.metrics",
    "smape": "anomx.metrics.metrics",
    "sse": "anomx.metrics.metrics",
    "wmape": "anomx.metrics.metrics",
}

_TIME_DEPENDENT_METRIC_NAMES = {
    "ae",
    "ape",
    "arre",
    "ase",
    "crps",
    "err",
    "ql",
    "sape",
    "se",
    "sle",
    "sse",
    "iw",
    "iws",
    "ic",
    "incs_qr",
}
_CLASSIFICATION_METRIC_NAMES = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
}


def _build_time_dependent_metrics():
    module = importlib.import_module("anomx.metrics.metrics")
    return {getattr(module, n) for n in _TIME_DEPENDENT_METRIC_NAMES}


def _build_classification_metrics():
    module = importlib.import_module("anomx.metrics.metrics")
    return {getattr(module, n) for n in _CLASSIFICATION_METRIC_NAMES}


__all__, __getattr__, __dir__ = setup_lazy_imports(
    _LAZY_IMPORTS,
    __name__,
    globals(),
    extra_attrs={
        "TIME_DEPENDENT_METRICS": _build_time_dependent_metrics,
        "CLASSIFICATION_METRICS": _build_classification_metrics,
    },
)
