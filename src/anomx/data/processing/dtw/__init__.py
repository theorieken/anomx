# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""
Dynamic Time Warping (DTW)
--------------------------

Tools for computing Dynamic Time Warping alignment between time series, including cost matrices,
alignment algorithms, and windowing constraints (Sakoe-Chiba, Itakura).
"""

from anomx.data.processing.dtw.cost_matrix import CostMatrix
from anomx.data.processing.dtw.dtw import DTWAlignment, dtw
from anomx.data.processing.dtw.window import (
    CRWindow,
    Itakura,
    NoWindow,
    SakoeChiba,
    Window,
)

__all__ = [
    "CostMatrix",
    "DTWAlignment",
    "dtw",
    "CRWindow",
    "Itakura",
    "NoWindow",
    "SakoeChiba",
    "Window",
]
