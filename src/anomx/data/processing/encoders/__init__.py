# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""
Time Axis Encoders
------------------

Encoders for converting time index information into features, supporting cyclic encoding, datetime
attributes, integer indices, and custom callable encoders for both past and future covariates.
"""

from anomx.data.processing.encoders.encoders import (
    FutureCallableIndexEncoder,
    FutureCyclicEncoder,
    FutureDatetimeAttributeEncoder,
    FutureIntegerIndexEncoder,
    PastCallableIndexEncoder,
    PastCyclicEncoder,
    PastDatetimeAttributeEncoder,
    PastIntegerIndexEncoder,
    SequentialEncoder,
)

__all__ = [
    "FutureCallableIndexEncoder",
    "FutureCyclicEncoder",
    "FutureDatetimeAttributeEncoder",
    "FutureIntegerIndexEncoder",
    "PastCallableIndexEncoder",
    "PastCyclicEncoder",
    "PastDatetimeAttributeEncoder",
    "PastIntegerIndexEncoder",
    "SequentialEncoder",
]
