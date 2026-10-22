# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

"""
Data Transformers
-----------------

Data transformers for preprocessing time series data, including scalers, missing value fillers,
differencing, BoxCox transformations, and hierarchical reconciliation methods.
"""

from typing import TYPE_CHECKING

from anomx._series_utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from anomx.data.processing.transformers.base_data_transformer import (
        BaseDataTransformer as BaseDataTransformer,
    )
    from anomx.data.processing.transformers.boxcox import BoxCox as BoxCox
    from anomx.data.processing.transformers.diff import Diff as Diff
    from anomx.data.processing.transformers.fittable_data_transformer import (
        FittableDataTransformer as FittableDataTransformer,
    )
    from anomx.data.processing.transformers.invertible_data_transformer import (
        InvertibleDataTransformer as InvertibleDataTransformer,
    )
    from anomx.data.processing.transformers.mappers import (
        InvertibleMapper as InvertibleMapper,
    )
    from anomx.data.processing.transformers.mappers import Mapper as Mapper
    from anomx.data.processing.transformers.midas import MIDAS as MIDAS
    from anomx.data.processing.transformers.missing_values_filler import (
        MissingValuesFiller as MissingValuesFiller,
    )
    from anomx.data.processing.transformers.reconciliation import (
        BottomUpReconciliator as BottomUpReconciliator,
    )
    from anomx.data.processing.transformers.reconciliation import (
        MinTReconciliator as MinTReconciliator,
    )
    from anomx.data.processing.transformers.reconciliation import (
        TopDownReconciliator as TopDownReconciliator,
    )
    from anomx.data.processing.transformers.scaler import Scaler as Scaler
    from anomx.data.processing.transformers.static_covariates_transformer import (
        StaticCovariatesTransformer as StaticCovariatesTransformer,
    )
    from anomx.data.processing.transformers.window_transformer import (
        WindowTransformer as WindowTransformer,
    )

_LAZY_IMPORTS: dict[str, str] = {
    "BaseDataTransformer": "anomx.data.processing.transformers.base_data_transformer",
    "BoxCox": "anomx.data.processing.transformers.boxcox",
    "Diff": "anomx.data.processing.transformers.diff",
    "FittableDataTransformer": "anomx.data.processing.transformers.fittable_data_transformer",
    "InvertibleDataTransformer": "anomx.data.processing.transformers.invertible_data_transformer",
    "InvertibleMapper": "anomx.data.processing.transformers.mappers",
    "Mapper": "anomx.data.processing.transformers.mappers",
    "MIDAS": "anomx.data.processing.transformers.midas",
    "MissingValuesFiller": "anomx.data.processing.transformers.missing_values_filler",
    "BottomUpReconciliator": "anomx.data.processing.transformers.reconciliation",
    "MinTReconciliator": "anomx.data.processing.transformers.reconciliation",
    "TopDownReconciliator": "anomx.data.processing.transformers.reconciliation",
    "Scaler": "anomx.data.processing.transformers.scaler",
    "StaticCovariatesTransformer": "anomx.data.processing.transformers.static_covariates_transformer",
    "WindowTransformer": "anomx.data.processing.transformers.window_transformer",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
