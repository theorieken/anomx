# Derived from Unit8 Darts (darts/typing.py), Apache-2.0.
# Copyright 2019 Unit8 SA. Additional contributions by the Darts contributors.
# Modified for Anomx: native imports, namespace, and configuration.
# See THIRD_PARTY_NOTICES.md and licenses/DARTS-APACHE-2.0.txt.

"""
Typing
------

This module contains custom Anomx types and type aliases.
"""

from collections.abc import Sequence
from datetime import tzinfo
from typing import TYPE_CHECKING, TypeAlias, Union

import pandas as pd

# prevent circular import at runtime (in timeseries.py)
if TYPE_CHECKING:
    from anomx.data.timeseries import TimeSeries

TimeIndex: TypeAlias = pd.DatetimeIndex | pd.RangeIndex

# req. Union with string literals (again, to avoid circular dependency issues)
TimeSeriesLike: TypeAlias = Union["TimeSeries", Sequence["TimeSeries"]]

TimeZone: TypeAlias = str | tzinfo | None
