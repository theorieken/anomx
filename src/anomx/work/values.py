"""Lossless work transport for arrays and tables without pickle or arbitrary objects."""

from __future__ import annotations

import base64
import io
import json
import math

MAX_VALUE_BYTES = 16 * 1024 * 1024
_TAG = "__anomx_work_value__"


def encode_work_value(value: object) -> object:
    """Encode supported values into JSON, failing instead of silently stringifying."""
    if value is None or isinstance(value, (str, int, float, bool)):
        json.dumps(value, allow_nan=False)
        return value
    if isinstance(value, (list, tuple)):
        return [encode_work_value(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Work mappings require string keys.")
        if _TAG in value:
            raise ValueError(f"{_TAG} is reserved for work transport.")
        return {key: encode_work_value(item) for key, item in value.items()}
    import numpy as np
    import pandas as pd

    if isinstance(value, np.generic):
        return encode_work_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.nbytes > MAX_VALUE_BYTES:
            raise ValueError("Work arrays must not contain objects or exceed 16 MiB.")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {_TAG: "ndarray", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
    if isinstance(value, pd.DataFrame):
        data = value.to_json(orient="table", date_format="iso", index=True)
        if len(data.encode()) > MAX_VALUE_BYTES:
            raise ValueError("Work tables must not exceed 16 MiB.")
        return {_TAG: "dataframe", "data": data}
    raise TypeError(
        f"Unsupported work value: {type(value).__name__}. "
        "Return data or a stored artifact reference."
    )


def decode_work_value(value: object) -> object:
    """Decode only the documented array/table types, never executable objects."""
    if isinstance(value, list):
        return [decode_work_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if _TAG not in value:
        return {key: decode_work_value(item) for key, item in value.items()}
    if set(value) != {_TAG, "data"} or not isinstance(value["data"], str):
        raise ValueError("Invalid work transport value.")
    if len(value["data"].encode()) > MAX_VALUE_BYTES * 4 // 3 + 4096:
        raise ValueError("Work transport value is too large.")
    if value[_TAG] == "ndarray":
        import numpy as np

        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) > MAX_VALUE_BYTES + 4096:
            raise ValueError("Work array is too large.")
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)  # type: ignore[no-untyped-call]
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("Unsupported NumPy transport format.")
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _, dtype = reader(stream)  # type: ignore[no-untyped-call]
        if (
            dtype.hasobject
            or math.prod(shape) * dtype.itemsize > MAX_VALUE_BYTES
            or math.prod(shape) * dtype.itemsize != len(raw) - stream.tell()
        ):
            raise ValueError("Invalid work array size or dtype.")
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.hasobject
            or array.nbytes > MAX_VALUE_BYTES
        ):
            raise ValueError("Invalid work array.")
        return array
    if value[_TAG] == "dataframe":
        import pandas as pd

        if len(value["data"].encode()) > MAX_VALUE_BYTES:
            raise ValueError("Work table is too large.")
        return pd.read_json(io.StringIO(value["data"]), orient="table")
    raise ValueError("Unknown work transport value type.")
