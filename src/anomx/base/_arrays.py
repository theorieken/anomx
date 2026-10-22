# ruff: noqa: ANN401
# Adapter boundaries intentionally accept arbitrary third-party model/tensor types.
"""Private, dependency-light array conversion for user supplied components."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def component_names(data: Any) -> tuple[object, ...] | None:
    """Return ordered named input features when the container provides them."""
    if isinstance(data, pd.DataFrame):
        return tuple(data.columns)
    if isinstance(data, pd.Series):
        return (data.name,)
    components = getattr(data, "components", None)
    return tuple(components) if components is not None else None


def call_array(model: Any, values: np.ndarray) -> Any:
    """Invoke a callable, converting tensor inputs/outputs for Torch modules lazily."""
    is_torch = any(
        base.__name__ == "Module" and base.__module__.startswith("torch.")
        for base in type(model).__mro__
    )
    if not is_torch:
        return model(values)
    import torch

    parameter = next(model.parameters(), None)
    if parameter is None:
        parameter = next(model.buffers(), None)
    device = parameter.device if parameter is not None else "cpu"
    dtype = (
        parameter.dtype
        if parameter is not None and parameter.is_floating_point()
        else torch.float32
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.as_tensor(values, device=device, dtype=dtype))
    if not isinstance(output, torch.Tensor):
        raise ValueError("PyTorch scoring/detection modules must return a tensor.")
    return output.detach().cpu().numpy()


def as_array(data: Any, *, name: str = "data") -> np.ndarray:
    """Validate numeric, finite observations without flattening tensor axes."""
    from anomx.data.timeseries import TimeSeries

    if isinstance(data, TimeSeries):
        if not data.is_deterministic:
            raise ValueError(f"{name} must be deterministic; reduce samples explicitly.")
        values = data.values(copy=False)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        values = data.to_numpy()
    else:
        values = data
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.ndim == 0 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty, finite, and have an observation axis.")
    return array[:, None] if array.ndim == 1 else array


def restore_output(
    data: Any,
    values: Any,
    *,
    name: str = "output",
    preserve_components: bool = True,
) -> Any:
    """Restore the input container, validating the output observation axis."""
    from anomx.data.timeseries import TimeSeries

    array = as_array(values, name=name)
    if len(array) != len(data):
        raise ValueError(f"{name} must return one output per input observation.")
    if isinstance(data, (TimeSeries, pd.DataFrame, pd.Series)):
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional for a tabular/time-series input.")
        columns = (
            getattr(data, "components", getattr(data, "columns", None))
            if preserve_components
            else None
        )
        if columns is None or len(columns) != array.shape[1]:
            columns = (
                [name] if array.shape[1] == 1 else [f"{name}_{i}" for i in range(array.shape[1])]
            )
        if isinstance(data, TimeSeries):
            same_width = preserve_components and array.shape[1] == data.n_components
            return TimeSeries.from_times_and_values(
                data.time_index,
                array,
                columns=columns,
                static_covariates=data.static_covariates if same_width else None,
                hierarchy=data.hierarchy if same_width else None,
                metadata=data.metadata,
            )
        return pd.DataFrame(array, index=data.index, columns=columns)
    return array
