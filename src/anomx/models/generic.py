# ruff: noqa: ANN401
# Adapter boundaries intentionally accept arbitrary third-party model/tensor types.
"""Adapters for user supplied estimators, functions, and PyTorch modules."""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from anomx.base._arrays import as_array, component_names, restore_output
from anomx.models.base.normality import NormalityModel


class GenericModel(NormalityModel):
    """Adapt an estimator, callable, or ``torch.nn.Module`` to Anomx.

    ``fit_fn(model, x, y)`` and ``predict_fn(model, x)`` explicitly override the
    underlying implementation. Estimators otherwise use their ``fit`` and
    ``predict`` methods; callable models support inference without fitting.
    The observation axis is always first. Prediction must preserve its length,
    while the remaining axes may change (for example for embeddings).

    PyTorch modules receive float32 inputs and train with Adam/MSE unless an
    optimizer factory or loss function is provided. Classification uses integer
    class targets and cross-entropy; prediction returns class logits.
    Reconstruction models may
    omit targets; other families require them. No scaling, window creation, or
    forecast alignment is implicit. Use :class:`TorchForecastingModel` for
    sliding-window forecasting. Torch remains an optional dependency.

    Parameters
    ----------
    model
        Estimator, callable, or torch module owned by this adapter.
    family
        ``predictive``, ``forecasting``, ``reconstruction``, ``representation``,
        or ``classification``. Describes purpose without restricting tensor axes.
    epochs, batch_size, learning_rate
        Positive PyTorch training settings; ignored by estimator backends.
    """

    component_type = "model"
    component_key = "generic_model"
    component_name = "Generic Model"
    component_discoverable = False

    def __init__(
        self,
        model: Any,
        *,
        family: str = "predictive",
        fit_fn: Callable | None = None,
        predict_fn: Callable | None = None,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        loss_fn: Callable | None = None,
        optimizer_factory: Callable | None = None,
        device: str = "cpu",
        random_state: int = 42,
    ) -> None:
        if family not in {
            "predictive",
            "forecasting",
            "reconstruction",
            "representation",
            "classification",
        }:
            raise ValueError("Unknown model family.")
        for name, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        self.model = model
        self.family = family
        self.fit_fn = fit_fn
        self.predict_fn = predict_fn
        self.epochs, self.batch_size = epochs, batch_size
        self.learning_rate = learning_rate
        self.loss_fn, self.optimizer_factory = loss_fn, optimizer_factory
        self.device, self.random_state = device, random_state
        self.training_history: list[dict[str, float]] = []
        self._input_shape: tuple[int, ...] | None = None
        self._components: tuple[object, ...] | None = None

    @property
    def is_torch(self) -> bool:
        """Whether the wrapped object inherits from torch.nn.Module."""
        return any(
            base.__name__ == "Module" and base.__module__.startswith("torch.")
            for base in type(self.model).__mro__
        )

    def fit(self, data: Any, target: Any = None, *, verbose: bool = False) -> GenericModel:
        """Fit only the supplied observations; record PyTorch epoch losses."""
        x = as_array(data)
        y = as_array(target, name="target") if target is not None else None
        if y is not None and len(x) != len(y):
            raise ValueError("Input and target observation counts must match.")
        self._input_shape = None
        if self.fit_fn is not None:
            self.fit_fn(self.model, x, y)
        elif self.is_torch:
            self._fit_torch(x, y, verbose=verbose)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(x) if y is None else self.model.fit(x, y)
        else:
            raise TypeError("This model has no fit method; provide fit_fn or use predict directly.")
        self._input_shape = x.shape[1:]
        self._components = component_names(data)
        return self

    def _fit_torch(self, x: np.ndarray, y: np.ndarray | None, *, verbose: bool) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if y is None:
            if self.family != "reconstruction":
                raise ValueError("PyTorch training needs targets except for reconstruction models.")
            y = x
        self.model.to(self.device)
        classification = self.family == "classification"
        if classification:
            if y.shape[1:] != (1,) or not np.equal(y, np.floor(y)).all() or (y < 0).any():
                raise ValueError(
                    "Classification targets must be nonnegative integer class indices."
                )
            y = y[:, 0]
        loss_fn = self.loss_fn or (
            torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
        )
        optimizer = (
            self.optimizer_factory(self.model.parameters())
            if self.optimizer_factory is not None
            else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(y, dtype=torch.long if classification else torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        from anomx.console import progress_bar

        epochs = progress_bar(
            "training", enabled=verbose, iterable=range(self.epochs), unit="epoch"
        )
        self.training_history = []
        self.model.train()
        for epoch in epochs:
            total, count = 0.0, 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                output = self.model(inputs)
                if not isinstance(output, torch.Tensor):
                    raise ValueError("PyTorch module must return a tensor.")
                valid_shape = (
                    output.ndim == 2 and output.shape[0] == targets.shape[0]
                    if classification
                    else output.shape == targets.shape
                )
                if not valid_shape:
                    raise ValueError("PyTorch output shape must exactly match the target shape.")
                loss = loss_fn(output, targets)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Training loss must be a finite scalar.")
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(inputs)
                count += len(inputs)
            self.training_history.append({"epoch": float(epoch + 1), "loss": total / count})
            if verbose:
                epochs.set_postfix(loss=f"{total / count:.6g}")
        self.model.eval()

    def predict(self, data: Any) -> Any:
        """Predict while retaining a DataFrame index or TimeSeries coordinates."""
        x = as_array(data)
        if self._input_shape is not None and x.shape[1:] != self._input_shape:
            raise ValueError("Prediction feature/tensor shape must match the training inputs.")
        if self._components is not None and component_names(data) != self._components:
            raise ValueError(
                "Prediction component names must match the ordered training components."
            )
        if self.predict_fn is not None:
            output = self.predict_fn(self.model, x)
        elif self.is_torch:
            import torch

            self.model.to(self.device).eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(x), self.batch_size):
                    output = self.model(
                        torch.as_tensor(
                            x[start : start + self.batch_size],
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                    if not isinstance(output, torch.Tensor):
                        raise ValueError("PyTorch inference must return a tensor.")
                    chunks.append(output.detach().cpu().numpy())
            output = np.concatenate(chunks)
        elif callable(getattr(self.model, "predict", None)):
            output = self.model.predict(x)
        elif self.family == "representation" and callable(getattr(self.model, "transform", None)):
            output = self.model.transform(x)
        elif callable(self.model):
            output = self.model(x)
        else:
            raise TypeError("Provide a callable, predict method, or predict_fn.")
        return restore_output(
            data,
            output,
            name=self.family,
            preserve_components=self.family not in {"representation", "classification"},
        )

    def save(self, path: str) -> None:
        """Serialize this adapter; only load trusted pickle files."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    def load(self, path: str) -> GenericModel:
        """Replace adapter state from a trusted file written by ``save``."""
        with Path(path).open("rb") as handle:
            loaded = pickle.load(handle)
        if not isinstance(loaded, type(self)):
            raise TypeError("Saved object is not the same model type.")
        self.__dict__.update(loaded.__dict__)
        return self
