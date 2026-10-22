"""Optional native LSTM module. Copy to lstm_model.py in the job workspace.

Import only on compute when architecture=lstm. Fitting, chronological windows
and recursive prediction are supplied by Anomx's TorchForecastingModel.
"""

from __future__ import annotations

from typing import Any

import torch

from anomx.models import TorchForecastingModel


class LSTMForecast(torch.nn.Module):
    def __init__(self, channels: int, horizon: int, hidden_size: int, layers: int) -> None:
        super().__init__()
        self.channels, self.horizon = channels, horizon
        self.recurrent = torch.nn.LSTM(channels, hidden_size, layers, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, horizon * channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(values)
        return self.output(sequence[:, -1]).reshape(-1, self.horizon, self.channels)


def create_model(
    inputs: dict[str, Any], lag: int, horizon: int, channels: int
) -> TorchForecastingModel:
    seed = int(inputs.get("seed", 42))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = LSTMForecast(
            channels, horizon, int(inputs.get("hidden_size", 16)), int(inputs.get("layers", 1))
        )
    return TorchForecastingModel(
        input_chunk_length=lag,
        output_chunk_length=horizon,
        module=module,
        n_epochs=int(inputs.get("epochs", 10)),
        batch_size=int(inputs.get("batch_size", 32)),
        learning_rate=float(inputs.get("learning_rate", 0.001)),
        random_state=seed,
        device=inputs.get("device", "cpu"),
        verbose=True,
    )
