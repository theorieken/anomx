"""Editable compute-block template. Inspect data and set the explicit inputs first.

Required: dataset_reference, target_column, model_name, lookback, horizon.
Ordering: order_column OR ordered_rows_confirmed=true from verified source order.
Optional: architecture=mlp/transformer; segment_column; max_gap_seconds;
target_reduction=scalar/mean/rms/last; epochs; seed.
This compact baseline is an example; replace the network with user-defined code.
"""

import copy
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from anomx import Dataset

seed = int(inputs.get("seed", 42))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)
dataset = Dataset(inputs["dataset_reference"])
definition = dataset.describe()
frames = []
row_count = 0
for batch in dataset.iter_batches(batch_size=100):
    row_count += len(batch)
    if row_count > int(inputs.get("max_rows", 100000)):
        raise ValueError("Dataset exceeds the configured bounded training size.")
    frames.append(batch.reset_index() if dataset.kind == "time_series" else batch)
if not frames:
    raise ValueError("Dataset contains no readable observations.")
frame = pd.concat(frames, ignore_index=True)
# iter_batches pins its resolution. Read it again after loading for exact provenance.
definition = dataset.describe()
target_column = inputs["target_column"]
reduction = inputs.get("target_reduction", "scalar")


def reduce_target(value):
    values = np.asarray(value, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        return np.nan
    if reduction == "scalar":
        if values.size != 1:
            raise ValueError("Array target requires an explicitly justified reduction.")
        return float(values.reshape(-1)[0])
    if reduction == "mean":
        return float(values.mean())
    if reduction == "rms":
        return float(np.sqrt(np.mean(values**2)))
    if reduction == "last":
        return float(values.reshape(-1)[-1])
    raise ValueError("Unsupported target reduction.")


frame["__target"] = frame[target_column].map(reduce_target)
segment_column = inputs.get("segment_column")
order_column = inputs.get("order_column")
if inputs.get("max_gap_seconds"):
    if not order_column or segment_column:
        raise ValueError(
            "Automatic gap segmentation requires an order column and no supplied segment column."
        )
    frame[order_column] = pd.to_datetime(frame[order_column], utc=True, errors="raise")
    frame = frame.sort_values(order_column, kind="stable")
    delta = frame[order_column].diff().dt.total_seconds()
    if (delta <= 0).any():
        raise ValueError("Duplicate or non-increasing sample timestamps.")
    frame["__segment"] = (delta > float(inputs["max_gap_seconds"])).cumsum()
    segment_column = "__segment"
if order_column:
    if frame[order_column].isna().any():
        raise ValueError("Ordering contains missing values.")
    frame = frame.sort_values(order_column, kind="stable")
elif inputs.get("ordered_rows_confirmed") is not True:
    raise ValueError("Inspect and establish row order before forecasting.")
# Split the complete timeline before grouping gaps/segments. Per-segment splits
# would let later training data influence evaluation of earlier segments.
first_cut, second_cut = len(frame) * 70 // 100, len(frame) * 85 // 100
if not 0 < first_cut < second_cut < len(frame):
    raise ValueError("Dataset is too short for chronological splits.")
if order_column:
    first_order = frame[order_column].iloc[first_cut]
    second_order = frame[order_column].iloc[second_cut]
    frame["__split"] = (frame[order_column] >= first_order).astype(int) + (
        frame[order_column] >= second_order
    ).astype(int)
else:
    frame["__split"] = (np.arange(len(frame)) >= first_cut).astype(int) + (
        np.arange(len(frame)) >= second_cut
    ).astype(int)
groups = list(frame.groupby(segment_column, sort=False)) if segment_column else [("all", frame)]
lookback, horizon = int(inputs["lookback"]), int(inputs["horizon"])
if min(lookback, horizon) < 1:
    raise ValueError("Lookback and horizon must be positive sample counts.")
splits = [[], [], []]
split_details = []
for key, group in groups:
    if order_column:
        if group[order_column].isna().any() or group[order_column].duplicated().any():
            raise ValueError("Ordering is missing or ambiguous within a segment.")
        group = group.sort_values(order_column, kind="stable")
    values = group["__target"].to_numpy(dtype=np.float32)
    counts = [int((group["__split"] == index).sum()) for index in range(3)]
    boundaries = [0, counts[0], counts[0] + counts[1], len(values)]
    split_details.append({"segment": str(key), "rows": len(values), "boundaries": boundaries})
    for index in range(3):
        splits[index].append(values[boundaries[index] : boundaries[index + 1]])
train_values = np.concatenate(splits[0])
mean, scale = float(np.nanmean(train_values)), float(np.nanstd(train_values))
if not np.isfinite([mean, scale]).all() or scale <= 1e-8:
    raise ValueError("Training target has no usable finite variation.")


def make_windows(parts):
    xs, ys, skipped = [], [], 0
    for values in parts:
        for target in range(lookback + horizon - 1, len(values)):
            stop = target - horizon + 1
            window = values[stop - lookback : stop]
            if not np.isfinite(window).all() or not np.isfinite(values[target]):
                skipped += 1
                continue
            xs.append((window - mean) / scale)
            ys.append((values[target] - mean) / scale)
    if len(xs) < 10:
        raise ValueError(
            "A split has fewer than ten complete windows; use more data or smaller windows."
        )
    return torch.tensor(np.stack(xs)), torch.tensor(ys, dtype=torch.float32).reshape(-1, 1), skipped


train_x, train_y, skipped_train = make_windows(splits[0])
val_x, val_y, skipped_val = make_windows(splits[1])
test_x, test_y, skipped_test = make_windows(splits[2])
work.log(
    "Prepared chronological, disjoint windows",
    train=len(train_x),
    validation=len(val_x),
    test=len(test_x),
    skipped=skipped_train + skipped_val + skipped_test,
)


class AttentionBlock(torch.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.norm1, self.norm2 = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
        self.qkv = torch.nn.Linear(width, width * 3)
        self.projection = torch.nn.Linear(width, width)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(width, width * 2), torch.nn.GELU(), torch.nn.Linear(width * 2, width)
        )

    def forward(self, sequence):
        batch, length, width = sequence.shape
        q, k, v = self.qkv(self.norm1(sequence)).chunk(3, dim=-1)
        q, k, v = [
            value.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
            for value in (q, k, v)
        ]
        weights = torch.softmax(q @ k.transpose(-2, -1) / self.head_width**0.5, dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        sequence = sequence + self.projection(attended)
        return sequence + self.feedforward(self.norm2(sequence))


class TemporalTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        width, heads = int(inputs.get("hidden_size", 24)), int(inputs.get("attention_heads", 4))
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("Transformer width must be divisible by its attention head count.")
        self.embedding = torch.nn.Linear(1, width)
        self.position = torch.nn.Parameter(torch.randn(1, lookback, width) * 0.02)
        self.blocks = torch.nn.Sequential(
            *[AttentionBlock(width, heads) for _ in range(int(inputs.get("layers", 2)))]
        )
        self.output = torch.nn.Linear(width, 1)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, history):
        sequence = self.blocks(self.embedding(history.unsqueeze(-1)) + self.position)
        return history[:, -1:] + self.output(sequence[:, -1])


architecture = inputs.get("architecture", "mlp")
if architecture not in {"mlp", "transformer"}:
    raise ValueError(
        "Select mlp or transformer, or adapt the template to the requested architecture."
    )
model = (
    TemporalTransformer()
    if architecture == "transformer"
    else torch.nn.Sequential(torch.nn.Linear(lookback, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
)


def predict(values):
    with torch.no_grad():
        return torch.cat(
            [model(batch) for batch in values.split(int(inputs.get("batch_size", 64)))]
        )


optimizer = torch.optim.AdamW(model.parameters(), lr=float(inputs.get("learning_rate", 0.001)))
criterion = torch.nn.MSELoss()
best_loss, best_epoch, best_state = float("inf"), 0, None
for epoch in range(1, int(inputs.get("epochs", 20)) + 1):
    model.train()
    ordering = torch.randperm(len(train_x))
    for offset in range(0, len(ordering), int(inputs.get("batch_size", 64))):
        indices = ordering[offset : offset + int(inputs.get("batch_size", 64))]
        optimizer.zero_grad()
        loss = criterion(model(train_x[indices]), train_y[indices])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = float(criterion(predict(train_x), train_y))
        val_loss = float(criterion(predict(val_x), val_y))
    work.metric("train_loss", train_loss, step=epoch)
    work.metric("validation_loss", val_loss, step=epoch)
    print(f"Epoch {epoch}: train={train_loss:.6f}, validation={val_loss:.6f}")
    if val_loss < best_loss:
        best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
if best_state is None:
    raise ValueError("Training produced no finite validation checkpoint.")
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    prediction = predict(test_x).numpy().ravel() * scale + mean
truth = test_y.numpy().ravel() * scale + mean
persistence = test_x[:, -1].numpy() * scale + mean
metrics = {
    "test_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
    "test_mae": float(np.mean(np.abs(prediction - truth))),
    "persistence_rmse": float(np.sqrt(np.mean((persistence - truth) ** 2))),
    "best_validation_loss": best_loss,
    "best_epoch": best_epoch,
    "train_windows": len(train_x),
    "validation_windows": len(val_x),
    "test_windows": len(test_x),
}
with torch.no_grad():
    validation_errors = (predict(val_x) - val_y).numpy().ravel() * scale
validation_model_mse = float(np.mean(validation_errors**2))
validation_persistence_mse = float(torch.mean((val_x[:, -1:] - val_y) ** 2)) * scale**2
quality_gate = validation_model_mse < validation_persistence_mse * float(inputs.get("quality_ratio", 0.95))
threshold = float(
    np.quantile(np.abs(validation_errors), float(inputs.get("anomaly_quantile", 0.99)))
)
metrics.update(
    validation_persistence_mse=validation_persistence_mse,
    validation_model_mse=validation_model_mse,
    validation_rmse=float(np.sqrt(validation_model_mse)),
    validation_persistence_rmse=float(np.sqrt(validation_persistence_mse)),
    quality_gate_passed=float(quality_gate),
    validation_residual_threshold=threshold,
)
for name, value in metrics.items():
    work.metric(name, float(value))


class RawForecast(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, history):
        return self.network((history - self.mean) / self.scale) * self.scale + self.mean


export = RawForecast(model).eval()
example = test_x[:3] * scale + mean
artifact = Path("forecast.onnx")
torch.onnx.export(
    export,
    (example,),
    str(artifact),
    input_names=["history"],
    output_names=["forecast"],
    dynamic_axes={"history": {0: "batch"}, "forecast": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)
import onnxruntime

session = onnxruntime.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
with torch.no_grad():
    expected = export(example).numpy()
np.testing.assert_allclose(
    session.run(None, {"history": example.numpy()})[0], expected, rtol=1e-4, atol=1e-5
)
metadata = {
    "framework": "pytorch",
    "task": "forecasting",
    "architecture": architecture,
    "quality_gate": {
        "passed": quality_gate,
        "rule": "validation MSE < persistence MSE * quality_ratio",
        "ratio": float(inputs.get("quality_ratio", 0.95)),
    },
    "anomaly_threshold": {
        "value": threshold,
        "basis": "validation absolute residual quantile",
        "quantile": float(inputs.get("anomaly_quantile", 0.99)),
    },
    "parameters": dict(inputs),
    "dataset_version": definition["version"],
    "target_sha256": hashlib.sha256(
        frame["__target"].to_numpy(dtype=np.float32).tobytes()
    ).hexdigest(),
    "normalization": {
        "mean": mean,
        "scale": scale,
        "fitted_on": "training only",
        "embedded_in_onnx": True,
    },
    "split": split_details,
    "split_policy": "global chronological 70/15/15; windows never cross segments or splits",
    "packages": {
        name: importlib.metadata.version(name) for name in ("anomx", "torch", "onnx", "onnxruntime")
    },
}
Path("training-metadata.json").write_text(json.dumps(metadata, indent=2))
published = work.publish_model(
    artifact, name=inputs["model_name"], metrics=metrics, metadata=metadata
)
preview = [
    {
        "observed": float(truth[i]),
        "forecast": float(prediction[i]),
        "persistence": float(persistence[i]),
    }
    for i in np.linspace(0, len(truth) - 1, min(20, len(truth)), dtype=int)
]
result = {
    "model": published,
    "metrics": metrics,
    "dataset_version": definition["version"],
    "preview": preview,
}
