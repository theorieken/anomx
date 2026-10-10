# Architecture is code

Anomx Work accepts ordinary Python. The platform does not constrain training to
a dropdown or one baseline network. Define a `torch.nn.Module`, use a library's
model class, or compose data preparation, training, inference and scoring stages.
Architecture, loss, optimizer, feature transforms and training loop belong in the
saved code; useful tunable values belong in inputs. The compact bundled template
is a starting point, not a required architecture.

## Plan before allocating compute

Record input/output tensor contracts, history/horizon, covariates and availability
at inference, target semantics, checkpoint rule and expected resource needs. A
future covariate must truly be known at prediction time; future target values are
not valid covariates. Select actual available CPU/GPU/Maxwell resources and confirm
dependencies in that runtime. A package installed in the agent or task worker is
not necessarily installed in remote compute.

Heavy libraries remain optional. `anomx[darts]`, `anomx[torch]` and `anomx[ml]`
provide optional integrations; pin versions for a reproducible managed runtime.
Do not run `pip install` every five minutes inside the training loop. Provision the
managed environment once through the authorized runtime setup, then record package
versions in model metadata. Never silently replace an unavailable requested model
with another architecture and describe it as the requested one.

## Custom PyTorch

Replace the template's network with a class such as a temporal CNN, Transformer,
attention model or a domain-specific architecture. The training procedure can use
any matching tensor contract. Match loss to targets, assert output shapes, retain
the best validation checkpoint and log actual metrics. Export and compare a real
batch before publication. The optional `anomx.models.PyTorchModel` adapter accepts
a user-defined module with `(batch, history, channels)` to
`(batch, horizon, channels)`; its convenience loop does not provide every advanced
training/covariate feature. Use a custom loop when needed.

## Darts TiDE and Temporal Fusion Transformer

Darts provides `TiDEModel` and `TFTModel`; they remain ordinary Python classes in
a compute block. Read the documentation matching the installed Darts version and
inspect constructor capabilities before using version-specific options. For example:

```python
from darts.models import TiDEModel, TFTModel
import torch

# Illustrative configurations; adapt lengths and sizes to inspected data.
tide = TiDEModel(input_chunk_length=24, output_chunk_length=6,
                 hidden_size=32, n_epochs=20, random_state=42)
tft = TFTModel(input_chunk_length=24, output_chunk_length=6,
               hidden_size=16, num_attention_heads=4,
               add_relative_index=True, likelihood=None,
               loss_fn=torch.nn.MSELoss(), n_epochs=20, random_state=42)
```

Convert well-ordered segments to Darts TimeSeries and fit a scaler on training
only. Pass validation series/covariates explicitly. Use the Lightning callback
namespace expected by the installed Darts release to forward train/validation
metrics to `work.metric` at epoch boundaries. Check `trainer.sanity_checking` to
avoid recording pre-training validation as an epoch. Record numeric finite scalars
from callback metrics; do not silently relabel a quantile loss as MSE. Fit separate
segments as a list instead of filling long acquisition gaps with invented data.

`model.to_onnx(path)` is available for supported Darts torch models, but export and
forecast inference have version- and architecture-specific requirements. TFT may
have multiple inputs for static/past/future covariates. Read the exported input
schema; do not assume a single `history` input. Verify ONNX output against the
selected trained checkpoint, using the documented preparation for that installed
version. Library export documentation is not proof that a particular graph works.
If an operator is unsupported, try a justified export configuration or wrapper
within the user's scope; report the exact limitation rather than publishing an
incorrect surrogate or marking an unverified model ready.

Official references (consult the installed-version documentation):
- [TiDE](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tide_model.html)
- [TFT](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html)
- [Torch models and callbacks](https://unit8co.github.io/darts/userguide/torch_forecasting_models.html)

The execution concept is architecture-independent; portable publication still
requires a valid self-contained ONNX graph in the current platform. Unsupported
export is a concrete artifact limitation, not a reason to restrict every job to
the small example MLP.
