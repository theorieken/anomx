# Model domain

Read [source guidance](../AGENTS.md). `base/` defines model/normality/classifier
contracts. `forecasting/`, `reconstruction/` and `representation/` own concrete
implementations; `results.py` owns typed outputs.

- Follow existing capabilities and signatures. A generic Model needs prediction;
  trainable NormalityModel adds training/persistence. Do not require every model
  to implement irrelevant operations.
- Preserve shape/index alignment, fitted state and persistence metadata. Keep
  training separate from inference and avoid fitting on evaluation inputs.
- Keep Torch/ONNX/Darts optional and lazy. Provider/framework adapters do not
  justify framework imports across the scientific core.
- Preserve component keys and discoverable public names. Update exports, stubs,
  model docs, examples and platform catalog consumers together.
- Product scheduling, tenancy, permissions and model artifact persistence belong
  to the platform, not these reusable classes.
