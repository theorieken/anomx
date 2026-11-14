# Package source ownership

Read [repository guidance](../../AGENTS.md). Paths here are relative to `anomx/`.

| Location | Responsibility |
| --- | --- |
| `base/` | Shared component metadata, signatures, capabilities and discovery. |
| `data/` | Dataset/TimeSeries containers, sources, views, loaders and processing. |
| `models/` | Inference/training contracts and forecasting/reconstruction/representation implementations. |
| `detection/` | Scorers, detectors, detection pipelines and results. |
| `execution/` | Portable jobs/work graphs, execution context, run tracking and local backends. |
| `integrations/` | Optional external adapters, platform HTTP access and workspaces. |
| `agent/` | Independent CLI/agent application within this distribution. |
| `metrics/` | Native metric implementations and optional plotting. |
| `helpers/utils/` | Small library-wide utilities with real cross-domain consumers. |
| `_series_utils/`, `_series_typing.py`, `_series_logging.py` | Attributed native TimeSeries support; preserve notices and contracts. |
| `cli.py`, `__main__.py`, `console.py` | Public CLI dispatch and console entry points. |

Data does not depend on models, detection, execution or the agent. Models consume
data; detection consumes observations/model outputs; execution composes work.
Integrations adapt external contracts through explicit clients. Scientific core
imports must not eagerly load the agent, Django, Darts, Torch or ONNX.

Keep domain contracts in that domain's `base`; shared metadata in top-level
`base`. Do not introduce a universal object superclass. Private helpers stay
with their owner unless actual consumers justify a shared utility.

When moving code, update lazy export maps, stubs, discovery, docs, examples,
agent skill resources and platform consumers. Preserve public import paths or
explicitly coordinate an API migration. Do not add wrappers merely to avoid
updating known internal consumers.
