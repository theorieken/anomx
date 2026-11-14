# Detection domain

Read [source guidance](../AGENTS.md). `base/` owns scorer/detector contracts;
`scorers/` converts observations/residuals/outputs into scores;
`detectors/` converts evidence into thresholds/decisions; `pipeline.py`
composes stages and `results.py` defines inspectable results.

Preserve score direction, index/timestamp alignment, thresholds, labels, missing
values and metadata. Keep fit/online state explicit and avoid train/evaluation
leakage. A scorer should not grow platform alert delivery or persistence.

Use native data/model contracts and optional adapters for external ecosystems.
Update public exports/stubs and component discovery for new implementations;
preserve existing keys and result shapes.
