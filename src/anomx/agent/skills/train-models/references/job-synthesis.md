# Synthesize a complete anomaly-detection job

A Job is a configurable research or operational procedure. This framework is a
checklist for reasoning and evidence, not a fixed pipeline implementation.
Choose, omit or compose operations according to the task and inspected data.
Explain omissions in the saved procedure. Use editable Work modules and explicit
inputs; never hide facility paths, model choices or cuts in a special endpoint.

## Acquisition and source mapping

Create and verify a versioned Dataset before training. Preserve channel identity,
timestamps or sequence IDs, units and uncertainty, file/row provenance, sampling
rates and available operating-state annotations. Sequence position is not a
timestamp; an operating state is not automatically a fault label.

On compute with a native dCache mount, use `DCacheDataset(reference,
mount_root=...)` from `anomx`. Resolve the actual integration starting folder and
its mounted equivalent. The resolved `source_manifest` and hash must match the
Run's dataset snapshot. Read via `iter_batches` or `sources()[i].open()`, not a
separately invented glob. The mounted reader uses the compute account's existing
access and never writes to sources. Record content hashes for consumed
sources/arrays when needed; metadata versions are not byte-level checksums.

## Data processing

- **Preprocessing:** verify shapes, rates, gaps, sentinels, units and alignment.
  Declare exclusions before comparing models; retain rejection counts/reasons
  and failures. Never compress time by silently dropping missing samples.
  Split by time, recording, system or subject as the question requires, before
  fitting normalization or imputation.
- **Feature extraction:** choose raw sequences, spectra, statistics or learned
  representations. State their meaning, window duration, transform and
  dimensions. Spectral features must resolve relevant frequencies; array size
  alone does not establish that.
- **Feature selection:** fit selection on training only and retain feature
  identifiers. If all features are retained, say so. Keep test data out of
  feature and model decisions.

## Model construction

Choose forecasting, reconstruction or representation. Compare architectures on
the same objective, target, splits and preprocessing where possible. If objectives
differ, report metrics separately rather than ranking incommensurate losses.
Include an inexpensive meaningful baseline.

Keep architecture, optimization, seed, budgets and early stopping in Work inputs.
Record actual trainable parameters, completed epochs, validation-selected
checkpoints, fit time and execution environment. Use repeated seeds for claims
that depend on training variation. Equal epochs do not imply equal compute.

## Evaluation and selection

Use validation for selection and an independent calibration split for thresholds
when appropriate. Evaluate test data after these choices are frozen. Log
aggregate metrics and per-window scores with source/split identifiers. Preserve
predictions/reconstructions, labels and their provenance, residuals and transforms.

For labeled data, distinguish window from event metrics; document matching,
latency and imbalance. For unlabeled data, report reconstruction/forecast quality,
stability, operating-context associations and threshold exceedance. Do not call
these fault accuracy, precision, recall or false-alarm rates. Injected anomalies
are a separate sensitivity experiment, never observed faults.

## Artifacts and inspection

Use `work.metric`/`anomx.log` for scalar step histories and `work.artifact` for
split/QC tables, configuration, transforms, curves, predictions and checkpoints.
Publish via the supported portable model contract after export/parity checks.
A raw checkpoint is an artifact, not automatically a registered Model. Inspect
the completed Run, metrics, artifacts and downloadable export. Upload failure
remains a failed Run even if fitting completed; preserve history during recovery.

## Deployment and feedback

Deployment is another Job. Separate Predictions, Human/Agent-in-the-Loop review,
and Decisions & Effects. Predictions retain model/data/time/uncertainty context.
Reviewed labels retain author, rationale and version. Agent suggestions are not
independent ground truth. Incorporate approved labels into new dataset/training
versions and reevaluate before promotion. Activate recurrence or physical effects
only within the requested scope and after the relevant quality gate.

## Persist the decisions

Save `procedure.json` with dataset references/versions, task, targets, semantics,
splits, preprocessing, feature extraction/selection, models/budgets, metrics,
calibration, deployment policy and unresolved assumptions. Explain omitted
optional stages. This describes actual code and inputs; it does not replace
them or turn a planned step into verified completion.
