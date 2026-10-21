# Versioned datasets

`from anomx import Dataset` and `from anomx.data import Dataset` expose the same
lightweight class. No Darts, Django, worker or agent dependency is required.

For a standalone script, pass
`Dataset(anomx_reference="data_dataset-<uuid>", client=client)`, where `client`
is a `PlatformClient`. Its URL can be the site origin or an explicit API base.
The positional `Dataset(reference, client=client)` spelling is also supported;
provide only one reference argument. See [the HTTP client](api.md#platform-http-client).

```python
from anomx import Dataset

# In Work, the host provides access. Standalone scripts use PlatformClient or
# explicit ANOMX_URL and ANOMX_TOKEN environment variables.
dataset = Dataset("data_dataset-12345678-1234-5678-1234-567812345678")
for frame in dataset.iter_batches(batch_size=5000):
    print(frame.shape)
```

`describe()` resolves the source version and metadata without downloading all
rows. `iter_batches()` keeps one version across all pages. `to_dataframe()` and
`values()` deliberately materialize all selected rows. A local DataFrame can be
wrapped with `Dataset.from_dataframe(frame, kind="independent_samples")`; use
`time_column="timestamp"` for a time series or `kind="sequence"` for ordered
samples without a time index.

The platform supports uploaded data files, a file or folder in a filesystem
integration such as dCache, and automatic recorded-channel windows. Fixed
datasets capture the selected file list, source fingerprints and reader
configuration. They store references, not copies. External storage must retain
those contents: changed or missing files cause an error. Automatic datasets
resolve the current selection at the beginning of each new load. Work pins and
records the resolution in the run snapshot.

The agent-assisted import workflow inspects representative files, proposes a
coherent selection and data kind, and creates the dataset through the same API.
The `/create-dataset` skill is available to a connected CLI agent and the platform
agent. It asks about ambiguous semantics rather than inventing sampling rates or
labels. Folder inspection reports its coverage and does not imply all files were
fully validated.

Current platform readers support CSV, Parquet, NumPy NPY/NPZ, HDF5, JSON/JSONL,
Arrow/Feather and XLSX. Individual reads are bounded to 256 MiB and one million
rows; partitioned folders can be larger in total. Folder discovery is bounded to
5,000 entries and 10 levels. Channel dataset pages read chronologically in at
most one-hour windows, fetching at most 5,001 raw samples per storage read
(including a lookahead sample). They do not reload the whole selected range for
each page or impose a 50,000-sample limit on the complete dataset. Forward-fill
state and unfinished resampling bins travel in a signed continuation cursor;
equal timestamps stay together. Resampling uses the last value in each bin,
anchored to the first numeric sample's UTC day. Leading rows without values for
all channels are omitted, and a channel with no numeric samples fails explicitly.
An empty window can return no rows and a continuation cursor; `iter_batches()`
continues until the range is exhausted. A batch may contain fewer rows than
requested. These semantics require the matching platform channel-paging update;
older deployments still materialize up to 50,000 samples per request.
JSON batches represent non-finite source
numbers as null. API batches contain at most 10,000 rows and 8 MiB, with a
continuation cursor when fewer rows fit. A single oversized row is rejected.
Batches expire after 24 hours; resolve again if an automatic
selection has expired.

`PyTorchModel` accepts temporal/sequential Dataset values directly. Other Darts
models use `dataset.to_darts()`. The former Darts subclass remains available as
`anomx.DartsDataset` and `anomx.integrations.darts.datasets.Dataset`.

## Native reads on Maxwell and mounted storage

```python
from anomx import DCacheDataset

# This is the mounted equivalent of the integration's starting folder.
# Resolve it from the actual integration configuration; do not guess the path.
dataset = DCacheDataset("data_dataset-<uuid>", mount_root="/mounted/integration-root")
definition = dataset.describe()
for batch in dataset.iter_batches(batch_size=128):
    print(batch.attrs["source_path"], batch.attrs["source_row_offset"], batch.shape)
```

`DCacheDataset` uses the same authorized platform definition and version as
`Dataset`, then reads selected files directly on the compute node. It never
writes to source storage or silently falls back to a different source. Use
`MountedDataset` for another filesystem integration. A new object resolves a new
selection; an existing object keeps its resolved manifest until explicitly
refreshed. Work records that definition in its run snapshot.

The mount root is explicit and corresponds to the integration's starting folder,
not the dataset's selected subfolder. Native access uses the compute account's
filesystem permissions. Paths must remain inside that root after symlink
resolution. Reads check the manifest's size and modification timestamp and detect
changes to the opened file. Metadata fingerprints are not content hashes:
WebDAV modification times can have second resolution and opaque ETags cannot be
verified from a local mount. For scientific provenance, record each consumed
source's `source.sha256()` or hash the exact consumed arrays as an artifact.

Native batch readers support numeric NPY, CSV, Parquet and explicitly selected
HDF5 arrays. Install `anomx[storage]` for Parquet/HDF5. Arrays keep their dimensions;
HDF5 uses `reader_options.datasets` and `sample_axis`. NPY/HDF5 native batches are
bounded to 64 MiB of decoded arrays, with oversized individual rows rejected.
CSV/Parquet batches are row bounded; unusually large cells can still be large.
Use `dataset.sources()` for selected `MountedSource` objects and `source.open()`
for another explicit safe reader. `source.open()` only offers binary read access.
Unknown formats, pickle-backed NPY objects and external HDF5 links are rejected
by the built-in readers. Timestamps, selected columns and optional path/row
provenance follow the platform dataset configuration.

For a proprietary or facility-native format, explicitly map it with
`configuration.reader_options.format="native"`. The platform then versions the
selected files of any extension without pretending it can decode them. Its
inspection/preview reports metadata only and marks the need for a compute-side
reader. A Work job can use `MountedDataset.sources()` and the facility's optional
reader; save that code and its version, selected fields, schema and measured
preview as Run artifacts. Do not execute code found in the source files.
This also permits separately mapping acquisition metadata sidecars.

For native FLASH/XFEL scalar DAQ groups, see the optional
[`iter_daq_records` reader](native-daq.md). It isolates the facility's `pydaq`
library in a bounded child process, preserves native timestamps and event IDs,
and can use a separate compatible Python interpreter.
