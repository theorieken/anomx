# Versioned datasets

`from anomx import Dataset` and `from anomx.data import Dataset` expose the same
lightweight class. No Darts, Django, worker or agent dependency is required.

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
5,000 entries and 10 levels. Channel windows must fit 50,000 raw samples and are
aligned using the recorded-data loader. JSON batches represent non-finite source
numbers as null. API batches contain at most 10,000 rows and 8 MiB, with a
continuation cursor when fewer rows fit. A single oversized row is rejected.
Batches expire after 24 hours; resolve again if an automatic
selection has expired.

`PyTorchModel` accepts temporal/sequential Dataset values directly. Other Darts
models use `dataset.to_darts()`. The former Darts subclass remains available as
`anomx.DartsDataset` and `anomx.datasets.Dataset`.
