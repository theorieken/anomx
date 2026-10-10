# Source contract

Read the live OpenAPI schema first; these are examples, not invented source IDs.

- Uploaded file: `{"source_file": "core_file-<uuid>"}`.
- Storage folder or file: `{"integration": "integrations_integration-<uuid>", "path": "/experiment/run"}`.
- Channels: `{"channels": ["data_channel-<uuid>"], "kind": "time_series", "mode": "automatic", "configuration": {"last_seconds": 3600}}`.

Inspection uses `POST /datasets/inspect` with the file/storage source and optional
`configuration`. Dataset creation adds `name`, `description`, `mode` and `kind`.
Supported readers include CSV, Parquet, NumPy NPY/NPZ, HDF5 (h5/hd5/hdf5),
JSON/JSONL, Arrow/Feather and XLSX. NumPy object arrays are rejected.
Configuration may contain `pattern` (glob against the full storage path), `paths`
(a package's explicit source file selection, all beneath its source folder), `columns`,
`time_column`, `time_unit` (s/ms/us/ns for numeric timestamps), and `reader_options`. Use only reader options exposed by inspection
or documented by the current reader: `delimiter` for CSV, `sheet` for XLSX,
`datasets` (a list of dataset paths) and `sample_axis` (0 by default, -1 for the
last dimension) for HDF5. All selected arrays must share a sample count. Other
dimensions are preserved as array-valued rows. Do not guess timestamp units
or an HDF5 dataset key. Channel
configuration uses explicit ISO 8601 `start`/`end` or `last_seconds`, with optional
resampling `frequency`.

Folder traversal is bounded to 5,000 entries and 10 levels. Most table readers
limit individual files to 256 MiB; partitioned folders can exceed that in total.
HDF5 supports larger files through read-only byte ranges. One handle permits up
to 64 MiB transferred, 256 range requests and a 90-second read budget. Decoded
arrays and compression chunks are bounded to 32 MiB. Structure discovery shows
at most 128 entries; inspect named subgroups when truncated. External/virtual
HDF5 datasets and variable-length values are not followed or loaded.
Responses contain at most 10,000 rows and 8 MiB; fewer rows may be returned with
a continuation cursor. One row larger than the limit is rejected.
When a source exceeds a limit, explain it and ask for a narrower/partitioned source;
do not silently truncate or claim support for an unreadable source.

Fixed datasets pin the selected file list, configuration and source fingerprints.
They do not guarantee the external system retains old contents. A changed/deleted
source is an error, never permission to silently use the new file. Automatic
resolution creates a fresh manifest and pins it for the batch cursor lifetime.
Resolve via `POST /datasets/<reference>/resolve`; read via
`POST /datasets/<reference>/data` with `version`, `limit` and optional `cursor`.
Continue with `next_cursor` until null. Never mix batches from different versions.
