# Native DAQ files on a mounted dataset

`anomx.data.sources.daq.iter_daq_records` is an optional reader for explicitly selected
scalar `A_TS_GSPECTRUM` subchannels in native FLASH/XFEL DAQ files. It uses the
facility-provided `pydaq` library. Anomx does not install that native dependency.

Map the source files as a versioned external dataset, with a native format when
the platform should retain file metadata rather than previewing file contents.
The mount root is the local counterpart of the integration root:

```python
from contextlib import closing

from anomx.data.sources.mounted import DCacheDataset
from anomx.data.sources.daq import iter_daq_records

dataset = DCacheDataset(
    "data_dataset-example",
    mount_root="/mounted/integration-root",
)
for source in dataset.sources():
    with closing(iter_daq_records(
        source,
        channel="FACILITY/SYSTEM.DAQ/EXPLICIT.GROUP",
        fields={"measurement": 7, "actuation": 31},
        experiment="experiment-name",
        linac="XFEL",
        python_executable="/usr/bin/python3",
        environment={
            "PYTHONPATH": "/facility/native-library-directory",
            "LD_LIBRARY_PATH": "/facility/native-library-directory",
            "PYDAQ_NO_CHANNEL_DESCRIPTIONS": "1",
        },
        batch_size=128,
        timeout_seconds=60,
        limit=None,
    )) as records:
        for record in records:
            print(record.timestamp, record.event_id, record.values)
```

Names and indices above are examples, not discovered schema or defaults.
Inspect the source's native channel/subchannel descriptions first. The reader
preserves the returned native comment/name, index and type in `field_metadata`.
It does not infer physical units, convert timestamps, calibrate measurements,
join channels, interpret status codes or choose scientific windows/splits.
NaN/infinite scalar values become `None`, with a reason in `invalid_fields`.
Absent requested subchannels, inconsistent field timestamps/event IDs and
unsupported shapes/types raise `DaqReadError`.

One child process is reused for the complete file/channel request. Batches are
transferred over a bounded NDJSON protocol; this is not a subprocess per event.
The child is stdlib-only apart from `pydaq`, so it can use a facility Python ABI
independent of the ML environment. Native stdout is redirected away from the
protocol and diagnostics retain at most 16 KiB. Messages are limited to 8 MiB,
the parent queue holds at most two messages, and stalled reads have a configurable
progress timeout covering interpreter startup and request transmission as well.
Reduce batch size if the selected metadata exceeds a message
limit. Explicitly closing the iterator terminates and reaps its child.

`MountedSource.open()` remains active throughout the native read, checking
the manifest's size/modification time and opened-file identity before and after.
The child also checks file identity. These checks are not a content checksum;
use `source.sha256()` when the study requires a content fingerprint. A consumer
can receive earlier records before a later source-change/read error, so derived
outputs must remain provisional until the iterator completes successfully.
No source files are changed by the reader.

The initial adapter intentionally covers the verified grouped scalar format.
Other native payloads such as images, waveforms or ungrouped slow-channel
records require an explicit supported adapter rather than automatic flattening.
Configuration streams may have different cadences from monitor streams; align
them with an application-defined validity policy outside this reader.
