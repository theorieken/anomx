# Third-party notices

## Unit8 Darts native time-series baseline

Parts of Anomx are derived from [Unit8 Darts](https://github.com/unit8co/darts),
Copyright 2019 Unit8 SA, with additional contributions by the Darts contributors, licensed under the
Apache License, Version 2.0. The complete license is reproduced in
[licenses/DARTS-APACHE-2.0.txt](licenses/DARTS-APACHE-2.0.txt).

Source revision: `dcdde2414f07f13b3bbb095428d9c6cc41492588` (upstream package
version `0.48.1`). This revision is pinned for provenance; it is not a runtime
dependency. No upstream NOTICE file was present at this revision.

The adapted baseline includes `src/anomx/data/timeseries.py`,
`src/anomx/data/processing/`, `src/anomx/metrics/`, `src/anomx/_series_utils/`,
`src/anomx/_series_logging.py`, `src/anomx/_series_typing.py`,
`src/anomx/config.py`, and the explicitly attributed upstream regression tests
in `tests/timeseries_baseline/`. File headers identify individual source paths.
Existing Anomx implementations in those directories retain their own origins.

Changes include native Anomx imports and API placement, optional integration
boundaries, an Anomx plotting configuration, and composition with Anomx datasets
and model interfaces. The source and documentation are maintained in this
repository so they can evolve independently. Anomx is not endorsed by Unit8.

The upstream public dataset catalog, external model implementations, and
training infrastructure are not imported as part of this baseline.
