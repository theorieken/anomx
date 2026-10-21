"""Local SAFEPROCESS experiments using the anomx package."""

import os
from getpass import getpass

from anomx.data import Dataset
from anomx.integrations.platform import PlatformClient


def main() -> None:

    a = 0

    client = PlatformClient(
        url="https://anomalies.msktools.desy.de",
        token="XcVGVKNJkENp3nIarzTFkwQO8fafOpGhPI9foMhagww",
    )

    # Provisional dataset ID; replace once the dataset is finalized.
    dataset = Dataset(anomx_reference="data_dataset-486091c4-9de9-46d7-a1ee-74c1a4de7cd9", client=client)
    print(dataset.describe())

    for batch in dataset.iter_batches(batch_size=5000):
        print(batch.head())

    # For a small dataset, load all rows into one DataFrame:
    # frame = dataset.to_dataframe()


if __name__ == "__main__":
    main()
