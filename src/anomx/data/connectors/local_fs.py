"""Filesystem-backed CSV connector."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.data.connectors.base import BaseConnector


class LocalFSConnector(BaseConnector):
    """Read and write CSV files from the local filesystem."""

    def read(self, config: dict[str, Any]) -> pd.DataFrame:
        path = self._resolve_path(config)
        return pd.read_csv(path)

    def write(self, data: Any, config: dict[str, Any]) -> None:
        path = self._resolve_output_path(config)
        frame = ensure_dataframe(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def stream(self, config: dict[str, Any]) -> Iterable[dict[str, Any]]:
        frame = self.read(config)
        for record in frame.to_dict(orient="records"):
            yield record

    def schema(self) -> dict[str, Any]:
        return {
            "path": "Relative or absolute CSV path.",
            "output_path": "Optional CSV path used by write().",
        }

    @staticmethod
    def _resolve_path(config: dict[str, Any]) -> Path:
        configured_path = config.get("path")
        if not configured_path:
            raise ValueError("LocalFSConnector requires a 'path' entry in connector config.")
        candidate = Path(str(configured_path))
        return candidate if candidate.is_absolute() else Path.cwd() / candidate

    @staticmethod
    def _resolve_output_path(config: dict[str, Any]) -> Path:
        output_path = str(config.get("output_path") or "artifacts/local_fs/output.csv")
        candidate = Path(output_path)
        return candidate if candidate.is_absolute() else Path.cwd() / candidate
