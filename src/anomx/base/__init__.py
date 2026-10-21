"""Shared metadata, capability, and signature contracts for reusable components."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BaseComponent": "anomx.base.component",
    "BoundaryEstimating": "anomx.base.capabilities",
    "Capability": "anomx.base.capabilities",
    "ComponentCapabilities": "anomx.base.capabilities",
    "DataStructure": "anomx.base.signature",
    "Distributional": "anomx.base.capabilities",
    "ModelSignature": "anomx.base.signature",
    "Predictive": "anomx.base.capabilities",
    "Reconstructive": "anomx.base.capabilities",
    "Representational": "anomx.base.capabilities",
    "collect_capability_keys": "anomx.base.capabilities",
    "humanize_component_name": "anomx.base.component",
    "import_component_modules": "anomx.base.discovery",
    "iter_component_classes": "anomx.base.discovery",
    "read_component_source_path": "anomx.base.component",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - lazy public API
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
