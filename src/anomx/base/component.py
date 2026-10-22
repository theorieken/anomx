"""Shared metadata for discoverable native algorithms.

Models, scorers, detectors, classifiers, and execution algorithms expose their
configuration, capabilities, and signatures here. The platform builds its
catalog payloads from this metadata in its own component-catalog helper.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, ClassVar

from anomx._shared import normalise_component_key, normalize_text
from anomx.base.capabilities import collect_capability_keys
from anomx.base.signature import ModelSignature

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def humanize_component_name(value: str | None) -> str:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return ""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", normalized_value).strip()


def read_component_source_path(component_class: type[object]) -> str:
    source_path = inspect.getsourcefile(component_class) or inspect.getfile(component_class)
    resolved_path = Path(source_path).resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


class BaseComponent:
    """Common metadata surface for installable pipeline components."""

    component_type: ClassVar[str | None] = None
    component_key: ClassVar[str] = ""
    component_name: ClassVar[str] = ""
    component_description: ClassVar[str] = ""
    component_docs: ClassVar[str] = ""
    component_icon: ClassVar[str] = ""
    component_image_path: ClassVar[str] = ""
    component_status: ClassVar[str] = "active"
    # Backend adapters need an explicitly supplied model and are not catalog algorithms.
    component_discoverable: ClassVar[bool] = True
    component_default_config: ClassVar[dict[str, Any]] = {}
    component_config_schema: ClassVar[dict[str, Any]] = {}
    component_code_version: ClassVar[str] = ""
    signature: ClassVar[ModelSignature | None] = None

    @classmethod
    def is_component_abstract(cls) -> bool:
        return inspect.isabstract(cls) or not cls.get_component_type()

    @classmethod
    def get_component_type(cls) -> str:
        return normalize_text(getattr(cls, "component_type", "")) or ""

    @classmethod
    def get_component_key(cls) -> str:
        value = getattr(cls, "component_key", "") or cls.get_component_name() or cls.__name__
        return normalise_component_key(value)

    @classmethod
    def get_component_name(cls) -> str:
        explicit_name = normalize_text(getattr(cls, "component_name", ""))
        return explicit_name or humanize_component_name(cls.__name__)

    @classmethod
    def get_component_docs(cls) -> str:
        explicit_docs = normalize_text(getattr(cls, "component_docs", ""))
        if explicit_docs:
            return explicit_docs
        return normalize_text(inspect.getdoc(cls))

    @classmethod
    def get_component_description(cls) -> str:
        explicit_description = normalize_text(getattr(cls, "component_description", ""))
        if explicit_description:
            return explicit_description
        docs = cls.get_component_docs()
        if not docs:
            return ""
        return normalize_text(docs.split("\n\n", 1)[0])

    @classmethod
    def get_component_icon(cls) -> str:
        return normalize_text(getattr(cls, "component_icon", ""))

    @classmethod
    def get_component_image_path(cls) -> str:
        return normalize_text(getattr(cls, "component_image_path", ""))

    @classmethod
    def get_component_status(cls) -> str:
        return normalize_text(getattr(cls, "component_status", "")) or "active"

    @classmethod
    def get_component_default_config(cls) -> dict[str, Any]:
        value = getattr(cls, "component_default_config", {})
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def get_component_config_schema(cls) -> dict[str, Any]:
        value = getattr(cls, "component_config_schema", {})
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def get_component_code_version(cls) -> str:
        return normalize_text(getattr(cls, "component_code_version", ""))

    @classmethod
    def get_component_source_path(cls) -> str:
        return read_component_source_path(cls)

    @classmethod
    def get_component_import_path(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def get_component_capabilities(cls) -> list[str]:
        return collect_capability_keys(cls)

    @classmethod
    def get_component_signature(cls) -> ModelSignature | None:
        signature = getattr(cls, "signature", None)
        return signature if isinstance(signature, ModelSignature) else None

    @classmethod
    def get_component_parameters(cls) -> list[dict[str, Any]]:
        """Describe the configurable parameters of this component.

        The schema and defaults are the source of truth; the `__init__`
        docstring may add per-parameter descriptions later.
        """
        schema = cls.get_component_config_schema()
        defaults = cls.get_component_default_config()
        parameter_names = list(dict.fromkeys([*schema.keys(), *defaults.keys()]))
        return [
            {
                "name": parameter_name,
                "type": normalize_text((schema.get(parameter_name) or {}).get("type")) or "string",
                "default": defaults.get(parameter_name),
            }
            for parameter_name in parameter_names
        ]

    def get_config(self) -> dict[str, Any]:
        """Return the effective configuration of this component instance."""
        config = getattr(self, "config", None)
        merged_config = {**self.get_component_default_config()}
        if isinstance(config, dict):
            merged_config.update(config)
        return merged_config
