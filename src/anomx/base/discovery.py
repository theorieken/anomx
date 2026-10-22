"""Discover native component classes without platform payloads or optional ML imports."""

import pkgutil
from collections.abc import Iterator
from importlib import import_module

from anomx.base.component import BaseComponent

COMPONENT_PACKAGE_NAMES = (
    "anomx.models.forecasting",
    "anomx.models.reconstruction",
    "anomx.models.representation",
    "anomx.detection.scorers",
    "anomx.detection.detectors",
    "anomx.execution.pipeline",
)


def import_component_modules(package_names: tuple[str, ...] = COMPONENT_PACKAGE_NAMES) -> None:
    """Import every module below the configured component packages."""
    for package_name in package_names:
        package = import_module(package_name)
        package_paths = getattr(package, "__path__", None)
        if package_paths is None:
            continue
        for module_info in pkgutil.walk_packages(package_paths, f"{package.__name__}."):
            import_module(module_info.name)


def iter_component_classes(
    package_names: tuple[str, ...] = COMPONENT_PACKAGE_NAMES,
) -> Iterator[type[BaseComponent]]:
    """Yield every non-abstract component class in the package set."""
    import_component_modules(package_names=package_names)
    seen_classes: set[type[BaseComponent]] = set()
    pending_classes: list[type[BaseComponent]] = [BaseComponent]

    while pending_classes:
        current_class = pending_classes.pop()
        for subclass in current_class.__subclasses__():
            if subclass in seen_classes:
                continue
            seen_classes.add(subclass)
            pending_classes.append(subclass)
            if subclass.is_component_abstract() or not subclass.component_discoverable:
                continue
            yield subclass
