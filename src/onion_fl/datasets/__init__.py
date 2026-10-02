"""The loaders from before the redesign.

The framework reads data through ``onion_fl.data`` and the descriptors in
``datasets/``; these loaders stay as the reference that the descriptors are
checked against (``tests/test_datasets_descriptors.py``).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "load_sweet_sample_dataset": (
        "onion_fl.datasets.sweet_samples",
        "load_sweet_sample_dataset",
    ),
    "load_sweet_sample_full": (
        "onion_fl.datasets.sweet_samples",
        "load_sweet_sample_full",
    ),
    "get_swell_info": ("onion_fl.datasets.swell", "get_swell_info"),
    "load_swell_all_samples": ("onion_fl.datasets.swell", "load_swell_all_samples"),
    "load_swell_dataset": ("onion_fl.datasets.swell", "load_swell_dataset"),
    "partition_swell_by_subjects": (
        "onion_fl.datasets.swell",
        "partition_swell_by_subjects",
    ),
    "load_wesad_dataset": ("onion_fl.datasets.wesad", "load_wesad_dataset"),
    "partition_wesad_by_subjects": (
        "onion_fl.datasets.wesad",
        "partition_wesad_by_subjects",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
