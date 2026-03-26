"""Dataset loaders for federated learning.

Dataset modules pull in optional scientific stacks. Keep imports lazy so
callers can use unrelated parts of the framework without installing every
dataset dependency upfront.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "load_ecg5000_dataset": ("flower_basic.datasets.ecg5000", "load_ecg5000_dataset"),
    "partition_ecg5000_by_subjects": (
        "flower_basic.datasets.ecg5000",
        "partition_ecg5000_by_subjects",
    ),
    "plan_and_materialize_sweet_federated": (
        "flower_basic.datasets.sweet_federated",
        "plan_and_materialize_sweet_federated",
    ),
    "load_sweet_sample_dataset": (
        "flower_basic.datasets.sweet_samples",
        "load_sweet_sample_dataset",
    ),
    "load_sweet_sample_full": (
        "flower_basic.datasets.sweet_samples",
        "load_sweet_sample_full",
    ),
    "get_swell_info": ("flower_basic.datasets.swell", "get_swell_info"),
    "load_swell_all_samples": ("flower_basic.datasets.swell", "load_swell_all_samples"),
    "load_swell_dataset": ("flower_basic.datasets.swell", "load_swell_dataset"),
    "partition_swell_by_subjects": (
        "flower_basic.datasets.swell",
        "partition_swell_by_subjects",
    ),
    "plan_and_materialize_swell_federated": (
        "flower_basic.datasets.swell_federated",
        "plan_and_materialize_swell_federated",
    ),
    "load_wesad_dataset": ("flower_basic.datasets.wesad", "load_wesad_dataset"),
    "partition_wesad_by_subjects": (
        "flower_basic.datasets.wesad",
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
