"""Dataset loaders for federated learning.

Dataset modules pull in optional scientific stacks. Keep imports lazy so
callers can use unrelated parts of the framework without installing every
dataset dependency upfront.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "plan_and_materialize_sweet_federated": (
        "onion_fl.datasets.sweet_federated",
        "plan_and_materialize_sweet_federated",
    ),
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
    "plan_and_materialize_swell_federated": (
        "onion_fl.datasets.swell_federated",
        "plan_and_materialize_swell_federated",
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
