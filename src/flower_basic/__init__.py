"""Federated Learning with Fog Computing Demo.

This package exposes a small public API, but keeps heavy dependencies lazy so
importing ``flower_basic`` does not trigger torch/matplotlib side effects.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "Adriano Garcia"
__email__ = "adriano.garcia@example.com"

_LAZY_EXPORTS = {
    "BaselineTrainer": ("flower_basic.baseline_model", "BaselineTrainer"),
    "weighted_average": ("flower_basic.brokers.fog", "weighted_average"),
    "ModelComparator": ("flower_basic.compare_models", "ModelComparator"),
    "load_swell_dataset": ("flower_basic.datasets", "load_swell_dataset"),
    "load_wesad_dataset": ("flower_basic.datasets", "load_wesad_dataset"),
    "ECGModel": ("flower_basic.model", "ECGModel"),
    "get_parameters": ("flower_basic.model", "get_parameters"),
    "set_parameters": ("flower_basic.model", "set_parameters"),
    "detect_data_leakage": ("flower_basic.utils", "detect_data_leakage"),
    "state_dict_to_numpy": ("flower_basic.utils", "state_dict_to_numpy"),
    "statistical_significance_test": (
        "flower_basic.utils",
        "statistical_significance_test",
    ),
    "OTEL_AVAILABLE": ("flower_basic.telemetry", "OTEL_AVAILABLE"),
    "SpanKind": ("flower_basic.telemetry", "SpanKind"),
    "create_counter": ("flower_basic.telemetry", "create_counter"),
    "create_gauge": ("flower_basic.telemetry", "create_gauge"),
    "create_histogram": ("flower_basic.telemetry", "create_histogram"),
    "init_otel": ("flower_basic.telemetry", "init_otel"),
    "record_metric": ("flower_basic.telemetry", "record_metric"),
    "shutdown_telemetry": ("flower_basic.telemetry", "shutdown_telemetry"),
    "start_client_span": ("flower_basic.telemetry", "start_client_span"),
    "start_consumer_span": ("flower_basic.telemetry", "start_consumer_span"),
    "start_producer_span": ("flower_basic.telemetry", "start_producer_span"),
    "start_server_span": ("flower_basic.telemetry", "start_server_span"),
    "start_span": ("flower_basic.telemetry", "start_span"),
}

_DEPRECATED_EXPORTS = {
    "_deprecated_load_ecg5000_dataset": (
        "flower_basic.datasets",
        "load_ecg5000_dataset",
        "load_ecg5000_dataset is deprecated and will be removed in v0.2.0. "
        "Use load_wesad_dataset or load_swell_dataset instead.",
    ),
    "_deprecated_load_ecg5000_subject_based": (
        "flower_basic.utils",
        "load_ecg5000_subject_based",
        "load_ecg5000_subject_based is deprecated and will be removed in v0.2.0. "
        "Use load_wesad_dataset or load_swell_dataset with subject partitioning instead.",
    ),
}

__all__ = [
    "ModelComparator",
    "BaselineTrainer",
    "ECGModel",
    "load_wesad_dataset",
    "load_swell_dataset",
    "detect_data_leakage",
    "statistical_significance_test",
    "get_parameters",
    "set_parameters",
    "weighted_average",
    "init_otel",
    "start_span",
    "start_client_span",
    "start_server_span",
    "start_producer_span",
    "start_consumer_span",
    "shutdown_telemetry",
    "create_counter",
    "create_histogram",
    "create_gauge",
    "record_metric",
    "OTEL_AVAILABLE",
    "SpanKind",
    "_deprecated_load_ecg5000_dataset",
    "_deprecated_load_ecg5000_subject_based",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    if name in _DEPRECATED_EXPORTS:
        module_name, attr_name, warning = _DEPRECATED_EXPORTS[name]

        def _deprecated(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(warning, DeprecationWarning, stacklevel=2)
            return getattr(import_module(module_name), attr_name)(*args, **kwargs)

        globals()[name] = _deprecated
        return _deprecated

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
