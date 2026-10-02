"""Federated Learning with Fog Computing Demo.

This package exposes a small public API, but keeps heavy dependencies lazy so
importing ``onion_fl`` does not trigger torch/matplotlib side effects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"
__author__ = "Adriano Garcia"

_LAZY_EXPORTS = {
    "weighted_average": ("onion_fl.brokers.fog", "weighted_average"),
    "load_swell_dataset": ("onion_fl.datasets", "load_swell_dataset"),
    "load_wesad_dataset": ("onion_fl.datasets", "load_wesad_dataset"),
    "OTEL_AVAILABLE": ("onion_fl.telemetry", "OTEL_AVAILABLE"),
    "SpanKind": ("onion_fl.telemetry", "SpanKind"),
    "create_counter": ("onion_fl.telemetry", "create_counter"),
    "create_gauge": ("onion_fl.telemetry", "create_gauge"),
    "create_histogram": ("onion_fl.telemetry", "create_histogram"),
    "init_otel": ("onion_fl.telemetry", "init_otel"),
    "record_metric": ("onion_fl.telemetry", "record_metric"),
    "shutdown_telemetry": ("onion_fl.telemetry", "shutdown_telemetry"),
    "start_client_span": ("onion_fl.telemetry", "start_client_span"),
    "start_consumer_span": ("onion_fl.telemetry", "start_consumer_span"),
    "start_producer_span": ("onion_fl.telemetry", "start_producer_span"),
    "start_server_span": ("onion_fl.telemetry", "start_server_span"),
    "start_span": ("onion_fl.telemetry", "start_span"),
}

__all__ = [
    "load_wesad_dataset",
    "load_swell_dataset",
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
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
