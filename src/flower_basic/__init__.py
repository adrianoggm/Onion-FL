"""Federated Learning with Fog Computing Demo.

A comprehensive implementation of federated learning using Flower framework
with fog computing architecture for ECG classification.

This package provides:
- Federated learning simulation with fog computing
- Robust evaluation framework with statistical validation
- Data leakage detection and analysis
- Comprehensive testing and benchmarking tools

Example:
    >>> from flower_basic import ModelComparator
    >>> comparator = ModelComparator()
    >>> results = comparator.run_robust_comparison()
"""

__version__ = "0.1.0"
__author__ = "Adriano Garcia"
__email__ = "adriano.garcia@example.com"

# Deprecated ECG5000 imports - will be removed in v0.2.0
import warnings

from .baseline_model import BaselineTrainer
from .brokers.fog import weighted_average
from .compare_models import ModelComparator
from .datasets import load_swell_dataset, load_wesad_dataset
from .model import ECGModel, get_parameters, set_parameters

# Telemetry exports
from .telemetry import (
    OTEL_AVAILABLE,
    SpanKind,
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_client_span,
    start_consumer_span,
    start_producer_span,
    start_server_span,
    start_span,
)
from .utils import (
    detect_data_leakage,
    state_dict_to_numpy,
    statistical_significance_test,
)

try:
    from .datasets import load_ecg5000_dataset
    from .utils import load_ecg5000_subject_based

    # Wrap deprecated functions with warnings
    def _deprecated_load_ecg5000_dataset(*args, **kwargs):
        warnings.warn(
            "load_ecg5000_dataset is deprecated and will be removed in v0.2.0. "
            "Use load_wesad_dataset or load_swell_dataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_ecg5000_dataset(*args, **kwargs)

    def _deprecated_load_ecg5000_subject_based(*args, **kwargs):
        warnings.warn(
            "load_ecg5000_subject_based is deprecated and will be removed in v0.2.0. "
            "Use load_wesad_dataset or load_swell_dataset with subject partitioning instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_ecg5000_subject_based(*args, **kwargs)

except ImportError:
    # ECG5000 modules not available
    pass

__all__ = [
    "ModelComparator",
    "BaselineTrainer",
    "ECGModel",
    # Primary dataset loaders
    "load_wesad_dataset",
    "load_swell_dataset",
    # Utility functions
    "detect_data_leakage",
    "statistical_significance_test",
    "get_parameters",
    "set_parameters",
    "weighted_average",
    # Telemetry
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
    # Deprecated (will be removed in v0.2.0)
    "_deprecated_load_ecg5000_dataset",
    "_deprecated_load_ecg5000_subject_based",
]
