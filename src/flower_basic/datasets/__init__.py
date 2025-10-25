"""Dataset loaders for federated learning.

This module provides standardized dataset loaders for various physiological
and biomedical datasets used in federated learning research.

Available datasets:
- ECG5000: Time series classification for ECG signals (⚠️ DEPRECATED)
- WESAD: Wearable Stress and Affect Detection dataset (physiological signals)
- SWELL: Stress and Workload in Knowledge Work dataset (multimodal)

All loaders follow consistent interfaces and provide:
- Type-safe implementations with comprehensive type hints
- Stratified train/test splitting
- Data preprocessing and normalization
- Subject-based data partitioning for federated scenarios
- Comprehensive error handling and validation
"""

from __future__ import annotations

from .ecg5000 import load_ecg5000_dataset, partition_ecg5000_by_subjects
from .swell import (
    get_swell_info,
    load_swell_dataset,
    partition_swell_by_subjects,
    load_swell_all_samples,
)
from .swell_federated import plan_and_materialize_swell_federated
from .wesad import load_wesad_dataset, partition_wesad_by_subjects

__all__ = [
    # ECG5000 (DEPRECATED)
    "load_ecg5000_dataset",
    "partition_ecg5000_by_subjects",
    # WESAD (Physiological stress detection)
    "load_wesad_dataset",
    "partition_wesad_by_subjects",
    # SWELL (Multimodal stress detection in knowledge work)
    "load_swell_dataset",
    "partition_swell_by_subjects",
    "load_swell_all_samples",
    "plan_and_materialize_swell_federated",
    "get_swell_info",
]
