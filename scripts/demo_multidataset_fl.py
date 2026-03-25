#!/usr/bin/env python3
"""Demo: Multi-dataset FL with real WESAD and SWELL data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets import (
    partition_swell_by_subjects,
    partition_wesad_by_subjects,
)
from flower_basic.datasets.multimodal import load_real_multimodal_dataset


def _split_by_subject(
    subject_ids: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create masks for subject-based partitioning."""

    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size < 2:
        raise ValueError("Not enough subjects to split")

    train_subjects, val_subjects = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Subject split produced empty partitions")

    return train_mask, val_mask


def _print_partition_summary(
    name: str, partitions: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
) -> None:
    """Display a concise summary for each federated partition."""
    print(f"{name} partitions: {len(partitions)} clients")
    for idx, (X_train, X_test, y_train, y_test) in enumerate(partitions, start=1):
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
        print(
            f"  Client {idx}: {X_train.shape[0]} train / {X_test.shape[0]} test samples"
            f" | classes train={train_dist} test={test_dist}"
        )


def demo_dataset_loading() -> None:
    """Show dataset statistics using real data only."""
    wesad_split, swell_split, combined = load_real_multimodal_dataset()

    print("=== Dataset Loading ===")
    print(
        f"WESAD: {wesad_split.X_train.shape[0]} train ({np.unique(wesad_split.train_subject_ids).size} subjects) / "
        f"{wesad_split.X_test.shape[0]} test ({np.unique(wesad_split.test_subject_ids).size} subjects) with {wesad_split.X_train.shape[1]} features"
    )
    print(
        f"SWELL: {swell_split.X_train.shape[0]} train ({np.unique(swell_split.train_subject_ids).size} subjects) / "
        f"{swell_split.X_test.shape[0]} test ({np.unique(swell_split.test_subject_ids).size} subjects) with {swell_split.X_train.shape[1]} features"
    )
    print(
        f"Combined: {combined.X_train.shape[0]} train ({np.unique(combined.train_subject_ids).size} subjects) / "
        f"{combined.X_test.shape[0]} test ({np.unique(combined.test_subject_ids).size} subjects) with {combined.X_train.shape[1]} aligned features"
    )


def demo_federated_partitioning() -> None:
    """Demonstrate subject-based partitioning for both datasets."""
    print("\n=== Federated Partitioning ===")
    wesad_partitions = partition_wesad_by_subjects(num_clients=3)
    swell_partitions = partition_swell_by_subjects(
        n_partitions=3, modalities=["computer"]
    )

    _print_partition_summary("WESAD", wesad_partitions)
    _print_partition_summary("SWELL", swell_partitions)


def demo_model_training() -> None:
    """Train a simple baseline on the combined dataset."""
    print("\n=== Multimodal Baseline Training ===")
    _, _, combined = load_real_multimodal_dataset()

    train_mask, val_mask = _split_by_subject(combined.train_subject_ids)

    X_train = combined.X_train[train_mask]
    y_train = combined.y_train[train_mask]
    X_val = combined.X_train[val_mask]
    y_val = combined.y_train[val_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(combined.X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))
    test_accuracy = accuracy_score(combined.y_test, model.predict(X_test_scaled))

    print(f"Validation accuracy (subject-based): {val_accuracy:.3f}")
    print(f"Test accuracy (subject-based): {test_accuracy:.3f}")


def main() -> None:
    """Run the end-to-end demo."""
    print("=== Multi-dataset Federated Learning Demo ===\n")
    demo_dataset_loading()
    demo_federated_partitioning()
    demo_model_training()


if __name__ == "__main__":
    main()
