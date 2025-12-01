"""ECG5000 dataset loader for federated learning.

This module provides a standardized loader for the ECG5000 dataset from OpenML,
optimized for federated learning scenarios with proper data partitioning,
preprocessing, and subject-based splitting capabilities.

The ECG5000 dataset contains:
- 5000 ECG recordings (140 time points each)
- Binary classification: normal vs abnormal heartbeats
- Originally from PhysioBank database
- Suitable for time series classification tasks
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class ECGDatasetError(Exception):
    """Raised when ECG5000 dataset loading or processing fails."""

    pass


def load_ecg5000_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
    stratified: bool = True,
    normalize: bool = True,
    return_subject_ids: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Load ECG5000 dataset from OpenML with comprehensive preprocessing.

    Downloads and preprocesses the ECG5000 dataset for federated learning,
    providing binary classification (normal vs abnormal) with proper data
    splitting and normalization.

    Args:
        test_size: Fraction of data to use for testing (0.0-1.0).
            Defaults to 0.2.
        random_state: Random seed for reproducible splits. Defaults to 42.
        stratified: Whether to maintain class distribution in splits.
            Defaults to True.
        normalize: Whether to apply StandardScaler normalization.
            Defaults to True.
        return_subject_ids: Whether to return synthetic subject IDs for
            federated partitioning. Defaults to False.

    Returns:
        If return_subject_ids is False:
            Tuple containing (X_train, X_test, y_train, y_test)
        If return_subject_ids is True:
            Tuple containing (X_train, X_test, y_train, y_test, subject_ids)

        Where:
        - X_train: Training features (n_train_samples, 140)
        - X_test: Test features (n_test_samples, 140)
        - y_train: Training labels (n_train_samples,) - binary (0/1)
        - y_test: Test labels (n_test_samples,) - binary (0/1)
        - subject_ids: Synthetic subject identifiers for federated scenarios

    Raises:
        ECGDatasetError: If dataset download or processing fails.
        ValueError: If test_size is not in valid range (0.0, 1.0).

    Example:
        ```python
        # Basic usage
        X_train, X_test, y_train, y_test = load_ecg5000_dataset()

        # With subject IDs for federated learning
        X_train, X_test, y_train, y_test, subjects = load_ecg5000_dataset(
            return_subject_ids=True
        )

        # Custom configuration
        X_train, X_test, y_train, y_test = load_ecg5000_dataset(
            test_size=0.3,
            normalize=False,
            random_state=123
        )
        ```

    Note:
        - Original dataset has 5 classes, converted to binary (normal vs abnormal)
        - Class 1 (normal) → 0, Classes 2-5 (abnormal) → 1
        - Data is normalized using StandardScaler if normalize=True
        - Subject IDs are synthetic for federated partitioning simulation
    """
    # Validate parameters
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    try:
        logger.info("Downloading ECG5000 dataset from OpenML...")

        # Download dataset from OpenML
        ecg_data = fetch_openml(
            name="ECG5000", version=1, parser="auto", return_X_y=False
        )

        X = ecg_data.data.values.astype(np.float32)
        y_raw = ecg_data.target.values

        logger.info(f"Downloaded ECG5000: {X.shape[0]} samples, {X.shape[1]} features")

        # Convert to binary classification (normal vs abnormal)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw)

        # Class 1 (index 0) is normal, classes 2-5 (indices 1-4) are abnormal
        y_binary = (y_encoded > 0).astype(np.int64)

        logger.info(
            f"Binary classes - Normal: {np.sum(y_binary == 0)}, "
            f"Abnormal: {np.sum(y_binary == 1)}"
        )

        # Apply normalization if requested
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float32)
            logger.info("Applied StandardScaler normalization")

        # Generate synthetic subject IDs for federated scenarios
        # Simulate 15 subjects (similar to ECG5000) for consistency
        n_subjects = 15
        subjects_per_class = n_subjects // 2

        subject_ids = np.empty(len(X), dtype="U10")  # String array with sufficient size

        # Assign subjects within each class
        for class_label in [0, 1]:
            class_mask = y_binary == class_label
            class_indices = np.where(class_mask)[0]

            # Distribute samples across subjects for this class
            subjects_for_class = [
                f"S{i+1:02d}"
                for i in range(
                    class_label * subjects_per_class,
                    (class_label + 1) * subjects_per_class,
                )
            ]

            # Add extra subject if needed for odd number of subjects
            if class_label == 1 and n_subjects % 2 == 1:
                subjects_for_class.append(f"S{n_subjects:02d}")

            # Assign subject IDs cyclically
            for i, idx in enumerate(class_indices):
                subject_ids[idx] = subjects_for_class[i % len(subjects_for_class)]

        # Perform train/test split
        split_args = {
            "test_size": test_size,
            "random_state": random_state,
        }

        if stratified:
            split_args["stratify"] = y_binary

        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, **split_args)

        logger.info(
            f"Train/test split - Train: {X_train.shape[0]}, " f"Test: {X_test.shape[0]}"
        )

        # Return results
        if return_subject_ids:
            # Split subject IDs accordingly
            subject_train, subject_test = train_test_split(subject_ids, **split_args)
            return X_train, X_test, y_train, y_test, subject_train
        else:
            return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Failed to load ECG5000 dataset: {e}")
        raise ECGDatasetError(f"ECG5000 dataset loading failed: {e}") from e


def partition_ecg5000_by_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    num_clients: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition ECG5000 dataset by subjects for federated learning.

    Creates federated data partitions by assigning subjects to different
    clients, ensuring no subject appears in multiple clients (realistic
    federated scenario).

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels (n_samples,).
        subject_ids: Subject identifiers (n_samples,).
        num_clients: Number of federated clients to create.
        random_state: Random seed for reproducible partitioning.

    Returns:
        List of (X_client, y_client) tuples for each client.

    Raises:
        ValueError: If num_clients exceeds number of unique subjects.

    Example:
        ```python
        # Load data with subject IDs
        X_train, X_test, y_train, y_test, subjects = load_ecg5000_dataset(
            return_subject_ids=True
        )

        # Partition into 5 federated clients
        client_data = partition_ecg5000_by_subjects(
            X_train, y_train, subjects, num_clients=5
        )

        # Access data for first client
        X_client_0, y_client_0 = client_data[0]
        ```
    """
    unique_subjects = np.unique(subject_ids)

    if num_clients > len(unique_subjects):
        raise ValueError(
            f"Cannot create {num_clients} clients with only "
            f"{len(unique_subjects)} subjects"
        )

    # Shuffle subjects for random assignment
    np.random.seed(random_state)
    shuffled_subjects = np.random.permutation(unique_subjects)

    # Assign subjects to clients
    subjects_per_client = len(unique_subjects) // num_clients
    extra_subjects = len(unique_subjects) % num_clients

    client_data = []
    subject_idx = 0

    for client_id in range(num_clients):
        # Calculate number of subjects for this client
        num_subjects_client = subjects_per_client
        if client_id < extra_subjects:
            num_subjects_client += 1

        # Get subjects for this client
        client_subjects = shuffled_subjects[
            subject_idx : subject_idx + num_subjects_client
        ]
        subject_idx += num_subjects_client

        # Get data for these subjects
        client_mask = np.isin(subject_ids, client_subjects)
        X_client = X[client_mask]
        y_client = y[client_mask]

        client_data.append((X_client, y_client))

        logger.info(
            f"Client {client_id}: {len(client_subjects)} subjects, "
            f"{len(X_client)} samples"
        )

    return client_data
