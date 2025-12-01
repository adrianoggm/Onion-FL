"""WESAD (Wearable Stress and Affect Detection) dataset loader.

This module provides comprehensive loading and preprocessing for the WESAD dataset,
a multimodal dataset for stress detection using wearable physiological sensors.

The WESAD dataset contains:
- 15 subjects (S2-S17, excluding S1 and S12)
- Physiological signals: BVP, EDA, ACC, TEMP, HR, IBI
- Conditions: baseline, stress, amusement, meditation
- Sampling rates: 700Hz (chest), 64Hz (wrist) for most signals
- Labels: stress detection (binary classification)

Reference:
    Schmidt, P., Reiss, A., Duerr, R., Marberger, C., & Van Laerhoven, K. (2018).
    Introducing WESAD, a multimodal dataset for wearable stress and affect detection.
    In Proceedings of the 20th ACM international conference on multimodal interaction.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WESADDatasetError(Exception):
    """Raised when WESAD dataset loading or processing fails."""

    pass


# WESAD dataset configuration constants
WESAD_SUBJECTS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
]  # S1 and S12 excluded as per dataset documentation

WESAD_SIGNALS = {
    "chest": {
        "ACC": 700,  # 3-axis accelerometer
        "ECG": 700,  # Electrocardiogram
        "EMG": 700,  # Electromyogram
        "EDA": 700,  # Electrodermal activity
        "TEMP": 700,  # Temperature
        "RESP": 700,  # Respiration
    },
    "wrist": {
        "ACC": 32,  # 3-axis accelerometer
        "BVP": 64,  # Blood volume pulse
        "EDA": 4,  # Electrodermal activity
        "TEMP": 4,  # Temperature
    },
}

WESAD_CONDITIONS = {
    0: "not_defined",
    1: "baseline",  # Relaxed state
    2: "stress",  # Trier Social Stress Test (TSST)
    3: "amusement",  # Funny video clips
    4: "meditation",  # Meditation/recovery
    # Additional transient states exist but are typically excluded
}


LABEL_SAMPLING_RATE = WESAD_SIGNALS["chest"]["ACC"]


def load_wesad_dataset(
    data_dir: str | Path | None = None,
    subjects: list[str] | None = None,
    signals: list[str] = None,
    sensor_location: str = "wrist",
    conditions: list[str] = None,
    test_size: float = 0.2,
    normalize: bool = True,
    window_size: int = 60,
    overlap: float = 0.5,
    random_state: int = 42,
    return_subject_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]
):
    """Load and preprocess WESAD dataset for federated learning."""

    if conditions is None:
        conditions = ["baseline", "stress"]
    if signals is None:
        signals = ["BVP", "EDA", "ACC", "TEMP"]
    if data_dir is None:
        data_dir = Path("data/WESAD")
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"WESAD data directory not found: {data_dir}")

    if subjects is None:
        subjects = WESAD_SUBJECTS.copy()

    invalid_subjects = set(subjects) - set(WESAD_SUBJECTS)
    if invalid_subjects:
        raise ValueError(
            f"Invalid subjects: {invalid_subjects}. Valid subjects: {WESAD_SUBJECTS}"
        )

    if sensor_location not in ["wrist", "chest"]:
        raise ValueError(
            f"sensor_location must be 'wrist' or 'chest', got {sensor_location}"
        )

    available_signals = list(WESAD_SIGNALS[sensor_location].keys())
    invalid_signals = set(signals) - set(available_signals)
    if invalid_signals:
        raise ValueError(
            f"Invalid signals for {sensor_location}: {invalid_signals}."
            f" Available: {available_signals}"
        )

    available_conditions = ["baseline", "stress", "amusement", "meditation"]
    invalid_conditions = set(conditions) - set(available_conditions)
    if invalid_conditions:
        raise ValueError(
            f"Invalid conditions: {invalid_conditions}."
            f" Available: {available_conditions}"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be between 0.0 and 1.0, got {overlap}")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    try:

        def _distribution(values: np.ndarray) -> dict[int, int]:
            unique, counts = np.unique(values, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}

        logger.info("Loading WESAD dataset from %s", data_dir)
        logger.info(
            "Subjects: %s, Signals: %s, Sensor: %s, Conditions: %s",
            subjects,
            signals,
            sensor_location,
            conditions,
        )

        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_subject_ids: list[str] = []
        feature_names: list[str] | None = None

        for subject_id in subjects:
            try:
                (
                    subject_features,
                    subject_labels,
                    subject_feature_names,
                ) = _load_subject_data(
                    data_dir=data_dir,
                    subject_id=subject_id,
                    signals=signals,
                    sensor_location=sensor_location,
                    conditions=conditions,
                    window_size=window_size,
                    overlap=overlap,
                )

                if feature_names is None:
                    feature_names = subject_feature_names
                elif feature_names != subject_feature_names:
                    logger.warning(
                        "Feature names mismatch for %s; using first subject's schema",
                        subject_id,
                    )

                all_features.append(subject_features)
                all_labels.append(subject_labels)
                all_subject_ids.extend([subject_id] * len(subject_features))

                logger.info("Loaded %s: %d windows", subject_id, len(subject_features))

            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load %s: %s", subject_id, exc)
                continue

        if not all_features:
            raise WESADDatasetError("No valid subject data could be loaded")

        X = np.vstack(all_features)
        y = np.concatenate(all_labels).astype(np.int64)
        subject_ids = np.asarray(all_subject_ids)

        logger.info("Combined data: %d windows, %d features", X.shape[0], X.shape[1])
        logger.info(
            "Class distribution - Non-stress: %d, Stress: %d",
            int(np.sum(y == 0)),
            int(np.sum(y == 1)),
        )

        unique_subjects = np.unique(subject_ids)
        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

        train_mask = np.isin(subject_ids, train_subjects)
        test_mask = np.isin(subject_ids, test_subjects)

        X_train_raw = X[train_mask]
        X_test_raw = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        train_subject_ids = subject_ids[train_mask]
        test_subject_ids = subject_ids[test_mask]

        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
            X_test = scaler.transform(X_test_raw).astype(np.float32)
        else:
            X_train = X_train_raw.astype(np.float32)
            X_test = X_test_raw.astype(np.float32)

        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(X_train.shape[1])]

        if return_subject_info:
            info: dict[str, object] = {
                "feature_names": feature_names,
                "subjects": unique_subjects.tolist(),
                "train_subjects": train_subjects.tolist(),
                "test_subjects": test_subjects.tolist(),
                "train_subject_ids": train_subject_ids.tolist(),
                "test_subject_ids": test_subject_ids.tolist(),
                "signals": signals,
                "sensor_location": sensor_location,
                "conditions": conditions,
                "window_size": window_size,
                "overlap": overlap,
                "class_distribution": _distribution(y),
                "train_class_distribution": _distribution(y_train),
                "test_class_distribution": _distribution(y_test),
            }
            if normalize:
                info["scaler_mean"] = scaler.mean_.astype(float).tolist()
                info["scaler_scale"] = scaler.scale_.astype(float).tolist()

            return X_train, X_test, y_train, y_test, info

        return X_train, X_test, y_train, y_test

    except Exception as exc:  # pragma: no cover
        logger.error("Failed to load WESAD dataset: %s", exc)
        raise WESADDatasetError(f"WESAD dataset loading failed: {exc}") from exc


def _load_subject_data(
    data_dir: Path,
    subject_id: str,
    signals: list[str],
    sensor_location: str,
    conditions: list[str],
    window_size: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess data for a single subject."""

    subject_path = data_dir / subject_id / f"{subject_id}.pkl"
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_path}")

    with open(subject_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    signal_data = data["signal"][sensor_location]
    labels = np.asarray(data["label"], dtype=np.int64)

    condition_mapping = {v: k for k, v in WESAD_CONDITIONS.items()}
    target_labels = [condition_mapping[cond] for cond in conditions]
    stress_label = condition_mapping["stress"]

    label_rate = LABEL_SAMPLING_RATE
    if label_rate <= 0:
        raise WESADDatasetError("Invalid label sampling rate")

    total_duration = len(labels) / label_rate
    if total_duration <= 0:
        raise WESADDatasetError(f"No label data available for {subject_id}")

    # Precompute channel naming for deterministic feature ordering
    channel_names: list[str] = []
    for signal_name in signals:
        if signal_name not in signal_data:
            logger.warning("Signal %s not found for %s", signal_name, subject_id)
            continue

        values = signal_data[signal_name]
        if values.ndim > 1:
            for idx in range(values.shape[1]):
                channel_names.append(f"{signal_name.lower()}_{idx}")
        else:
            channel_names.append(signal_name.lower())

    features: list[list[float]] = []
    window_labels: list[int] = []

    step_seconds = window_size * (1 - overlap)
    if step_seconds <= 0:
        step_seconds = window_size

    start_time = 0.0
    while start_time + window_size <= total_duration:
        end_time = start_time + window_size

        label_start = int(round(start_time * label_rate))
        label_end = int(round(end_time * label_rate))
        if label_end <= label_start or label_end > len(labels):
            break

        label_segment = labels[label_start:label_end]
        if label_segment.size == 0:
            break

        if not np.all(np.isin(label_segment, target_labels)):
            start_time += step_seconds
            continue

        window_feature_values: list[float] = []
        valid_window = True

        for signal_name in signals:
            if signal_name not in signal_data:
                valid_window = False
                break

            signal_values = signal_data[signal_name]
            sample_rate = WESAD_SIGNALS[sensor_location][signal_name]

            signal_start = int(round(start_time * sample_rate))
            signal_end = int(round(end_time * sample_rate))
            if signal_end <= signal_start or signal_end > signal_values.shape[0]:
                valid_window = False
                break

            segment = signal_values[signal_start:signal_end]
            if segment.size == 0:
                valid_window = False
                break

            if segment.ndim == 1:
                segment = segment.reshape(-1, 1)

            for channel_idx in range(segment.shape[1]):
                channel_data = segment[:, channel_idx]
                window_feature_values.extend(
                    [
                        float(np.mean(channel_data)),
                        float(np.std(channel_data)),
                        float(np.min(channel_data)),
                        float(np.max(channel_data)),
                        float(np.median(channel_data)),
                    ]
                )

        if not valid_window:
            start_time += step_seconds
            continue

        binary_segment = (label_segment == stress_label).astype(int)
        window_label = int(np.round(binary_segment.mean()))

        features.append(window_feature_values)
        window_labels.append(window_label)

        start_time += step_seconds

    if not features:
        raise WESADDatasetError(
            f"No valid data found for {subject_id} with the specified configuration"
        )

    stats = ["mean", "std", "min", "max", "median"]
    if not channel_names:
        raise WESADDatasetError(f"No valid signals found for {subject_id}")

    feature_names = [f"{channel}_{stat}" for channel in channel_names for stat in stats]

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(window_labels, dtype=np.int64),
        feature_names,
    )


def partition_wesad_by_subjects(
    data_dir: str | Path | None = None,
    num_clients: int = 5,
    signals: list[str] = None,
    sensor_location: str = "wrist",
    conditions: list[str] = None,
    **kwargs,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Partition WESAD dataset by subjects for federated learning.

    Creates federated data partitions by assigning subjects to different
    clients, ensuring no subject appears in multiple clients (realistic
    federated scenario for healthcare data).

    Args:
        data_dir: Path to WESAD dataset directory.
        num_clients: Number of federated clients to create.
        signals: List of physiological signals to extract.
        sensor_location: Sensor location ('wrist' or 'chest').
        conditions: List of experimental conditions.
        **kwargs: Additional arguments passed to load_wesad_dataset.

    Returns:
        List of (X_train, X_test, y_train, y_test) tuples for each client.

    Raises:
        ValueError: If num_clients exceeds number of available subjects.

    Example:
        ```python
        # Partition WESAD into 3 federated clients
        client_datasets = partition_wesad_by_subjects(
            num_clients=3,
            signals=['BVP', 'EDA'],
            test_size=0.2
        )

        # Access data for first client
        X_train_0, X_test_0, y_train_0, y_test_0 = client_datasets[0]
        ```
    """
    if conditions is None:
        conditions = ["baseline", "stress"]
    if signals is None:
        signals = ["BVP", "EDA", "ACC", "TEMP"]
    if num_clients > len(WESAD_SUBJECTS):
        raise ValueError(
            f"Cannot create {num_clients} clients with only "
            f"{len(WESAD_SUBJECTS)} subjects"
        )

    # Shuffle subjects for random assignment
    np.random.seed(kwargs.get("random_state", 42))
    shuffled_subjects = np.random.permutation(WESAD_SUBJECTS)

    # Distribute subjects among clients
    subjects_per_client = len(WESAD_SUBJECTS) // num_clients
    extra_subjects = len(WESAD_SUBJECTS) % num_clients

    client_datasets = []
    subject_idx = 0

    for client_id in range(num_clients):
        # Calculate number of subjects for this client
        num_subjects_client = subjects_per_client
        if client_id < extra_subjects:
            num_subjects_client += 1

        # Get subjects for this client
        client_subjects = shuffled_subjects[
            subject_idx : subject_idx + num_subjects_client
        ].tolist()
        subject_idx += num_subjects_client

        # Load data for this client's subjects
        try:
            X_train, X_test, y_train, y_test = load_wesad_dataset(
                data_dir=data_dir,
                subjects=client_subjects,
                signals=signals,
                sensor_location=sensor_location,
                conditions=conditions,
                **kwargs,
            )

            client_datasets.append((X_train, X_test, y_train, y_test))

            logger.info(
                f"Client {client_id}: subjects {client_subjects}, "
                f"train: {len(X_train)}, test: {len(X_test)} windows"
            )

        except Exception as e:
            logger.error(
                f"Failed to create client {client_id} with subjects "
                f"{client_subjects}: {e}"
            )
            # Create empty dataset for failed client
            client_datasets.append(
                (
                    np.array([]).reshape(0, 0),
                    np.array([]).reshape(0, 0),
                    np.array([]),
                    np.array([]),
                )
            )

    return client_datasets
