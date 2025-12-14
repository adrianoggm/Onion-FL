"""Loader utilities for SWEET sample-subject data used in quick baselines.

This module focuses on the curated subset stored under
``data/SWEET/sample_subjects``. Each subject directory contains:

- Minute-level feature aggregates (``userXXXX_Features_Day*.csv``)
- Self-reported stress annotations (``current_stress.csv``)

We align self-reports to the closest feature window by flooring timestamps
to the minute. The resulting samples are grouped by subject so that downstream
evaluations can enforce subject-disjoint splits (as required by the project
guidelines documented in ``Context.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_SWEET_SAMPLE_DIR = Path("data/SWEET/sample_subjects")
_DROPPED_COLUMNS = {
    "timestamp",
    "minute",
    "subject_id",
    "stress_timestamp",
    "TS",
    "PLEASURE",
    "AROUSAL",
    "DOMINANCE_CONTROL",
    "CONSUMPTION",
    "ACTIVITY",
    "respondent_id",
    "MAXIMUM_STRESS",
}
_OPTIONAL_NULL_COLUMN = "('ECG_mean_heart_rate', 'raw0')"


@dataclass
class SWEETSamplePartition:
    """Container for a SWEET sample partition."""

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray


@dataclass
class SWEETSampleDataset:
    """Full dataset split for SWEET sample subjects."""

    feature_names: list[str]
    label_strategy: str
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    train: SWEETSamplePartition
    val: SWEETSamplePartition
    test: SWEETSamplePartition


class SWEETSampleLoaderError(RuntimeError):
    """Raised when SWEET sample data cannot be materialized."""


def load_sweet_sample_dataset(
    data_dir: Path | str = DEFAULT_SWEET_SAMPLE_DIR,
    *,
    label_strategy: str = "binary",
    elevated_threshold: float = 2.0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    random_state: int = 42,
    min_samples_per_subject: int = 5,
) -> SWEETSampleDataset:
    """Load SWEET sample-subject data and create subject-disjoint splits.

    Args:
        data_dir: Base directory containing per-subject folders.
        label_strategy: 
            - ``"binary"``: maps MAXIMUM_STRESS >= threshold to 1, else 0
            - ``"ordinal"``: uses raw 1-5 stress values
            - ``"ordinal_3class"``: maps to 3 classes (1->0 low, 2->1 medium, 3/4/5->2 high)
        elevated_threshold: Threshold used when ``label_strategy == "binary"``.
        train_fraction: Proportion of subjects assigned to the training split.
        val_fraction: Proportion of subjects assigned to the validation split.
        random_state: Seed for the subject-level split shuffling.
        min_samples_per_subject: Minimum merged samples required to keep a subject.

    Returns:
        SWEETSampleDataset with train/val/test partitions and metadata.

    Raises:
        SWEETSampleLoaderError: If the directory is missing or splits fail.
    """

    base_path = Path(data_dir)
    if not base_path.exists() or not base_path.is_dir():
        raise SWEETSampleLoaderError(
            f"SWEET sample directory not found: {base_path.resolve()}"
        )

    subject_datasets, feature_names = _materialize_subject_datasets(
        base_path=base_path,
        label_strategy=label_strategy,
        elevated_threshold=elevated_threshold,
        min_samples=min_samples_per_subject,
    )

    if not subject_datasets:
        raise SWEETSampleLoaderError(
            f"No SWEET sample subjects available under {base_path.resolve()}"
        )

    subject_ids = [entry[0] for entry in subject_datasets]
    train_ids, val_ids, test_ids = _split_subjects(
        subject_ids=subject_ids,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        random_state=random_state,
    )

    train_partition = _collate_partition(subject_datasets, train_ids, feature_names)
    val_partition = _collate_partition(subject_datasets, val_ids, feature_names)
    test_partition = _collate_partition(subject_datasets, test_ids, feature_names)

    return SWEETSampleDataset(
        feature_names=feature_names,
        label_strategy=label_strategy,
        train_subjects=train_ids,
        val_subjects=val_ids,
        test_subjects=test_ids,
        train=train_partition,
        val=val_partition,
        test=test_partition,
    )


def _materialize_subject_datasets(
    *,
    base_path: Path,
    label_strategy: str,
    elevated_threshold: float,
    min_samples: int,
) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
    """Return per-subject aligned samples and the shared feature schema."""

    aligned_subjects: list[tuple[str, pd.DataFrame]] = []
    feature_names: list[str] | None = None

    for subject_dir in sorted(base_path.iterdir()):
        if not subject_dir.is_dir():
            continue

        try:
            subject_df = _load_single_subject(
                subject_dir=subject_dir,
                label_strategy=label_strategy,
                elevated_threshold=elevated_threshold,
            )
        except SWEETSampleLoaderError:
            continue

        if len(subject_df) < min_samples:
            continue

        feature_cols = [
            col
            for col in subject_df.columns
            if col not in {"label", "minute", "subject_id", "stress_timestamp"}
        ]

        if feature_names is None:
            feature_names = feature_cols
        else:
            # Ensure consistent schema across subjects.
            missing = set(feature_names) - set(feature_cols)
            extra = set(feature_cols) - set(feature_names)
            if missing or extra:
                raise SWEETSampleLoaderError(
                    f"Inconsistent feature schema for {subject_dir.name}: "
                    f"missing={sorted(missing)} extra={sorted(extra)}"
                )

        aligned_subjects.append((subject_dir.name, subject_df))

    if feature_names is None:
        raise SWEETSampleLoaderError("Unable to infer SWEET sample feature schema")

    return aligned_subjects, feature_names


def _load_single_subject(
    *,
    subject_dir: Path,
    label_strategy: str,
    elevated_threshold: float,
) -> pd.DataFrame:
    """Return an aligned DataFrame for a single subject."""

    feature_files = sorted(subject_dir.glob(f"{subject_dir.name}_Features_Day*.csv"))
    if not feature_files:
        raise SWEETSampleLoaderError(
            f"No feature files found for subject {subject_dir.name}"
        )

    feature_frames = []
    for file_path in feature_files:
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Unnamed: 0": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if _OPTIONAL_NULL_COLUMN in df.columns:
            df = df.drop(columns=[_OPTIONAL_NULL_COLUMN])

        feature_frames.append(df)

    all_features = (
        pd.concat(feature_frames, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    all_features["minute"] = all_features["timestamp"].dt.floor("min")

    stress_path = subject_dir / "current_stress.csv"
    if not stress_path.exists():
        raise SWEETSampleLoaderError(
            f"Missing current_stress.csv for subject {subject_dir.name}"
        )

    stress_df = pd.read_csv(stress_path)
    stress_df["stress_timestamp"] = pd.to_datetime(
        stress_df["TS"], errors="coerce"
    ).dt.floor("min")
    stress_df["MAXIMUM_STRESS"] = pd.to_numeric(
        stress_df["MAXIMUM_STRESS"], errors="coerce"
    )
    stress_df = stress_df.dropna(subset=["stress_timestamp", "MAXIMUM_STRESS"])
    stress_df = stress_df.drop_duplicates(subset=["stress_timestamp"])

    if stress_df.empty:
        raise SWEETSampleLoaderError(
            f"No valid stress annotations for subject {subject_dir.name}"
        )

    merged = pd.merge(
        stress_df,
        all_features,
        left_on="stress_timestamp",
        right_on="minute",
        how="inner",
    )
    if merged.empty:
        raise SWEETSampleLoaderError(
            f"Failed to align stress annotations with features for {subject_dir.name}"
        )

    feature_cols = [
        col for col in merged.columns if col not in _DROPPED_COLUMNS | {"label"}
    ]

    # Replace non-finite values prior to filling missing entries.
    merged[feature_cols] = merged[feature_cols].replace([np.inf, -np.inf], np.nan)
    merged[feature_cols] = merged[feature_cols].fillna(
        merged[feature_cols].median(numeric_only=True)
    )

    if label_strategy == "binary":
        merged["label"] = (merged["MAXIMUM_STRESS"] >= elevated_threshold).astype(
            np.int64
        )
    elif label_strategy == "ordinal":
        merged["label"] = merged["MAXIMUM_STRESS"].round().astype(np.int64)
    elif label_strategy == "ordinal_3class":
        # Map to 3 classes: 1->0 (low), 2->1 (medium), 3/4/5->2 (high)
        stress_values = merged["MAXIMUM_STRESS"].round().astype(np.int64)
        merged["label"] = np.where(
            stress_values == 1, 0,
            np.where(stress_values == 2, 1, 2)
        )
    else:
        raise SWEETSampleLoaderError(
            f"Unsupported label strategy: {label_strategy!r} "
            "(expected 'binary', 'ordinal', or 'ordinal_3class')"
        )

    merged["subject_id"] = subject_dir.name

    return merged[feature_cols + ["label", "subject_id", "stress_timestamp", "minute"]]


def _split_subjects(
    *,
    subject_ids: Sequence[str],
    train_fraction: float,
    val_fraction: float,
    random_state: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return subject identifiers for train/val/test partitions."""

    if not 0.0 < train_fraction < 1.0:
        raise SWEETSampleLoaderError(
            f"train_fraction must be in (0,1), received {train_fraction}"
        )
    if not 0.0 <= val_fraction < 1.0:
        raise SWEETSampleLoaderError(
            f"val_fraction must be in [0,1), received {val_fraction}"
        )

    rng = np.random.default_rng(random_state)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    n_subjects = len(shuffled)
    train_count = max(1, int(round(train_fraction * n_subjects)))
    val_count = max(1, int(round(val_fraction * n_subjects)))

    if train_count + val_count >= n_subjects:
        # Ensure at least one subject is reserved for testing.
        val_count = max(1, n_subjects - train_count - 1)
        if train_count + val_count >= n_subjects:
            train_count = max(1, n_subjects - val_count - 1)

    test_count = n_subjects - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count = max(1, val_count - 1)

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]

    return train_ids, val_ids, test_ids


def _collate_partition(
    subject_datasets: Iterable[tuple[str, pd.DataFrame]],
    selected_subjects: Sequence[str],
    feature_names: Sequence[str],
) -> SWEETSamplePartition:
    """Stack samples for the requested subject identifiers."""

    selected_set = set(selected_subjects)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subject_labels: list[np.ndarray] = []

    for subject_id, df in subject_datasets:
        if subject_id not in selected_set:
            continue

        X = df[list(feature_names)].to_numpy(dtype=np.float32, copy=False)
        y = df["label"].to_numpy(dtype=np.int64, copy=False)
        subjects = np.asarray([subject_id] * len(df), dtype=object)

        features.append(X)
        labels.append(y)
        subject_labels.append(subjects)

    if not features:
        n_features = len(feature_names)
        return SWEETSamplePartition(
            X=np.empty((0, n_features), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            subject_ids=np.empty((0,), dtype=object),
        )

    return SWEETSamplePartition(
        X=np.vstack(features).astype(np.float32, copy=False),
        y=np.concatenate(labels).astype(np.int64, copy=False),
        subject_ids=np.concatenate(subject_labels),
    )


def load_sweet_sample_full(
    data_dir: Path | str = DEFAULT_SWEET_SAMPLE_DIR,
    *,
    label_strategy: str = "binary",
    elevated_threshold: float = 2.0,
    min_samples_per_subject: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load all SWEET sample-subject data with aligned labels and subject ids.

    This helper is intended for subject-aware cross-validation where the entire
    dataset is required without partitioning into train/val/test splits.

    Args:
        data_dir: Base directory containing per-subject folders.
        label_strategy: ``"binary"`` converts MAXIMUM_STRESS >= threshold to 1,
            ``"ordinal"`` retains the original 1-5 levels.
        elevated_threshold: Threshold for binary label creation.
        min_samples_per_subject: Minimum merged samples required to keep a subject.

    Returns:
        Tuple of (features, labels, subject_ids, feature_names).

    Raises:
        SWEETSampleLoaderError: If loading fails or no subjects are available.
    """

    subject_datasets, feature_names = _materialize_subject_datasets(
        base_path=Path(data_dir),
        label_strategy=label_strategy,
        elevated_threshold=elevated_threshold,
        min_samples=min_samples_per_subject,
    )

    if not subject_datasets:
        raise SWEETSampleLoaderError(
            f"No SWEET sample subjects available under {Path(data_dir).resolve()}"
        )

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subject_ids: list[np.ndarray] = []

    for subject_id, df in subject_datasets:
        features.append(df[feature_names].to_numpy(dtype=np.float32, copy=False))
        labels.append(df["label"].to_numpy(dtype=np.int64, copy=False))
        subject_ids.append(np.asarray([subject_id] * len(df), dtype=object))

    X = np.vstack(features).astype(np.float32, copy=False)
    y = np.concatenate(labels).astype(np.int64, copy=False)
    groups = np.concatenate(subject_ids)

    return X, y, groups, feature_names
