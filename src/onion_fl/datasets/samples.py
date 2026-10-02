"""Helpers for loading real sample data for tests and quick demos."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLES_DIR = Path("data/samples")


def load_wesad_sample_windows(
    sample_path: Path | None = None,
    window_size: int = 200,
    step: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the real WESAD sample and create sliding-window features."""
    path = sample_path or (SAMPLES_DIR / "wesad_real_sample.pkl")
    with path.open("rb") as handle:
        data = pickle.load(handle)

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    stress_label = 2

    for subject_data in data.values():
        signals = subject_data["signal"]["wrist"]
        subject_labels = np.asarray(subject_data["label"], dtype=np.int64)

        subject_features: list[list[float]] = []
        subject_targets: list[int] = []

        for start in range(0, len(subject_labels) - window_size + 1, step):
            end = start + window_size
            window_labels = subject_labels[start:end]
            binary_label = 1 if np.mean(window_labels == stress_label) >= 0.5 else 0

            window_values: list[float] = []
            for _signal_name, values in signals.items():
                segment = np.asarray(values[start:end])
                if segment.ndim == 1:
                    segment = segment.reshape(-1, 1)

                for idx in range(segment.shape[1]):
                    series = segment[:, idx]
                    window_values.extend(
                        [
                            float(np.mean(series)),
                            float(np.std(series)),
                            float(np.min(series)),
                            float(np.max(series)),
                        ]
                    )

            subject_features.append(window_values)
            subject_targets.append(binary_label)

        if subject_features:
            features.append(np.asarray(subject_features, dtype=np.float32))
            labels.append(np.asarray(subject_targets, dtype=np.int64))

    return np.vstack(features), np.concatenate(labels)


def load_swell_sample_features(
    sample_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the SWELL sample and return numeric computer-interaction features."""
    path = sample_path or (SAMPLES_DIR / "swell_real_sample.pkl")
    with path.open("rb") as handle:
        data = pickle.load(handle)

    computer_df: pd.DataFrame = data["computer"]["data"].copy()
    numeric_columns = [
        "SnMouseAct",
        "SnLeftClicked",
        "SnRightClicked",
        "SnDoubleClicked",
        "SnWheel",
        "SnDragged",
        "SnMouseDistance",
        "SnKeyStrokes",
        "SnChars",
        "SnSpecialKeys",
        "SnDirectionKeys",
        "SnErrorKeys",
        "SnShortcutKeys",
        "SnSpaces",
        "SnAppChange",
        "SnTabfocusChange",
    ]

    features_df = (
        computer_df[numeric_columns]
        .replace("#VALUE!", np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    labels = computer_df["Condition"].isin(["T", "I", "R"]).astype(int)

    return features_df.to_numpy(dtype=np.float32), labels.to_numpy(dtype=np.int64)
