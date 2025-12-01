"""Utilities for combining real WESAD and SWELL datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .swell import load_swell_dataset
from .wesad import WESAD_SUBJECTS, load_wesad_dataset

CANONICAL_SHARED_FEATURES = {
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_median",
}


DEFAULT_WESAD_TEST_FRACTION = 5 / len(WESAD_SUBJECTS)


@dataclass
class DatasetSplit:
    """Container for a single dataset split."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_subject_ids: np.ndarray
    test_subject_ids: np.ndarray
    feature_names: list[str]


@dataclass
class CombinedDataset:
    """Combined WESAD + SWELL dataset representation."""

    feature_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_sources: np.ndarray
    test_sources: np.ndarray
    train_subject_ids: np.ndarray
    test_subject_ids: np.ndarray


def _canonicalize_feature_frame(
    X: np.ndarray,
    feature_names: list[str],
    dataset_label: str,
) -> pd.DataFrame:
    """Return a dataframe with canonical/shared feature names."""

    df = pd.DataFrame(X, columns=feature_names)
    rename_map: dict[str, str] = {}
    prefix = f"{dataset_label}_"

    for name in feature_names:
        if name in CANONICAL_SHARED_FEATURES:
            rename_map[name] = name
        elif name.startswith(prefix):
            rename_map[name] = name
        else:
            rename_map[name] = f"{prefix}{name}"

    return df.rename(columns=rename_map)


def _load_wesad_split(kwargs: dict[str, Any] | None = None) -> DatasetSplit:
    """Load WESAD data with real samples and metadata."""

    kwargs = kwargs or {}
    kwargs.setdefault("test_size", DEFAULT_WESAD_TEST_FRACTION)
    kwargs["return_subject_info"] = True
    X_train, X_test, y_train, y_test, info = load_wesad_dataset(**kwargs)

    raw_feature_names = info.get("feature_names") or [
        f"f{i}" for i in range(X_train.shape[1])
    ]
    train_df = _canonicalize_feature_frame(X_train, raw_feature_names, "wesad")
    feature_names = list(train_df.columns)
    test_df = _canonicalize_feature_frame(X_test, raw_feature_names, "wesad").reindex(
        columns=feature_names, fill_value=0.0
    )

    train_subject_ids = np.array([f"wesad:{sid}" for sid in info["train_subject_ids"]])
    test_subject_ids = np.array([f"wesad:{sid}" for sid in info["test_subject_ids"]])

    return DatasetSplit(
        name="wesad",
        X_train=train_df.to_numpy(np.float32, copy=False),
        X_test=test_df.to_numpy(np.float32, copy=False),
        y_train=y_train.astype(np.int64, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
        train_subject_ids=train_subject_ids,
        test_subject_ids=test_subject_ids,
        feature_names=feature_names,
    )


def _load_swell_split(kwargs: dict[str, Any] | None = None) -> DatasetSplit:
    """Load SWELL data with real samples and metadata."""

    kwargs = kwargs or {}
    kwargs.setdefault("modalities", ["computer"])
    kwargs["return_subject_info"] = True
    X_train, X_test, y_train, y_test, info = load_swell_dataset(**kwargs)

    raw_feature_names = info.get("feature_names") or [
        f"f{i}" for i in range(X_train.shape[1])
    ]
    train_df = _canonicalize_feature_frame(X_train, raw_feature_names, "swell")
    feature_names = list(train_df.columns)
    test_df = _canonicalize_feature_frame(X_test, raw_feature_names, "swell").reindex(
        columns=feature_names, fill_value=0.0
    )

    train_subject_ids = np.array([f"swell:{sid}" for sid in info["train_subject_ids"]])
    test_subject_ids = np.array([f"swell:{sid}" for sid in info["test_subject_ids"]])

    return DatasetSplit(
        name="swell",
        X_train=train_df.to_numpy(np.float32, copy=False),
        X_test=test_df.to_numpy(np.float32, copy=False),
        y_train=y_train.astype(np.int64, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
        train_subject_ids=train_subject_ids,
        test_subject_ids=test_subject_ids,
        feature_names=feature_names,
    )


def _combine_columns(wesad_columns: list[str], swell_columns: list[str]) -> list[str]:
    """Preserve order while merging feature name lists."""

    ordered: list[str] = []
    for name in wesad_columns + swell_columns:
        if name not in ordered:
            ordered.append(name)
    return ordered


def load_real_multimodal_dataset(
    wesad_kwargs: dict[str, Any] | None = None,
    swell_kwargs: dict[str, Any] | None = None,
    include_dataset_indicator: bool = True,
) -> tuple[DatasetSplit, DatasetSplit, CombinedDataset]:
    """Load real WESAD and SWELL datasets and produce a combined split."""

    wesad_split = _load_wesad_split(wesad_kwargs)
    swell_split = _load_swell_split(swell_kwargs)

    combined_columns = _combine_columns(
        wesad_split.feature_names, swell_split.feature_names
    )

    wesad_train_df = pd.DataFrame(
        wesad_split.X_train, columns=wesad_split.feature_names
    ).reindex(columns=combined_columns, fill_value=0.0)
    wesad_test_df = pd.DataFrame(
        wesad_split.X_test, columns=wesad_split.feature_names
    ).reindex(columns=combined_columns, fill_value=0.0)

    swell_train_df = pd.DataFrame(
        swell_split.X_train, columns=swell_split.feature_names
    ).reindex(columns=combined_columns, fill_value=0.0)
    swell_test_df = pd.DataFrame(
        swell_split.X_test, columns=swell_split.feature_names
    ).reindex(columns=combined_columns, fill_value=0.0)

    X_train_wesad = wesad_train_df.to_numpy(np.float32, copy=False)
    X_test_wesad = wesad_test_df.to_numpy(np.float32, copy=False)
    X_train_swell = swell_train_df.to_numpy(np.float32, copy=False)
    X_test_swell = swell_test_df.to_numpy(np.float32, copy=False)

    if include_dataset_indicator:
        wesad_indicator_train = np.ones((X_train_wesad.shape[0], 1), dtype=np.float32)
        wesad_indicator_test = np.ones((X_test_wesad.shape[0], 1), dtype=np.float32)
        swell_indicator_train = np.zeros((X_train_swell.shape[0], 1), dtype=np.float32)
        swell_indicator_test = np.zeros((X_test_swell.shape[0], 1), dtype=np.float32)

        X_train_wesad = np.hstack([X_train_wesad, wesad_indicator_train])
        X_test_wesad = np.hstack([X_test_wesad, wesad_indicator_test])
        X_train_swell = np.hstack([X_train_swell, swell_indicator_train])
        X_test_swell = np.hstack([X_test_swell, swell_indicator_test])

        combined_feature_names = combined_columns + ["dataset_is_wesad"]
    else:
        combined_feature_names = combined_columns

    X_train = np.vstack([X_train_wesad, X_train_swell])
    X_test = np.vstack([X_test_wesad, X_test_swell])
    y_train = np.concatenate([wesad_split.y_train, swell_split.y_train])
    y_test = np.concatenate([wesad_split.y_test, swell_split.y_test])
    train_sources = np.concatenate(
        [
            np.full(wesad_split.X_train.shape[0], "wesad"),
            np.full(swell_split.X_train.shape[0], "swell"),
        ]
    )
    test_sources = np.concatenate(
        [
            np.full(wesad_split.X_test.shape[0], "wesad"),
            np.full(swell_split.X_test.shape[0], "swell"),
        ]
    )

    train_subject_ids = np.concatenate(
        [wesad_split.train_subject_ids, swell_split.train_subject_ids]
    )
    test_subject_ids = np.concatenate(
        [wesad_split.test_subject_ids, swell_split.test_subject_ids]
    )

    combined = CombinedDataset(
        feature_names=combined_feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        train_sources=train_sources,
        test_sources=test_sources,
        train_subject_ids=train_subject_ids,
        test_subject_ids=test_subject_ids,
    )

    return wesad_split, swell_split, combined
