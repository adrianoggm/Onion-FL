"""Tests using real sample data for WESAD, SWELL, and multimodal loaders."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from flower_basic.datasets.multimodal import load_real_multimodal_dataset
from flower_basic.datasets.samples import (
    load_swell_sample_features,
    load_wesad_sample_windows,
)


def test_wesad_sample_windows_produces_real_features() -> None:
    if not Path("data/samples/wesad_real_sample.pkl").exists():
        pytest.skip("wesad_real_sample.pkl not available")
    X, y = load_wesad_sample_windows()
    assert X.size > 0
    assert y.size == X.shape[0]
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})


def test_swell_sample_features_produces_real_features() -> None:
    if not Path("data/samples/swell_real_sample.pkl").exists():
        pytest.skip("swell_real_sample.pkl not available")
    X, y = load_swell_sample_features()
    assert X.size > 0
    assert y.size == X.shape[0]
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert set(np.unique(y)).issubset({0, 1})


def test_multimodal_combination_uses_real_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (
        Path("data/samples/wesad_real_sample.pkl").exists()
        and Path("data/samples/swell_real_sample.pkl").exists()
    ):
        pytest.skip("Sample pickle files not available")
    X_wesad, y_wesad = load_wesad_sample_windows(window_size=200, step=200)
    X_swell, y_swell = load_swell_sample_features()

    def _create_split(X: np.ndarray, y: np.ndarray, prefix: str):
        idx = max(1, math.floor(0.7 * len(X)))
        X_train = X[:idx].astype(np.float32)
        X_test = X[idx:].astype(np.float32)
        y_train = y[:idx].astype(np.int64)
        y_test = y[idx:].astype(np.int64)

        train_subject_ids = [f"{prefix}_train_{i % 3}" for i in range(len(X_train))]
        test_subject_ids = [f"{prefix}_test_{i % 3}" for i in range(len(X_test))]

        info = {
            "feature_names": [f"f{i}" for i in range(X.shape[1])],
            "train_subject_ids": train_subject_ids,
            "test_subject_ids": test_subject_ids,
        }
        return X_train, X_test, y_train, y_test, info

    wesad_split = _create_split(X_wesad, y_wesad, "wesad")
    swell_split = _create_split(X_swell, y_swell, "swell")

    def _wesad_loader(**_kwargs):
        return wesad_split

    def _swell_loader(**_kwargs):
        return swell_split

    monkeypatch.setattr(
        "flower_basic.datasets.multimodal.load_wesad_dataset",
        _wesad_loader,
    )
    monkeypatch.setattr(
        "flower_basic.datasets.multimodal.load_swell_dataset",
        _swell_loader,
    )

    wesad_result, swell_result, combined = load_real_multimodal_dataset(
        include_dataset_indicator=True
    )

    assert wesad_result.X_train.shape[1] == wesad_result.X_test.shape[1]
    assert swell_result.X_train.shape[1] == swell_result.X_test.shape[1]
    assert combined.X_train.shape[1] == len(combined.feature_names)
    assert np.isfinite(combined.X_train).all()
    assert np.isfinite(combined.X_test).all()

    assert set(wesad_result.train_subject_ids).isdisjoint(wesad_result.test_subject_ids)
    assert set(swell_result.train_subject_ids).isdisjoint(swell_result.test_subject_ids)
    assert set(combined.train_subject_ids).isdisjoint(combined.test_subject_ids)

    if "eda_mean" in combined.feature_names:
        assert combined.feature_names.count("eda_mean") == 1
        assert "wesad_eda_mean" not in combined.feature_names
        assert "swell_eda_mean" not in combined.feature_names
    indicator_column = combined.X_train[:, -1]
    assert set(np.unique(indicator_column)).issubset({0.0, 1.0})
    assert set(combined.train_sources) == {"wesad", "swell"}
    assert len(combined.train_subject_ids) == combined.X_train.shape[0]
    assert len(combined.test_subject_ids) == combined.X_test.shape[0]
