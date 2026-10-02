"""Tests for the SubjectData contract (issue #86). Format fixtures only."""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.data.contract import DataError, SubjectData
from onion_fl.learning.model import DataShape


def subject(**overrides) -> SubjectData:
    fields = {
        "X": np.zeros((4, 2)),
        "y": np.array([0, 1, 1, 2]),
        "dataset": "sweet",
        "subject": "user0001",
        "task": "stress_3class",
        "n_classes": 3,
        "feature_names": ["hr", "eda"],
    }
    return SubjectData(**(fields | overrides))


def test_arrays_follow_the_contract_dtypes() -> None:
    data = subject()

    assert data.X.dtype == np.float32
    assert data.y.dtype == np.int64


def test_meta_is_derived_from_the_arrays() -> None:
    assert subject().meta == {
        "dataset": "sweet",
        "subject": "user0001",
        "task": "stress_3class",
        "n_classes": 3,
        "feature_names": ["hr", "eda"],
        "n_samples": 4,
        "class_counts": [1, 2, 1],
    }


def test_shape_feeds_the_model() -> None:
    assert subject().shape == DataShape(
        dataset="sweet", task="stress_3class", n_features=2, n_classes=3
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"y": np.array([0, 1, 1])}, "4 rows"),
        ({"y": np.array([0, 1, 1, 3])}, "n_classes"),
        ({"y": np.array([0, -1, 1, 2])}, "n_classes"),
        ({"feature_names": ["hr"]}, "feature_names"),
        ({"X": np.zeros(4)}, "2-D"),
        ({"subject": "a/b"}, "subject"),
        ({"dataset": "swell.v2"}, "dataset"),
        ({"n_classes": 1}, "n_classes"),
    ],
)
def test_inconsistent_data_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        subject(**overrides)


def test_labels_must_be_integers() -> None:
    with pytest.raises(DataError, match="integer"):
        subject(y=np.array([0.0, 1.5, 1.0, 2.0]))
