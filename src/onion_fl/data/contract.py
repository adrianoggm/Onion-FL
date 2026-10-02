from __future__ import annotations

"""The data contract every dataset is turned into (spec §8.1).

One ``SubjectData`` per subject: ``X: float32[n, f]``, ``y: int64[n]`` and the
meta the rest of the framework needs. Nothing downstream knows which dataset
produced it.
"""

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import ValidationError

from onion_fl.learning.model import DataShape

SUBJECT = r"^[A-Za-z0-9_.-]+$"  # also a safe file name in the cache


class DataError(ValueError):
    """Data does not follow the contract, or a dataset description is wrong."""


@dataclass(frozen=True, eq=False)
class SubjectData:
    X: np.ndarray
    y: np.ndarray
    dataset: str
    subject: str
    task: str
    n_classes: int
    feature_names: list[str]

    def __post_init__(self) -> None:
        X, y = np.asarray(self.X, dtype=np.float32), np.asarray(self.y)
        if X.ndim != 2:
            raise DataError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise DataError(f"X has {len(X)} rows but y has shape {y.shape}")
        if y.size and not np.array_equal(y, np.round(y)):
            raise DataError("labels must be integer classes")
        if len(self.feature_names) != X.shape[1]:
            raise DataError(
                f"{len(self.feature_names)} feature_names for {X.shape[1]} columns"
            )
        if not re.fullmatch(SUBJECT, self.subject):
            raise DataError(f"subject {self.subject!r} must match {SUBJECT}")
        try:
            DataShape(
                dataset=self.dataset,
                task=self.task,
                n_features=max(X.shape[1], 1),
                n_classes=self.n_classes,
            )
        except ValidationError as exc:
            raise DataError(str(exc)) from None
        y = y.astype(np.int64)
        if y.size and (y.min() < 0 or y.max() >= self.n_classes):
            raise DataError(
                f"labels must be in [0, n_classes={self.n_classes}), "
                f"got {y.min()}..{y.max()}"
            )
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "feature_names", list(self.feature_names))

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def class_counts(self) -> list[int]:
        return np.bincount(self.y, minlength=self.n_classes).tolist()

    @property
    def shape(self) -> DataShape:
        return DataShape(
            dataset=self.dataset,
            task=self.task,
            n_features=self.X.shape[1],
            n_classes=self.n_classes,
        )

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "task": self.task,
            "n_classes": self.n_classes,
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
            "class_counts": self.class_counts,
        }
