"""Evaluation utilities for subject-aware cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass
class CrossValidationResult:
    """Mean/stdev metrics for a model across folds."""

    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy_mean": float(self.accuracy_mean),
            "accuracy_std": float(self.accuracy_std),
            "macro_f1_mean": float(self.macro_f1_mean),
            "macro_f1_std": float(self.macro_f1_std),
        }


def _summarize(scores: Iterable[tuple[float, float]]) -> CrossValidationResult:
    accuracies, macro_f1 = zip(*scores)
    return CrossValidationResult(
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0),
        macro_f1_mean=float(np.mean(macro_f1)),
        macro_f1_std=float(np.std(macro_f1, ddof=1) if len(macro_f1) > 1 else 0.0),
    )


def group_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str] | np.ndarray,
    n_splits: int = 5,
) -> dict[str, dict[str, float]]:
    """Run group-aware k-fold cross-validation.

    Args:
        X: Feature matrix.
        y: Label vector.
        groups: Iterable of group identifiers (e.g., subject IDs).
        n_splits: Number of folds (default 5).

    Returns:
        Mapping from model name to mean/std metrics.

    Raises:
        ValueError: If number of unique groups is less than ``n_splits``.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if unique_groups.size < n_splits:
        raise ValueError(
            f"group_cross_validation requires at least {n_splits} distinct groups; "
            f"received {unique_groups.size}"
        )

    splitter = GroupKFold(n_splits=n_splits)

    lr_scores: list[tuple[float, float]] = []
    rf_scores: list[tuple[float, float]] = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Logistic Regression with per-fold scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        lr_scores.append(
            (
                accuracy_score(y_test, lr_pred),
                f1_score(y_test, lr_pred, average="macro", zero_division=0),
            )
        )

        # Random Forest (tree-based models don't require scaling)
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_scores.append(
            (
                accuracy_score(y_test, rf_pred),
                f1_score(y_test, rf_pred, average="macro", zero_division=0),
            )
        )

    return {
        "logistic_regression": _summarize(lr_scores).as_dict(),
        "random_forest": _summarize(rf_scores).as_dict(),
    }
