#!/usr/bin/env python3
"""Evaluate multimodal baseline on real WESAD + SWELL data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import json

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flower_basic.datasets.multimodal import load_real_multimodal_dataset
from flower_basic.evaluation import group_cross_validation

OUTPUT_PATH = Path("multimodal_baseline_results.json")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a standard metric dictionary."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _split_by_subject(
    subject_ids: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Create boolean masks for subject-based splitting."""

    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size < 2:
        raise ValueError("Not enough subjects to perform a split")

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


def _train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float]:
    """Fit the model and evaluate on the provided split."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    return _compute_metrics(y_eval, predictions)


def main() -> None:
    """Run baseline evaluation using real multimodal data."""

    wesad_split, swell_split, combined = load_real_multimodal_dataset()

    train_mask, val_mask = _split_by_subject(
        combined.train_subject_ids,
        test_size=0.2,
        random_state=42,
    )

    X_train = combined.X_train[train_mask]
    y_train = combined.y_train[train_mask]
    X_val = combined.X_train[val_mask]
    y_val = combined.y_train[val_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results: dict[str, dict[str, dict[str, float]]] = {}

    for name, model in models.items():
        val_model = clone(model)
        val_metrics = _train_model(
            val_model, X_train_scaled, y_train, X_val_scaled, y_val
        )

        full_model = clone(model)
        scaler_full = StandardScaler()
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        X_full_scaled = scaler_full.fit_transform(X_full)
        X_test_full = scaler_full.transform(combined.X_test)
        test_metrics = _train_model(
            full_model, X_full_scaled, y_full, X_test_full, combined.y_test
        )

        results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    X_full = np.vstack([combined.X_train, combined.X_test])
    y_full = np.concatenate([combined.y_train, combined.y_test])
    groups_full = np.concatenate(
        [combined.train_subject_ids, combined.test_subject_ids]
    )
    cv_results = group_cross_validation(X_full, y_full, groups_full)

    metadata = {
        "cross_validation": cv_results,
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(combined.X_test.shape[0]),
        "train_subjects": int(np.unique(combined.train_subject_ids[train_mask]).size),
        "validation_subjects": int(
            np.unique(combined.train_subject_ids[val_mask]).size
        ),
        "test_subjects": int(np.unique(combined.test_subject_ids).size),
        "train_sources": {
            "wesad": int(np.sum(combined.train_sources[train_mask] == "wesad")),
            "swell": int(np.sum(combined.train_sources[train_mask] == "swell")),
        },
        "validation_sources": {
            "wesad": int(np.sum(combined.train_sources[val_mask] == "wesad")),
            "swell": int(np.sum(combined.train_sources[val_mask] == "swell")),
        },
        "test_sources": {
            "wesad": int(np.sum(combined.test_sources == "wesad")),
            "swell": int(np.sum(combined.test_sources == "swell")),
        },
    }

    payload = {"metadata": metadata, "metrics": results}
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to {OUTPUT_PATH}")
    print("\n5-fold cross-validation (accuracy +/- std / macro-F1 +/- std):")
    for model_name, stats in cv_results.items():
        print(
            f"  {model_name}: {stats['accuracy_mean']:.3f} +/- {stats['accuracy_std']:.3f} / {stats['macro_f1_mean']:.3f} +/- {stats['macro_f1_std']:.3f}"
        )


if __name__ == "__main__":
    main()
