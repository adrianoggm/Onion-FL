#!/usr/bin/env python3
"""
Baseline evaluation for SWEET sample subjects.

This script mirrors the baseline studies created for WESAD and SWELL, but it
operates on the curated ``data/SWEET/sample_subjects`` subset. Key aspects:

- Subject-level splits: train/val/test partitions respect subject boundaries,
  as mandated in ``docs/Context.md`` to avoid leakage between splits.
- Lightweight classical models (Logistic Regression, Random Forest) to
  establish a reference before launching federated experiments.
- Binary stress detection by default (MAXIMUM_STRESS >= 2 is treated as
  elevated stress) with optional ordinal classification.

Usage (binary label, default subject split):

    python scripts/evaluate_sweet_sample_baseline.py \\
        --output-dir baseline_results/sweet_samples

Usage (ordinal labels, custom random seed):

    python scripts/evaluate_sweet_sample_baseline.py \\
        --label-strategy ordinal --random-state 123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets import load_sweet_sample_dataset


def _build_models(random_state: int) -> Dict[str, ClassifierMixin]:
    """Return the baseline models to evaluate."""

    models: Dict[str, ClassifierMixin] = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
    }
    return models


def _distribution_summary(y: np.ndarray) -> Dict[str, float]:
    """Return class distribution summary as percentages."""

    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    if total == 0:
        return {}
    return {
        str(int(cls)): float(count) / float(total) for cls, count in zip(unique, counts)
    }


def _compute_metrics(
    *,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    target_name_map: Dict[int, str],
) -> Dict[str, Any]:
    """Compute standard classification metrics."""

    unique_classes = np.unique(y_true)
    ordered_names = [
        target_name_map.get(int(cls), str(int(cls))) for cls in unique_classes
    ]

    report = classification_report(
        y_true,
        y_pred,
        labels=unique_classes,
        target_names=ordered_names,
        zero_division=0,
        output_dict=True,
    )

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=unique_classes
        ).tolist(),
        "classification_report": report,
        "classes": unique_classes.astype(int).tolist(),
    }

    if y_proba is not None and y_proba.ndim == 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    elif y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

    metrics["model"] = model_name
    return metrics


def evaluate_models(
    *,
    models: Dict[str, ClassifierMixin],
    dataset,
    target_name_map: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """Train models using subject-disjoint splits and gather metrics."""

    results: Dict[str, Dict[str, Any]] = {}
    combined_train_X = np.vstack([dataset.train.X, dataset.val.X])
    combined_train_y = np.concatenate([dataset.train.y, dataset.val.y])

    for name, estimator in models.items():
        print(
            f"\n[Baseline] Training {name} on train subjects: {dataset.train_subjects}"
        )
        model = clone(estimator)
        model.fit(dataset.train.X, dataset.train.y)

        y_val_pred = model.predict(dataset.val.X)
        try:
            y_val_proba = model.predict_proba(dataset.val.X)
        except AttributeError:
            y_val_proba = None

        if isinstance(y_val_proba, np.ndarray) and y_val_proba.ndim == 2:
            if y_val_proba.shape[1] == 2:
                y_val_proba_scored = y_val_proba[:, 1]
            else:
                y_val_proba_scored = None
        elif isinstance(y_val_proba, np.ndarray) and y_val_proba.ndim == 1:
            y_val_proba_scored = y_val_proba
        else:
            y_val_proba_scored = None

        val_metrics = _compute_metrics(
            model_name=name,
            y_true=dataset.val.y,
            y_pred=y_val_pred,
            y_proba=y_val_proba_scored,
            target_name_map=target_name_map,
        )

        print(
            f"[Baseline] Validation accuracy={val_metrics['accuracy']:.3f} "
            f"macro_f1={val_metrics['macro_f1']:.3f}"
        )

        # Retrain on train + val before testing to follow baseline convention.
        test_model = clone(estimator)
        print(
            f"[Baseline] Re-fitting {name} on train+val subjects: "
            f"{dataset.train_subjects + dataset.val_subjects}"
        )
        test_model.fit(combined_train_X, combined_train_y)
        y_test_pred = test_model.predict(dataset.test.X)

        try:
            y_test_proba = test_model.predict_proba(dataset.test.X)
        except AttributeError:
            y_test_proba = None

        if isinstance(y_test_proba, np.ndarray) and y_test_proba.ndim == 2:
            if y_test_proba.shape[1] == 2:
                y_test_proba_scored = y_test_proba[:, 1]
            else:
                y_test_proba_scored = None
        elif isinstance(y_test_proba, np.ndarray) and y_test_proba.ndim == 1:
            y_test_proba_scored = y_test_proba
        else:
            y_test_proba_scored = None

        test_metrics = _compute_metrics(
            model_name=name,
            y_true=dataset.test.y,
            y_pred=y_test_pred,
            y_proba=y_test_proba_scored,
            target_name_map=target_name_map,
        )

        print(
            f"[Baseline] Test subjects={dataset.test_subjects} "
            f"accuracy={test_metrics['accuracy']:.3f} "
            f"macro_f1={test_metrics['macro_f1']:.3f}"
        )

        results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
        }

    return results


def _target_name_map(label_strategy: str) -> Dict[int, str]:
    if label_strategy == "binary":
        return {0: "low", 1: "elevated"}
    return {level: str(level) for level in range(1, 6)}


def _split_summary(dataset) -> Dict[str, Any]:
    """Return a summary friendly for JSON storage."""

    return {
        "train_subjects": dataset.train_subjects,
        "val_subjects": dataset.val_subjects,
        "test_subjects": dataset.test_subjects,
        "train_samples": int(dataset.train.X.shape[0]),
        "val_samples": int(dataset.val.X.shape[0]),
        "test_samples": int(dataset.test.X.shape[0]),
        "train_distribution": _distribution_summary(dataset.train.y),
        "val_distribution": _distribution_summary(dataset.val.y),
        "test_distribution": _distribution_summary(dataset.test.y),
        "feature_names": dataset.feature_names,
        "label_strategy": dataset.label_strategy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline models for SWEET sample subjects with subject-disjoint splits "
            "(see docs/Context.md for the rationale)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/SWEET/sample_subjects"),
        help="Directory containing SWEET sample subjects.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("baseline_results/sweet_samples"),
        help="Directory to store metrics JSON.",
    )
    parser.add_argument(
        "--label-strategy",
        choices=["binary", "ordinal"],
        default="binary",
        help="Binary treats stress>=2 as elevated; ordinal keeps 1..5 levels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Threshold for elevated stress in binary mode.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Fraction of subjects allocated to training split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of subjects allocated to validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for subject splitting and model reproducibility.",
    )

    args = parser.parse_args()

    print("[Baseline] Loading SWEET sample dataset...")
    dataset = load_sweet_sample_dataset(
        data_dir=args.data_dir,
        label_strategy=args.label_strategy,
        elevated_threshold=args.threshold,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        random_state=args.random_state,
    )

    print(
        "[Baseline] Subject partitions "
        f"(train {dataset.train_subjects} | "
        f"val {dataset.val_subjects} | "
        f"test {dataset.test_subjects})"
    )
    print(
        "[Baseline] Samples per split "
        f"train={dataset.train.X.shape[0]} "
        f"val={dataset.val.X.shape[0]} "
        f"test={dataset.test.X.shape[0]}"
    )

    models = _build_models(random_state=args.random_state)
    results = evaluate_models(
        models=models,
        dataset=dataset,
        target_name_map=_target_name_map(args.label_strategy),
    )

    summary = _split_summary(dataset)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_payload = {
        "summary": summary,
        "results": results,
        "config": {
            "data_dir": str(args.data_dir),
            "label_strategy": args.label_strategy,
            "threshold": args.threshold,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "random_state": args.random_state,
        },
    }

    output_file = output_dir / "sweet_sample_baseline_metrics.json"
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)

    print(f"[Baseline] Metrics written to {output_file}")


if __name__ == "__main__":
    main()
