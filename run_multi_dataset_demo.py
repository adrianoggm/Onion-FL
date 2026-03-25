#!/usr/bin/env python3
"""Multi-dataset demo using real WESAD and SWELL data."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flower_basic.datasets.multimodal import (
    CombinedDataset,
    DatasetSplit,
    load_real_multimodal_dataset,
)
from flower_basic.evaluation import group_cross_validation

DATA_DIR = Path.cwd()


def _class_distribution(labels: np.ndarray) -> dict[int, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique_labels, counts)}


def _split_by_subject(
    subject_ids: np.ndarray,
    test_size: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create boolean masks for a subject-based split."""

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


def _summarize_split(split: DatasetSplit) -> dict[str, object]:
    """Return key statistics for a dataset split."""
    train_dist = _class_distribution(split.y_train)
    test_dist = _class_distribution(split.y_test)

    return {
        "name": split.name,
        "train_samples": int(split.X_train.shape[0]),
        "test_samples": int(split.X_test.shape[0]),
        "n_features": int(split.X_train.shape[1]),
        "train_subjects": int(np.unique(split.train_subject_ids).size),
        "train_distribution": train_dist,
        "test_subjects": int(np.unique(split.test_subject_ids).size),
        "test_distribution": test_dist,
    }


def _evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Train baseline classifiers and capture accuracy/F1 metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        }

    return results


def _build_full_dataset(
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack train/test partitions into full dataset arrays."""

    X_full = np.vstack([split.X_train, split.X_test])
    y_full = np.concatenate([split.y_train, split.y_test])
    groups_full = np.concatenate([split.train_subject_ids, split.test_subject_ids])
    return X_full, y_full, groups_full


def _evaluate_dataset(split: DatasetSplit) -> dict[str, object]:
    """Evaluate individual dataset baselines."""
    metrics = _evaluate_models(split.X_train, split.y_train, split.X_test, split.y_test)
    summary = _summarize_split(split)
    summary["metrics"] = metrics

    X_full, y_full, groups_full = _build_full_dataset(split)
    summary["cross_validation"] = group_cross_validation(X_full, y_full, groups_full)
    return summary


def _evaluate_combined(combined: CombinedDataset) -> dict[str, object]:
    """Evaluate the combined multimodal baseline with a validation split."""

    train_mask, val_mask = _split_by_subject(
        combined.train_subject_ids,
        test_size=0.2,
        random_state=42,
    )

    X_train = combined.X_train[train_mask]
    y_train = combined.y_train[train_mask]
    X_val = combined.X_train[val_mask]
    y_val = combined.y_train[val_mask]
    train_sources = combined.train_sources[train_mask]
    val_sources = combined.train_sources[val_mask]

    metrics = _evaluate_models(X_train, y_train, combined.X_test, combined.y_test)
    val_metrics = _evaluate_models(X_train, y_train, X_val, y_val)

    X_full = np.vstack([combined.X_train, combined.X_test])
    y_full = np.concatenate([combined.y_train, combined.y_test])
    groups_full = np.concatenate(
        [combined.train_subject_ids, combined.test_subject_ids]
    )

    return {
        "name": "multimodal",
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(combined.X_test.shape[0]),
        "n_features": int(combined.X_train.shape[1]),
        "train_subjects": int(np.unique(combined.train_subject_ids[train_mask]).size),
        "validation_subjects": int(
            np.unique(combined.train_subject_ids[val_mask]).size
        ),
        "test_subjects": int(np.unique(combined.test_subject_ids).size),
        "train_sources": {
            "wesad": int(np.sum(train_sources == "wesad")),
            "swell": int(np.sum(train_sources == "swell")),
        },
        "validation_sources": {
            "wesad": int(np.sum(val_sources == "wesad")),
            "swell": int(np.sum(val_sources == "swell")),
        },
        "test_sources": {
            "wesad": int(np.sum(combined.test_sources == "wesad")),
            "swell": int(np.sum(combined.test_sources == "swell")),
        },
        "metrics": metrics,
        "validation_metrics": val_metrics,
        "cross_validation": group_cross_validation(X_full, y_full, groups_full),
    }


def _write_markdown(summary: dict[str, object], path: Path) -> None:
    """Write a concise Markdown report for a dataset."""
    metrics = summary["metrics"]
    lines = [
        f"# {summary['name'].upper()} Baseline",
        "",
        f"* Train samples: {summary['train_samples']}",
        f"* Test samples: {summary['test_samples']}",
        f"* Features: {summary['n_features']}",
        "",
        "| Model | Accuracy | Macro F1 |",
        "| --- | --- | --- |",
    ]

    for model_name, metric_values in metrics.items():
        lines.append(
            f"| {model_name} | {metric_values['accuracy']:.3f} | {metric_values['macro_f1']:.3f} |"
        )

    cv = summary.get("cross_validation")
    if cv:
        lines.append("")
        lines.append(
            "**5-fold Group Cross-Validation (accuracy +/- std / macro-F1 +/- std)**"
        )
        for model_name, stats in cv.items():
            lines.append(
                f"- {model_name}: {stats['accuracy_mean']:.3f} +/- {stats['accuracy_std']:.3f} / "
                f"{stats['macro_f1_mean']:.3f} +/- {stats['macro_f1_std']:.3f}"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> bool:
    """Run the multi-dataset demo with real data only."""
    print("Loading real WESAD and SWELL datasets...")
    wesad_split, swell_split, combined = load_real_multimodal_dataset()

    print("\nDataset summary:")
    for summary in (_summarize_split(wesad_split), _summarize_split(swell_split)):
        print(
            f"- {summary['name'].upper()}: {summary['train_samples']} train ({summary['train_subjects']} subjects) / "
            f"{summary['test_samples']} test ({summary['test_subjects']} subjects) with {summary['n_features']} features"
        )

    wesad_report = _evaluate_dataset(wesad_split)
    swell_report = _evaluate_dataset(swell_split)
    multimodal_report = _evaluate_combined(combined)

    report = {
        "wesad": wesad_report,
        "swell": swell_report,
        "multimodal": multimodal_report,
    }

    output_json = DATA_DIR / "multi_dataset_demo_report.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _write_markdown(wesad_report, DATA_DIR / "WESAD_BASELINE_RESULTS.md")
    _write_markdown(swell_report, DATA_DIR / "SWELL_BASELINE_RESULTS.md")

    print("\nReports written to:")
    print(f"- {output_json.relative_to(DATA_DIR)}")
    print("- WESAD_BASELINE_RESULTS.md")
    print("- SWELL_BASELINE_RESULTS.md")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        raise SystemExit(1)
