#!/usr/bin/env python3
"""Run subject-aware cross-validation for WESAD, SWELL, combined, and SWEET datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets.multimodal import load_real_multimodal_dataset
from flower_basic.datasets.sweet_samples import load_sweet_sample_full
from flower_basic.datasets.swell import load_swell_dataset
from flower_basic.datasets.wesad import WESAD_SUBJECTS, load_wesad_dataset


def _stack_split(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_subject_ids: Iterable[str],
    test_subject_ids: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine train/test partitions into full dataset arrays."""
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    groups = np.concatenate(
        [
            np.asarray(list(train_subject_ids)),
            np.asarray(list(test_subject_ids)),
        ]
    )
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), groups


def _run_group_cv(
    dataset: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute subject-aware GroupKFold CV and return per-fold and summary metrics."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    folds = min(n_splits, unique_groups.size)
    if folds < 2:
        raise ValueError(
            f"Dataset '{dataset}' requires at least two distinct subjects; got {unique_groups.size}"
        )

    splitter = GroupKFold(n_splits=folds)
    records = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_train_scaled, y_train)
        lr_pred = lr.predict(X_test_scaled)
        records.append(
            {
                "dataset": dataset,
                "fold": fold_idx,
                "model": "logistic_regression",
                "accuracy": accuracy_score(y_test, lr_pred),
                "macro_f1": f1_score(y_test, lr_pred, average="macro", zero_division=0),
            }
        )

        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        records.append(
            {
                "dataset": dataset,
                "fold": fold_idx,
                "model": "random_forest",
                "accuracy": accuracy_score(y_test, rf_pred),
                "macro_f1": f1_score(y_test, rf_pred, average="macro", zero_division=0),
            }
        )

    fold_df = pd.DataFrame.from_records(records)
    summary = (
        fold_df.groupby(["dataset", "model"])
        .agg({"accuracy": ["mean", "std"], "macro_f1": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = [
        "dataset",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
    ]
    summary["n_folds"] = folds
    return fold_df, summary


def _load_wesad_full() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load WESAD once and return full dataset arrays and subject IDs."""
    X_train, X_test, y_train, y_test, info = load_wesad_dataset(
        normalize=False,
        test_size=5 / len(WESAD_SUBJECTS),
        return_subject_info=True,
    )
    return _stack_split(
        X_train,
        X_test,
        y_train,
        y_test,
        info["train_subject_ids"],
        info["test_subject_ids"],
    )


def _load_swell_full() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SWELL once and return full dataset arrays and subject IDs."""
    X_train, X_test, y_train, y_test, info = load_swell_dataset(
        modalities=["computer"],
        normalize_features=False,
        return_subject_info=True,
    )
    return _stack_split(
        X_train,
        X_test,
        y_train,
        y_test,
        info["train_subject_ids"],
        info["test_subject_ids"],
    )


def _load_combined_full() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined WESAD+SWELL dataset and return full arrays with prefixed IDs."""
    wesad_kwargs: dict[str, object] = {"normalize": False}
    swell_kwargs: dict[str, object] = {
        "normalize_features": False,
        "modalities": ["computer"],
    }

    _, _, combined = load_real_multimodal_dataset(
        wesad_kwargs=wesad_kwargs,
        swell_kwargs=swell_kwargs,
        include_dataset_indicator=True,
    )

    X = np.vstack([combined.X_train, combined.X_test])
    y = np.concatenate([combined.y_train, combined.y_test])
    groups = np.concatenate([combined.train_subject_ids, combined.test_subject_ids])
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), groups


def _load_sweet_samples_full(
    label_strategy: str = "binary",
    threshold: float = 2.0,
    min_samples: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SWEET sample-subject dataset and return arrays with subject IDs."""
    X, y, groups, _ = load_sweet_sample_full(
        label_strategy=label_strategy,
        elevated_threshold=threshold,
        min_samples_per_subject=min_samples,
    )
    return X, y, groups


def run_subject_cv(
    output_dir: Path,
    datasets: Iterable[str] | None = None,
    n_folds: int = 5,
    sweet_label_strategy: str = "binary",
    sweet_threshold: float = 2.0,
    sweet_min_samples: int = 5,
) -> None:
    """Execute CV for selected datasets and persist per-fold and summary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_loaders = {
        "wesad": _load_wesad_full,
        "swell": _load_swell_full,
        "combined": _load_combined_full,
        "sweet_samples": lambda ls=sweet_label_strategy, th=sweet_threshold, ms=sweet_min_samples: _load_sweet_samples_full(  # noqa: E501
            label_strategy=ls, threshold=th, min_samples=ms
        ),
    }

    selected = list(datasets) if datasets is not None else list(dataset_loaders.keys())

    fold_frames = []
    summary_frames = []

    for dataset_name in selected:
        if dataset_name not in dataset_loaders:
            raise ValueError(
                f"Dataset '{dataset_name}' is not supported. "
                f"Available options: {sorted(dataset_loaders.keys())}"
            )

        loader = dataset_loaders[dataset_name]
        print(f"Running subject-based cross-validation for {dataset_name}...")
        X, y, groups = loader()
        fold_df, summary_df = _run_group_cv(
            dataset_name, X, y, groups, n_splits=n_folds
        )
        fold_frames.append(fold_df)
        summary_frames.append(summary_df)

        for _, row in summary_df.iterrows():
            print(
                f"  {row['model']}: accuracy {row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f} | "
                f"macro-F1 {row['macro_f1_mean']:.3f} +/- {row['macro_f1_std']:.3f} (n={int(row['n_folds'])})"
            )

    fold_results = pd.concat(fold_frames, ignore_index=True)
    summary_results = pd.concat(summary_frames, ignore_index=True)

    fold_csv = output_dir / "subject_cv_folds.csv"
    summary_csv = output_dir / "subject_cv_summary.csv"
    summary_json = output_dir / "subject_cv_summary.json"

    fold_results.to_csv(fold_csv, index=False)
    summary_results.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    fold_path = fold_csv.resolve()
    summary_path = summary_csv.resolve()
    summary_json_path = summary_json.resolve()

    print("\nSaved cross-validation metrics to:")
    print(f"- {fold_path.relative_to(PROJECT_ROOT)}")
    print(f"- {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"- {summary_json_path.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run subject-aware GroupKFold cross-validation for WESAD, SWELL, "
            "combined multimodal, and SWEET sample-subject datasets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subject_cv_results"),
        help="Directory where CSV/JSON metrics will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["wesad", "swell", "combined", "sweet_samples"],
        help="Subset of datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Maximum number of cross-validation folds (capped by subject count).",
    )
    parser.add_argument(
        "--sweet-label-strategy",
        choices=["binary", "ordinal"],
        default="binary",
        help="Label strategy for SWEET sample subjects (default: binary).",
    )
    parser.add_argument(
        "--sweet-threshold",
        type=float,
        default=2.0,
        help="Stress threshold used when SWEET labels are binarized.",
    )
    parser.add_argument(
        "--sweet-min-samples",
        type=int,
        default=5,
        help="Minimum aligned samples required to include a SWEET subject.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_subject_cv(
        output_dir=args.output_dir,
        datasets=args.datasets,
        n_folds=args.folds,
        sweet_label_strategy=args.sweet_label_strategy,
        sweet_threshold=args.sweet_threshold,
        sweet_min_samples=args.sweet_min_samples,
    )


if __name__ == "__main__":
    main()
