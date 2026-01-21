#!/usr/bin/env python3
"""Analyze SWEET dataset and improve baseline model training.

This script helps diagnose issues with the baseline model and suggests improvements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Add src to path
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


def analyze_dataset(data_dir: str, label_strategy: str = "ordinal"):
    """Analyze SWEET dataset for potential issues."""

    print("=" * 80)
    print("SWEET Dataset Analysis")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset (label_strategy={label_strategy})...")
    dataset = load_sweet_sample_dataset(
        data_dir=data_dir,
        label_strategy=label_strategy,
        elevated_threshold=3.0,
        train_fraction=0.7,
        val_fraction=0.2,
        random_state=42,
    )

    # Basic stats
    print("\n1. Dataset Size:")
    print(
        f"   Train: {len(dataset.train.y)} samples, {len(dataset.train_subjects)} subjects"
    )
    print(
        f"   Val:   {len(dataset.val.y)} samples, {len(dataset.val_subjects)} subjects"
    )
    print(
        f"   Test:  {len(dataset.test.y)} samples, {len(dataset.test_subjects)} subjects"
    )
    print(f"   Features: {len(dataset.feature_names)}")

    # Class distribution
    if label_strategy == "binary":
        print("\n2. Class Distribution (binary: 0=no stress, 1=stress):")
    elif label_strategy == "ordinal":
        print("\n2. Class Distribution (ordinal: 1-5 stress levels):")
    elif label_strategy == "ordinal_3class":
        print("\n2. Class Distribution (3-class: 0=low, 1=medium, 2=high stress):")
    else:
        print(f"\n2. Class Distribution ({label_strategy}):")

    unique_classes = np.unique(dataset.train.y)
    print(f"   Unique classes: {unique_classes}")

    # Use unique with return_counts for sparse class indices
    train_unique, train_counts = np.unique(
        dataset.train.y.astype(int), return_counts=True
    )
    val_unique, val_counts = np.unique(dataset.val.y.astype(int), return_counts=True)
    test_unique, test_counts = np.unique(dataset.test.y.astype(int), return_counts=True)

    train_class_dict = dict(zip(train_unique, train_counts))
    val_class_dict = dict(zip(val_unique, val_counts))
    test_class_dict = dict(zip(test_unique, test_counts))

    print(f"\n   Train distribution:")
    for cls in unique_classes:
        count = train_class_dict.get(int(cls), 0)
        pct = count / len(dataset.train.y) * 100
        print(f"     Class {int(cls)}: {count:4d} samples ({pct:5.1f}%)")

    print(f"\n   Val distribution:")
    for cls in unique_classes:
        count = val_class_dict.get(int(cls), 0)
        pct = count / len(dataset.val.y) * 100
        print(f"     Class {int(cls)}: {count:4d} samples ({pct:5.1f}%)")

    print(f"\n   Test distribution:")
    for cls in unique_classes:
        count = test_class_dict.get(int(cls), 0)
        pct = count / len(dataset.test.y) * 100
        print(f"     Class {int(cls)}: {count:4d} samples ({pct:5.1f}%)")

    # Check for class imbalance
    all_counts = np.array(list(train_class_dict.values()))
    train_imbalance = max(all_counts) / min(all_counts)
    print(f"\n   ⚠️  Class imbalance ratio: {train_imbalance:.2f}:1")
    if train_imbalance > 2:
        print(f"   ⚠️  HIGH IMBALANCE! Consider using class_weight")

    # Feature statistics
    print("\n3. Feature Statistics:")
    print(f"   Features: {', '.join(dataset.feature_names[:5])}...")

    # Check for NaN or Inf
    nan_count = np.isnan(dataset.train.X).sum()
    inf_count = np.isinf(dataset.train.X).sum()
    print(f"   NaN values: {nan_count}")
    print(f"   Inf values: {inf_count}")

    # Feature ranges
    print(f"\n   Feature ranges (before scaling):")
    for i, name in enumerate(dataset.feature_names[:5]):
        values = dataset.train.X[:, i]
        print(
            f"   - {name}: [{values.min():.2f}, {values.max():.2f}], "
            f"mean={values.mean():.2f}, std={values.std():.2f}"
        )
    print(f"   ... ({len(dataset.feature_names) - 5} more features)")

    # Suggestions
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    suggestions = []

    if train_imbalance > 2:
        suggestions.append(
            "1. CLASS IMBALANCE: Use class weights in loss function:\n"
            "   --use-class-weight flag in training script"
        )

    if len(dataset.train.y) < 1000:
        suggestions.append(
            "2. SMALL DATASET: Consider:\n"
            "   - Reduce model complexity (smaller hidden layers)\n"
            "   - Increase regularization (dropout=0.3-0.5)\n"
            "   - Use more epochs with early stopping"
        )

    suggestions.append(
        "3. TRAINING IMPROVEMENTS:\n"
        "   - Try different learning rates: 0.0001, 0.001, 0.01\n"
        "   - Use learning rate scheduler (ReduceLROnPlateau)\n"
        "   - Try different optimizers (AdamW with weight_decay)\n"
        "   - Increase batch size if memory allows (32, 64)"
    )

    suggestions.append(
        "4. MODEL ARCHITECTURE:\n"
        "   - Try simpler model: [64, 32] instead of [128, 64]\n"
        "   - Try deeper model: [128, 64, 32]\n"
        "   - Adjust dropout rate (0.2 to 0.4)"
    )

    for sugg in suggestions:
        print(f"\n{sugg}")

    # Recommended command
    print("\n" + "=" * 80)
    print("RECOMMENDED TRAINING COMMAND:")
    print("=" * 80)

    num_classes = len(unique_classes)

    if label_strategy == "ordinal":
        print(f"\npython scripts/prepare_sweet_baseline.py `")
        print(f"    --data-dir {data_dir} `")
        print(f"    --output-dir baseline_models/sweet `")
        print(f"    --label-strategy ordinal `")
        print(f"    --hidden-dims 64 32 `")
        print(f"    --lr 0.001 `")
        print(f"    --epochs 100 `")
        print(f"    --batch-size 32")
        print(f"\nNote: Using ordinal labels (1-5) for {num_classes} stress levels")
    elif label_strategy == "ordinal_3class":
        print(f"\npython scripts/prepare_sweet_baseline.py `")
        print(f"    --data-dir {data_dir} `")
        print(f"    --output-dir baseline_models/sweet `")
        print(f"    --label-strategy ordinal_3class `")
        print(f"    --hidden-dims 64 32 `")
        print(f"    --lr 0.001 `")
        print(f"    --epochs 100 `")
        print(f"    --batch-size 32")
        print(f"\nNote: Using 3-class labels (0=low, 1=medium, 2=high stress)")
    else:
        print(f"\npython scripts/prepare_sweet_baseline.py `")
        print(f"    --data-dir {data_dir} `")
        print(f"    --output-dir baseline_models/sweet `")
        print(f"    --label-strategy binary `")
        print(f"    --threshold 3.0 `")
        print(f"    --hidden-dims 64 32 `")
        print(f"    --lr 0.001 `")
        print(f"    --epochs 100 `")
        print(f"    --batch-size 32")

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SWEET dataset and provide training recommendations"
    )
    parser.add_argument(
        "--data-dir",
        default="data/SWEET/selection1/users",
        help="Directory containing user data",
    )
    parser.add_argument(
        "--label-strategy",
        choices=["binary", "ordinal", "ordinal_3class"],
        default="ordinal",
        help="Label strategy to analyze",
    )

    args = parser.parse_args()

    try:
        analyze_dataset(args.data_dir, args.label_strategy)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
