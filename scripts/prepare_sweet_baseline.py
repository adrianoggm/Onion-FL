#!/usr/bin/env python3
"""Prepare SWEET baseline model from selection1 data.

This script:
1. Loads data from data/SWEET/selection1 (expected to have user folders)
2. Creates 70/20/10 train/val/test splits (per-subject)
3. Trains a baseline SweetMLP model
4. Saves the baseline model to use for fine-tuning in federated learning

Usage:
    python scripts/prepare_sweet_baseline.py \\
        --data-dir data/SWEET/selection1 \\
        --output-dir baseline_models/sweet \\
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset
from flower_basic.sweet_model import SweetMLP, get_parameters


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
):
    """Train baseline model with memory-efficient approach."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\nTraining baseline model...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += len(y_batch)

            # Clear GPU cache if using CUDA
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * len(y_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += len(y_batch)

                # Clear GPU cache if using CUDA
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

    print("=" * 60)
    print(f"✓ Training completed. Best val accuracy: {best_val_acc:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SWEET baseline model from selection1 data"
    )
    parser.add_argument(
        "--data-dir",
        default="data/SWEET/selection1",
        help="Directory containing SWEET selection1 user folders",
    )
    parser.add_argument(
        "--output-dir",
        default="baseline_models/sweet",
        help="Output directory for baseline model",
    )
    parser.add_argument(
        "--label-strategy",
        choices=["binary", "ordinal", "ordinal_3class"],
        default="ordinal",
        help="Label strategy: 'ordinal_3class' uses 3 classes (RECOMMENDED), 'ordinal' uses full 1-5 range, 'binary' uses threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Threshold for binary classification (MAXIMUM_STRESS >= threshold → stressed)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Hidden layer dimensions",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (reduce if memory issues)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 = single process, safer for memory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SWEET Baseline Model Preparation")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Label strategy: {args.label_strategy}")
    print(f"Split: 60% train, 20% val, 20% test")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset with 70/20/10 split
    print("\nLoading SWEET dataset...")
    print("  Note: Processing incrementally to avoid memory issues...")
    try:
        dataset = load_sweet_sample_dataset(
            data_dir=args.data_dir,
            label_strategy=args.label_strategy,
            elevated_threshold=args.threshold,
            train_fraction=0.7,
            val_fraction=0.2,
            random_state=args.seed,
            min_samples_per_subject=5,
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("\nTip: If memory error, reduce batch size or use fewer subjects")
        sys.exit(1)

    print(f"✓ Dataset loaded successfully")
    print(
        f"  Train subjects: {len(dataset.train_subjects)} ({len(dataset.train.y)} samples)"
    )
    print(f"  Val subjects: {len(dataset.val_subjects)} ({len(dataset.val.y)} samples)")
    print(
        f"  Test subjects: {len(dataset.test_subjects)} ({len(dataset.test.y)} samples)"
    )
    print(f"  Features: {len(dataset.feature_names)}")
    print(
        f"  Memory optimization: batch_size={args.batch_size}, num_workers={args.num_workers}"
    )

    # Create data loaders with memory-efficient settings
    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(dataset.train.X), torch.LongTensor(dataset.train.y)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # Disable pin_memory to save RAM
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(dataset.val.X), torch.LongTensor(dataset.val.y)
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(dataset.test.X), torch.LongTensor(dataset.test.y)
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    num_classes = len(np.unique(dataset.train.y))
    model = SweetMLP(
        input_dim=len(dataset.feature_names),
        hidden_dims=args.hidden_dims,
        num_classes=num_classes,
    ).to(device)

    print(f"Model: {model.__class__.__name__}")
    print(f"  Input dim: {len(dataset.feature_names)}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Output classes: {num_classes}")

    # Train
    history = train_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == y_batch).sum().item()
            test_total += len(y_batch)

            # Clear GPU cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    test_acc = test_correct / test_total
    print(f"✓ Test accuracy: {test_acc:.4f}")

    # Save model and metadata
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "baseline_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "data_dir": args.data_dir,
        "label_strategy": args.label_strategy,
        "threshold": args.threshold,
        "input_dim": len(dataset.feature_names),
        "hidden_dims": args.hidden_dims,
        "num_classes": num_classes,
        "train_samples": len(dataset.train.y),
        "val_samples": len(dataset.val.y),
        "test_samples": len(dataset.test.y),
        "train_subjects": dataset.train_subjects,
        "val_subjects": dataset.val_subjects,
        "test_subjects": dataset.test_subjects,
        "feature_names": dataset.feature_names,
        "final_test_accuracy": test_acc,
        "final_val_accuracy": history["val_acc"][-1],
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    metadata_path = output_path / "baseline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"✓ Metadata saved to: {metadata_path}")

    # Save training history
    history_path = output_path / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"✓ Training history saved to: {history_path}")

    print("\n" + "=" * 80)
    print("✓ SWEET Baseline Model Preparation Complete")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nNext steps:")
    print("  1. Use this baseline model as initialization for federated learning")
    print("  2. Run prepare_sweet_federated.py to create federated splits")
    print("  3. Run run_sweet_federated_demo.py with the baseline model")


if __name__ == "__main__":
    main()
