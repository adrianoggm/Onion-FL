"""Extreme deep architectures: Ultra-deep XGBoost and MLP configurations."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_full


def create_subject_level_cv_splits(data_dict, n_folds=5, seed=42):
    """Create subject-level CV splits."""
    rng = np.random.RandomState(seed)

    subjects = sorted(set(data_dict["subjects"]))
    rng.shuffle(subjects)

    from collections import Counter

    subject_class_counts = {}
    for subj in subjects:
        subj_labels = [
            l for s, l in zip(data_dict["subjects"], data_dict["labels"]) if s == subj
        ]
        subject_class_counts[subj] = Counter(subj_labels).most_common(1)[0][0]

    class_subjects = {}
    for subj, cls in subject_class_counts.items():
        if cls not in class_subjects:
            class_subjects[cls] = []
        class_subjects[cls].append(subj)

    folds = [[] for _ in range(n_folds)]
    for cls, cls_subjects in class_subjects.items():
        rng.shuffle(cls_subjects)
        for i, subj in enumerate(cls_subjects):
            folds[i % n_folds].append(subj)

    cv_splits = []
    for fold_idx in range(n_folds):
        test_subjects = set(folds[fold_idx])
        train_indices = [
            i for i, s in enumerate(data_dict["subjects"]) if s not in test_subjects
        ]
        test_indices = [
            i for i, s in enumerate(data_dict["subjects"]) if s in test_subjects
        ]
        cv_splits.append((train_indices, test_indices))

    return cv_splits


class UltraDeepMLP(nn.Module):
    """Ultra-deep MLP with residual connections, batch norm, and dropout."""

    def __init__(self, input_dim, num_classes=3, depth=20, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Deep residual blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.blocks.append(block)

        # Progressive dimensionality reduction
        self.reduction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output = nn.Linear(hidden_dim // 8, num_classes)

    def forward(self, x):
        x = self.input_proj(x)

        # Deep residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection
            x = torch.relu(x)

        x = self.reduction(x)
        x = self.output(x)
        return x


class VeryWideDeepMLP(nn.Module):
    """Very wide and deep MLP with attention."""

    def __init__(
        self,
        input_dim,
        num_classes=3,
        hidden_dims=[1024, 1024, 512, 512, 256, 256, 128, 128],
        dropout=0.4,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Self-attention layer
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.Tanh(),
            nn.Linear(prev_dim // 2, 1),
            nn.Softmax(dim=1),
        )

        self.output = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)

        # Self-attention (treat features as sequence)
        attn_weights = self.attention(x.unsqueeze(1))
        x = x * attn_weights.squeeze(1)

        x = self.output(x)
        return x


class PyramidMLP(nn.Module):
    """Pyramid architecture: wide input, narrow middle, wide output."""

    def __init__(
        self,
        input_dim,
        num_classes=3,
        max_width=2048,
        min_width=64,
        depth=12,
        dropout=0.3,
    ):
        super().__init__()

        # Encoder: expand then compress
        encoder_layers = []
        dims = []

        # Expansion phase
        for i in range(depth // 2):
            if i == 0:
                in_dim = input_dim
                out_dim = max_width // (2 ** (depth // 2 - 1 - i))
            else:
                in_dim = dims[-1]
                out_dim = min(max_width, in_dim * 2)

            dims.append(out_dim)
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        # Compression phase
        for i in range(depth // 2):
            in_dim = dims[-1]
            out_dim = max(min_width, in_dim // 2)
            dims.append(out_dim)
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )

        self.encoder = nn.Sequential(*encoder_layers)
        self.output = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output(x)
        return x


def train_pytorch_model(
    model, train_loader, val_loader, device, epochs=200, lr=0.001, patience=30
):
    """Train PyTorch model with advanced techniques."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def evaluate_extreme_models(
    X_train_full, y_train_full, subjects_full, cv_splits, device
):
    """Evaluate extreme deep models."""

    results = {}

    # ========================================================================
    # 1. EXTREME XGBOOST CONFIGURATIONS
    # ========================================================================

    if HAS_XGBOOST:
        print("\n" + "=" * 80)
        print("EXTREME XGBOOST CONFIGURATIONS")
        print("=" * 80)

        xgb_configs = {
            "XGBoost_Ultra_Deep": {
                "n_estimators": 1000,
                "max_depth": 15,
                "learning_rate": 0.005,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "gamma": 0.1,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
            },
            "XGBoost_Very_Wide": {
                "n_estimators": 700,
                "max_depth": 8,
                "learning_rate": 0.01,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "min_child_weight": 1,
                "gamma": 0.2,
                "reg_alpha": 0.3,
                "reg_lambda": 1.5,
                "max_leaves": 255,
            },
            "XGBoost_Extreme_Regularized": {
                "n_estimators": 500,
                "max_depth": 10,
                "learning_rate": 0.01,
                "subsample": 0.5,
                "colsample_bytree": 0.5,
                "min_child_weight": 5,
                "gamma": 0.5,
                "reg_alpha": 2.0,
                "reg_lambda": 3.0,
            },
        }

        for name, config in xgb_configs.items():
            print(f"\n📊 {name}:")
            print(
                f"   Config: depth={config['max_depth']}, n_estimators={config['n_estimators']}, lr={config['learning_rate']}"
            )

            fold_accs = []
            fold_f1s = []
            all_preds = []
            all_labels = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                X_train, X_test = X_train_full[train_idx], X_train_full[test_idx]
                y_train, y_test = y_train_full[train_idx], y_train_full[test_idx]

                model = xgb.XGBClassifier(
                    objective="multi:softmax",
                    num_class=3,
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    **config,
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                fold_accs.append(acc)
                fold_f1s.append(f1)
                all_preds.extend(y_pred)
                all_labels.extend(y_test)

                print(f"   Fold {fold_idx+1}/5... Acc: {acc:.4f}")

            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            mean_f1 = np.mean(fold_f1s)

            print(f"   ✓ Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"   ✓ Mean F1-Score: {mean_f1:.4f}")

            results[name] = {
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "mean_f1": float(mean_f1),
                "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
                "config": config,
            }

    # ========================================================================
    # 2. EXTREME MLP ARCHITECTURES
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXTREME MLP ARCHITECTURES")
    print("=" * 80)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full)

    mlp_configs = {
        "UltraDeep_MLP_20layers": {
            "model_class": UltraDeepMLP,
            "kwargs": {"depth": 20, "hidden_dim": 512, "dropout": 0.3},
            "epochs": 200,
            "lr": 0.0005,
            "batch_size": 64,
        },
        "UltraDeep_MLP_30layers": {
            "model_class": UltraDeepMLP,
            "kwargs": {"depth": 30, "hidden_dim": 256, "dropout": 0.4},
            "epochs": 200,
            "lr": 0.0003,
            "batch_size": 64,
        },
        "VeryWide_Deep_MLP": {
            "model_class": VeryWideDeepMLP,
            "kwargs": {
                "hidden_dims": [1024, 1024, 512, 512, 256, 256, 128, 128],
                "dropout": 0.4,
            },
            "epochs": 200,
            "lr": 0.001,
            "batch_size": 64,
        },
        "Pyramid_MLP_12layers": {
            "model_class": PyramidMLP,
            "kwargs": {"max_width": 2048, "min_width": 64, "depth": 12, "dropout": 0.3},
            "epochs": 200,
            "lr": 0.001,
            "batch_size": 64,
        },
        "Pyramid_MLP_16layers": {
            "model_class": PyramidMLP,
            "kwargs": {
                "max_width": 1024,
                "min_width": 32,
                "depth": 16,
                "dropout": 0.35,
            },
            "epochs": 200,
            "lr": 0.0007,
            "batch_size": 64,
        },
    }

    for name, config in mlp_configs.items():
        print(f"\n📊 {name}:")
        print(f"   Architecture: {config['model_class'].__name__}")
        print(f"   Config: {config['kwargs']}")

        fold_accs = []
        fold_f1s = []
        all_preds = []
        all_labels = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_train_full[train_idx], y_train_full[test_idx]

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), torch.LongTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), torch.LongTensor(y_test)
            )

            train_loader = DataLoader(
                train_dataset, batch_size=config["batch_size"], shuffle=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=config["batch_size"], shuffle=False
            )

            # Create and train model
            model = config["model_class"](
                input_dim=X_train.shape[1], num_classes=3, **config["kwargs"]
            ).to(device)

            model = train_pytorch_model(
                model,
                train_loader,
                test_loader,
                device,
                epochs=config["epochs"],
                lr=config["lr"],
                patience=30,
            )

            # Evaluate
            model.eval()
            test_preds = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    test_preds.extend(preds)

            acc = accuracy_score(y_test, test_preds)
            f1 = f1_score(y_test, test_preds, average="weighted")

            fold_accs.append(acc)
            fold_f1s.append(f1)
            all_preds.extend(test_preds)
            all_labels.extend(y_test)

            print(f"   Fold {fold_idx+1}/5... Acc: {acc:.4f}")

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        mean_f1 = np.mean(fold_f1s)

        print(f"   ✓ Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"   ✓ Mean F1-Score: {mean_f1:.4f}")

        results[name] = {
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "mean_f1": float(mean_f1),
            "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
            "architecture": config["model_class"].__name__,
            "config": str(config["kwargs"]),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Extreme deep models for SWEET")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--label-strategy", type=str, default="ordinal_3class")
    parser.add_argument("--output-dir", type=str, default="extreme_deep_results")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("EXTREME DEEP ARCHITECTURES: XGBoost + MLP")
    print("=" * 80)

    # Load data
    print("\nLoading SWEET dataset...")
    X_train_full, y_train_full, subjects_full, feature_names = load_sweet_sample_full(
        data_dir=Path(args.data_dir), label_strategy=args.label_strategy
    )
    print(
        f"✓ Dataset loaded: {len(X_train_full)} samples, {len(set(subjects_full))} subjects"
    )

    # Create CV splits
    print("\nCreating subject-level CV splits...")
    data_dict = {
        "features": X_train_full,
        "labels": y_train_full,
        "subjects": subjects_full,
    }
    cv_splits = create_subject_level_cv_splits(data_dict, args.n_folds, args.seed)
    print(f"✓ Created {args.n_folds} folds (subject-level stratified)")

    # Evaluate models
    start_time = time.time()
    results = evaluate_extreme_models(
        X_train_full, y_train_full, subjects_full, cv_splits, device
    )
    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["mean_accuracy"], reverse=True
    )

    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"{rank}. {name}: {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}")

    if sorted_results:
        print(f"\n🏆 Best Model: {sorted_results[0][0]}")
        print(f"   Accuracy: {sorted_results[0][1]['mean_accuracy']:.4f}")
        print(f"   F1-Score: {sorted_results[0][1]['mean_f1']:.4f}")

    # Save results
    output_file = output_dir / "extreme_deep_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "results": results,
                "elapsed_time_minutes": elapsed_time / 60,
                "best_model": sorted_results[0][0] if sorted_results else None,
                "best_accuracy": (
                    sorted_results[0][1]["mean_accuracy"] if sorted_results else None
                ),
            },
            f,
            indent=2,
        )

    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Total time: {elapsed_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
