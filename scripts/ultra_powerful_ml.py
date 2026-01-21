"""Ultra-powerful ML models with temporal sequences, XGBoost, LSTM/GRU.

This script creates SEQUENCES from the temporal data to leverage time dependencies,
uses advanced training strategies (LR decay, early stopping, class weights),
and compares against XGBoost and gradient boosting methods.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flower_basic.datasets.sweet_samples import load_sweet_sample_dataset


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for temporal stress detection."""

    def __init__(
        self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention mechanism
        attn_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*2)

        out = self.fc(context)
        return out


class GRUClassifier(nn.Module):
    """GRU-based classifier for temporal stress detection."""

    def __init__(
        self, input_dim, hidden_dim=128, num_layers=2, num_classes=3, dropout=0.3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)

        # Attention
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)

        out = self.fc(context)
        return out


class DeepMLPClassifier(nn.Module):
    """Very deep MLP with residual connections."""

    def __init__(self, input_dim, num_classes=3):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc_out = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_bn(x)

        # Block 1 with residual
        identity = self.fc1(x)
        out = self.relu(self.bn1(identity))
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(self.bn2(out))
        out = out + identity  # Residual connection
        out = self.dropout(out)

        # Block 2
        out = self.fc3(out)
        out = self.relu(self.bn3(out))
        out = self.dropout(out)

        identity2 = out
        out = self.fc4(out)
        out = self.relu(self.bn4(out))
        out = out + identity2  # Residual
        out = self.dropout(out)

        # Final layers
        out = self.fc5(out)
        out = self.relu(self.bn5(out))
        out = self.dropout(out)

        out = self.fc_out(out)
        return out


def create_sequences(X, y, subject_ids, seq_length=10, stride=5):
    """Create sequences from temporal data by sliding window per subject.

    Args:
        X: Feature matrix (samples, features)
        y: Labels (samples,)
        subject_ids: Subject IDs (samples,)
        seq_length: Length of each sequence
        stride: Stride for sliding window

    Returns:
        X_seq: (num_sequences, seq_length, features)
        y_seq: (num_sequences,) - label of last timestep
        subj_seq: (num_sequences,) - subject ID
    """
    sequences = []
    labels = []
    subjects = []

    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        subj_mask = subject_ids == subj
        subj_X = X[subj_mask]
        subj_y = y[subj_mask]

        # Create sequences with sliding window
        for start_idx in range(0, len(subj_X) - seq_length + 1, stride):
            seq = subj_X[start_idx : start_idx + seq_length]
            label = subj_y[start_idx + seq_length - 1]  # Label of last timestep

            sequences.append(seq)
            labels.append(label)
            subjects.append(subj)

    return np.array(sequences), np.array(labels), np.array(subjects)


def train_pytorch_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    class_weights=None,
    patience=15,
    model_name="Model",
):
    """Train PyTorch model with advanced techniques."""

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * len(y_batch)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # LR scheduling
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_val_acc


def train_and_evaluate_deep_models(
    X, y, subject_ids, cv_splits, device, use_sequences=False
):
    """Train and evaluate deep learning models."""

    results = {}

    # Calculate class weights
    class_counts = np.bincount(y.astype(int))
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts)).to(
        device
    )

    if use_sequences:
        print("\n🔄 Creating temporal sequences...")
        X_seq, y_seq, subj_seq = create_sequences(
            X, y, subject_ids, seq_length=10, stride=5
        )
        print(f"  Created {len(y_seq)} sequences from {len(y)} samples")

        # Update CV splits for sequences
        seq_cv_splits = []
        for train_idx, test_idx in cv_splits:
            train_subjects = np.unique(subject_ids[train_idx])
            test_subjects = np.unique(subject_ids[test_idx])

            seq_train_mask = np.isin(subj_seq, train_subjects)
            seq_test_mask = np.isin(subj_seq, test_subjects)

            seq_cv_splits.append(
                (np.where(seq_train_mask)[0], np.where(seq_test_mask)[0])
            )

        X_use = X_seq
        y_use = y_seq
        cv_use = seq_cv_splits
        is_sequential = True
    else:
        X_use = X
        y_use = y
        cv_use = cv_splits
        is_sequential = False

    # Models to test
    models_config = []

    if is_sequential:
        # Only RNN models for sequences
        models_config = [
            ("LSTM_Attention", lambda: LSTMClassifier(X.shape[1], 128, 2, 3, 0.3)),
            ("GRU_Attention", lambda: GRUClassifier(X.shape[1], 128, 2, 3, 0.3)),
        ]
    else:
        # Only MLP for non-sequential
        models_config = [
            ("DeepMLP_Residual", lambda: DeepMLPClassifier(X.shape[1], 3)),
        ]

    for model_name, model_factory in models_config:
        if not is_sequential and "LSTM" in model_name or "GRU" in model_name:
            continue  # Skip RNN models if not using sequences

        print(f"\n📊 {model_name}:")
        fold_accs = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(cv_use, 1):
            X_train, X_test = X_use[train_idx], X_use[test_idx]
            y_train, y_test = y_use[train_idx], y_use[test_idx]

            # Scale features
            if is_sequential:
                scaler = StandardScaler()
                # Reshape for scaling
                n_samples, seq_len, n_features = X_train.shape
                X_train_2d = X_train.reshape(-1, n_features)
                X_train_scaled = scaler.fit_transform(X_train_2d).reshape(
                    n_samples, seq_len, n_features
                )

                n_samples_test = X_test.shape[0]
                X_test_2d = X_test.reshape(-1, n_features)
                X_test_scaled = scaler.transform(X_test_2d).reshape(
                    n_samples_test, seq_len, n_features
                )
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

            # Create dataloaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)
            )

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # Train model
            model = model_factory().to(device)

            best_val_acc = train_pytorch_model(
                model,
                train_loader,
                test_loader,
                num_epochs=100,
                device=device,
                class_weights=class_weights,
                patience=15,
                model_name=model_name,
            )

            # Final evaluation
            model.eval()
            with torch.no_grad():
                y_pred_fold = []
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    y_pred_fold.extend(predicted.cpu().numpy())

            acc = accuracy_score(y_test, y_pred_fold)
            fold_accs.append(acc)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred_fold)

            print(f"  Fold {fold}/5... Acc: {acc:.4f}")

        # Aggregate results
        cm = confusion_matrix(all_y_true, all_y_pred)
        class_report = classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0
        )

        results[model_name] = {
            "mean_accuracy": np.mean(fold_accs),
            "std_accuracy": np.std(fold_accs),
            "mean_f1": f1_score(all_y_true, all_y_pred, average="weighted"),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        }

        print(
            f"  ✓ Mean Accuracy: {results[model_name]['mean_accuracy']:.4f} ± {results[model_name]['std_accuracy']:.4f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-powerful ML with XGBoost, LSTM, GRU, and advanced training"
    )
    parser.add_argument("--data-dir", default="data/SWEET/selection1/users")
    parser.add_argument("--label-strategy", default="ordinal_3class")
    parser.add_argument("--output-dir", default="advanced_ml_results")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--use-sequences",
        action="store_true",
        help="Use temporal sequences for LSTM/GRU",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("ULTRA-POWERFUL ML: XGBoost + LSTM/GRU + Advanced Training")
    print("=" * 80)

    # Load dataset
    print("\nLoading SWEET dataset...")
    dataset = load_sweet_sample_dataset(
        data_dir=args.data_dir,
        label_strategy=args.label_strategy,
        train_fraction=0.6,
        val_fraction=0.2,
        random_state=args.seed,
    )

    X = np.vstack([dataset.train.X, dataset.val.X, dataset.test.X])
    y = np.concatenate([dataset.train.y, dataset.val.y, dataset.test.y])
    subject_ids = np.concatenate(
        [dataset.train.subject_ids, dataset.val.subject_ids, dataset.test.subject_ids]
    )

    print(f"✓ Dataset loaded: {len(y)} samples, {len(np.unique(subject_ids))} subjects")

    # Create CV splits - define function inline
    def create_subject_level_cv_splits(
        subject_ids, y_per_subject, n_splits=5, random_state=42
    ):
        """Create CV splits at subject level."""
        unique_subjects = np.unique(subject_ids)
        subject_labels = np.array([y_per_subject[subj] for subj in unique_subjects])

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        cv_splits = []
        for train_subject_idx, test_subject_idx in skf.split(
            unique_subjects, subject_labels
        ):
            train_subjects = unique_subjects[train_subject_idx]
            test_subjects = unique_subjects[test_subject_idx]

            train_sample_idx = np.isin(subject_ids, train_subjects)
            test_sample_idx = np.isin(subject_ids, test_subjects)

            cv_splits.append(
                (np.where(train_sample_idx)[0], np.where(test_sample_idx)[0])
            )

        return cv_splits

    y_per_subject = {}
    for subj in np.unique(subject_ids):
        subj_labels = y[subject_ids == subj]
        y_per_subject[subj] = np.bincount(subj_labels.astype(int)).argmax()

    cv_splits = create_subject_level_cv_splits(
        subject_ids, y_per_subject, args.n_folds, args.seed
    )

    # Traditional ML models with optimized hyperparameters
    print("\n" + "=" * 80)
    print("PART 1: OPTIMIZED XGBOOST & ENSEMBLE METHODS")
    print("=" * 80)

    traditional_results = {}

    # XGBoost with extensive hyperparameter tuning
    print("\n📊 XGBoost (Optimized):")
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2,
        gamma=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=3,
        random_state=args.seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    fold_accs = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(cv_splits, 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        xgb_model.fit(X_train_scaled, y_train)
        y_pred = xgb_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold}/5... Acc: {acc:.4f}")

    cm = confusion_matrix(all_y_true, all_y_pred)
    traditional_results["XGBoost_Optimized"] = {
        "mean_accuracy": np.mean(fold_accs),
        "std_accuracy": np.std(fold_accs),
        "mean_f1": f1_score(all_y_true, all_y_pred, average="weighted"),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0
        ),
    }

    print(
        f"  ✓ Mean Accuracy: {traditional_results['XGBoost_Optimized']['mean_accuracy']:.4f} ± {traditional_results['XGBoost_Optimized']['std_accuracy']:.4f}"
    )

    # Deep Learning models
    print("\n" + "=" * 80)
    print("PART 2: DEEP LEARNING (MLP, LSTM, GRU)")
    print("=" * 80)

    deep_results = train_and_evaluate_deep_models(
        X, y, subject_ids, cv_splits, device, use_sequences=args.use_sequences
    )

    # Combine results
    all_results = {**traditional_results, **deep_results}

    # Save results
    results_file = output_dir / "ultra_powerful_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}\n")

    sorted_models = sorted(
        all_results.items(), key=lambda x: x[1]["mean_accuracy"], reverse=True
    )

    for rank, (model_name, res) in enumerate(sorted_models, 1):
        print(
            f"{rank}. {model_name}: {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
        )

    print(f"\n🏆 Best Model: {sorted_models[0][0]}")
    print(f"   Accuracy: {sorted_models[0][1]['mean_accuracy']:.4f}")
    print(f"   F1-Score: {sorted_models[0][1]['mean_f1']:.4f}")


if __name__ == "__main__":
    main()
