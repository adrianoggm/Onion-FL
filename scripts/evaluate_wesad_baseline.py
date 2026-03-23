#!/usr/bin/env python3
"""
Baseline Performance Evaluation: WESAD Dataset
==============================================

Evaluates WESAD dataset performance using classical machine learning approach
with proper subject-based splitting to prevent dat        print("Training classical models...")

        # Check if we have training data
        if len(X_train) == 0:
            print("Error: No training data available!")
            return {}

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle validation set (may be empty for small datasets)
        if len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val)
        else:
            X_val_scaled = X_test_scaled  # Use test set for validation if no val set
            y_val = y_testge.

Split strategy:
- 50% subjects for training
- 20% subjects for validation
- 30% subjects for testing
- No subject data mixing between splits

This establishes a baseline for comparison with federated learning results.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class WESADBaselineEvaluator:
    """Baseline performance evaluator for WESAD dataset."""

    def __init__(self, data_dir: str = "data/WESAD"):
        self.data_dir = Path(data_dir)
        self.results = {}

    # STRICT RULES ENFORCEMENT:
    # 1. Evaluations MUST use COMPLETE datasets (100% WESAD/SWELL)
    # 2. NO samples allowed in evaluate_*.py (samples only for test_*.py)
    # 3. ABSOLUTELY NO mock data generation - REAL DATA ONLY
    # 4. Mock data generation is PROHIBITED for ML/AI evaluation

    def load_wesad_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load COMPLETE WESAD dataset - MANDATORY for evaluations."""
        print("Loading COMPLETE WESAD dataset...")

        # Available WESAD subjects (excluding S1 and S12)
        available_subjects = [
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "S11",
            "S13",
            "S14",
            "S15",
            "S16",
            "S17",
        ]

        all_features = []
        all_labels = []
        all_subjects = []

        loaded_subjects = []

        for subject_id in available_subjects:
            subject_path = self.data_dir / subject_id / f"{subject_id}.pkl"

            if not subject_path.exists():
                print(f"  Warning: {subject_id} data not found, skipping...")
                continue

            try:
                import pickle

                with open(subject_path, "rb") as f:
                    data = pickle.load(f, encoding="latin1")

                print(f"  Processing {subject_id}...")

                # Extract physiological signals from wrist sensor (more reliable)
                signals = data["signal"]["wrist"]
                labels = data["label"]

                # Sample rate for wrist device (usually 64 Hz for most signals)
                # We'll extract windows of data rather than single values

                # The wrist signals are at different sampling rates than labels (700Hz)
                # We need to downsample labels to match the wrist signals

                # Use EDA/TEMP length as reference (typically 4Hz)
                target_length = len(signals["EDA"])

                # Downsample labels to match wrist signal sampling rate
                # Labels are at 700Hz, wrist EDA/TEMP at ~4Hz
                downsample_factor = len(labels) // target_length
                labels_downsampled = labels[::downsample_factor][:target_length]

                # Use the aligned signals
                bvp = signals["BVP"]
                eda = signals["EDA"]
                acc = signals["ACC"]
                temp = signals["TEMP"]

                # Ensure all signals have the same length
                min_length = min(
                    len(bvp), len(eda), len(acc), len(temp), len(labels_downsampled)
                )
                bvp = bvp[:min_length]
                eda = eda[:min_length]
                acc = acc[:min_length]
                temp = temp[:min_length]
                labels_truncated = labels_downsampled[:min_length]

                # Create sliding windows but focus on labeled periods
                # EDA/TEMP are at ~4Hz, so 30 seconds = 120 samples
                window_size = 120  # 30 seconds at 4 Hz (EDA/TEMP sampling rate)
                step_size = 60  # 15 seconds step (50% overlap)

                subject_features = []
                subject_labels = []

                # Extract windows from entire signal, filter by label quality later
                signal_length = min_length
                print(f"    Processing signal of length {signal_length}")
                print(
                    f"    Original signal lengths - BVP: {len(signals['BVP'])}, EDA: {len(signals['EDA'])}, ACC: {len(signals['ACC'])}, TEMP: {len(signals['TEMP'])}, Labels: {len(labels)}"
                )

                # Extract windows from entire signal
                extracted_windows = 0
                skipped_transient = 0
                skipped_inconsistent = 0

                for start in range(0, signal_length - window_size, step_size):
                    end = start + window_size

                    # Extract window
                    bvp_window = bvp[start:end]
                    eda_window = eda[start:end]
                    acc_window = acc[start:end]
                    temp_window = temp[start:end]
                    label_window = labels_truncated[start:end]

                    # Get dominant label in window
                    unique_labels, counts = np.unique(label_window, return_counts=True)
                    dominant_label = unique_labels[np.argmax(counts)]
                    consistency = np.max(counts) / len(label_window)

                    # Only process if window has consistent labeling (at least 50% same label)
                    if consistency < 0.5:
                        skipped_inconsistent += 1
                        continue

                    # Skip transient periods (label 0) but keep others
                    if dominant_label == 0:
                        skipped_transient += 1
                        continue

                    # Extract features from window
                    window_features = []

                    # BVP features (Blood Volume Pulse)
                    window_features.extend(
                        [
                            np.mean(bvp_window),
                            np.std(bvp_window),
                            np.max(bvp_window),
                            np.min(bvp_window),
                            np.percentile(bvp_window, 25),
                            np.percentile(bvp_window, 75),
                        ]
                    )

                    # EDA features (Electrodermal Activity)
                    window_features.extend(
                        [
                            np.mean(eda_window),
                            np.std(eda_window),
                            np.max(eda_window),
                            np.min(eda_window),
                            np.sum(np.diff(eda_window) > 0.01),  # Number of EDA peaks
                        ]
                    )

                    # ACC features (3-axis accelerometer)
                    if len(acc_window.shape) > 1:
                        for axis in range(min(3, acc_window.shape[1])):
                            axis_data = acc_window[:, axis]
                            window_features.extend(
                                [
                                    np.mean(axis_data),
                                    np.std(axis_data),
                                    np.sqrt(np.mean(axis_data**2)),  # RMS
                                ]
                            )
                    else:
                        # Handle 1D case
                        window_features.extend(
                            [
                                np.mean(acc_window),
                                np.std(acc_window),
                                np.sqrt(np.mean(acc_window**2)),
                            ]
                        )

                    # TEMP features (Temperature)
                    window_features.extend([np.mean(temp_window), np.std(temp_window)])

                    # Binary label: stress (2) vs non-stress (1=baseline, 3=amusement, 4=meditation)
                    window_label = 1 if dominant_label == 2 else 0

                    subject_features.append(window_features)
                    subject_labels.append(window_label)
                    extracted_windows += 1

                print(
                    f"    Diagnosis: Extracted {extracted_windows}, Skipped transient {skipped_transient}, Skipped inconsistent {skipped_inconsistent}"
                )

                if len(subject_features) > 0:
                    all_features.extend(subject_features)
                    all_labels.extend(subject_labels)
                    all_subjects.extend([subject_id] * len(subject_features))
                    loaded_subjects.append(subject_id)

                    print(
                        f"    ✓ Extracted {len(subject_features)} windows, {len(subject_features[0]) if subject_features else 0} features each"
                    )
                    print(
                        f"    Class distribution: {dict(zip(*np.unique(subject_labels, return_counts=True)))}"
                    )
                else:
                    print("    ✗ No valid windows extracted")

            except Exception as e:
                print(f"  ✗ Error loading {subject_id}: {e}")
                continue

        if len(all_features) == 0:
            raise ValueError(
                "No WESAD data could be loaded. Please check data directory and files."
            )

        # Combine all features
        X = np.array(all_features)
        y = np.array(all_labels)

        print(f"\n✓ Successfully loaded {len(loaded_subjects)} subjects")
        print(f"  Total samples: {len(X)}")
        print(f"  Feature dimensions: {X.shape[1]}")
        print(
            f"  Overall class distribution: {dict(zip(*np.unique(y, return_counts=True)))}"
        )

        return X, y, all_subjects

    def split_by_subjects(
        self, X: np.ndarray, y: np.ndarray, subjects: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data by subjects to prevent data leakage."""
        unique_subjects = list(set(subjects))
        n_subjects = len(unique_subjects)

        print(f"\nSplitting {n_subjects} subjects:")

        # Calculate split sizes - handle small datasets
        if n_subjects < 5:
            print(
                f"  Warning: Only {n_subjects} subjects available. Using simple train/test split."
            )
            n_train = max(1, int(0.7 * n_subjects))  # 70% for training
            n_val = 0  # No validation set
        else:
            n_train = int(0.5 * n_subjects)  # 50% for training
            n_val = int(0.2 * n_subjects)  # 20% for validation

        # Shuffle subjects
        np.random.seed(42)
        shuffled_subjects = np.random.permutation(unique_subjects)

        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train : n_train + n_val]
        test_subjects = shuffled_subjects[n_train + n_val :]

        print(f"  Training: {len(train_subjects)} subjects - {list(train_subjects)}")
        print(f"  Validation: {len(val_subjects)} subjects - {list(val_subjects)}")
        print(f"  Test: {len(test_subjects)} subjects - {list(test_subjects)}")

        # Create splits
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        test_mask = np.isin(subjects, test_subjects)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_classical_models(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Train classical ML models and evaluate performance."""
        print("\nTraining classical models...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42, probability=True),
        }

        results = {}

        for name, model in models.items():
            print(f"\n  Training {name}...")

            # Train model
            model.fit(X_train_scaled, y_train)

            # Predict on validation and test sets
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # Calculate metrics
            val_metrics = self.calculate_metrics(y_val, y_val_pred, "Validation")
            test_metrics = self.calculate_metrics(y_test, y_test_pred, "Test")

            results[name] = {
                "validation": val_metrics,
                "test": test_metrics,
                "model": model,
            }

            print(f"    Validation Accuracy: {val_metrics['accuracy']:.3f}")
            print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return results

    def train_neural_network(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Train neural network model."""
        print("\nTraining Neural Network...")

        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test_scaled)

        # Define neural network
        class WESADNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 2)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x

        model = WESADNet(X_train_scaled.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0

        for epoch in range(100):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    _, val_pred = torch.max(val_outputs.data, 1)
                    val_acc = accuracy_score(y_val_tensor.numpy(), val_pred.numpy())

                    print(
                        f"    Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.3f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"    Early stopping at epoch {epoch}")
                        break

        # Load best model and evaluate on test
        model.load_state_dict(best_model_state)
        model.eval()

        with torch.no_grad():
            # Validation predictions
            val_outputs = model(X_val_tensor)
            _, y_val_pred = torch.max(val_outputs.data, 1)

            # Test predictions
            test_outputs = model(X_test_tensor)
            _, y_test_pred = torch.max(test_outputs.data, 1)

        # Calculate metrics
        val_metrics = self.calculate_metrics(y_val, y_val_pred.numpy(), "Validation")
        test_metrics = self.calculate_metrics(y_test, y_test_pred.numpy(), "Test")

        print(f"    Best Validation Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"    Test Accuracy: {test_metrics['accuracy']:.3f}")

        return {"validation": val_metrics, "test": test_metrics, "model": model}

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str
    ) -> dict:
        """Calculate comprehensive metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="binary", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def run_evaluation(self) -> dict:
        """Run complete baseline evaluation."""
        print("🧬 WESAD Dataset Baseline Performance Evaluation")
        print("=" * 60)

        # Load data
        X, y, subjects = self.load_wesad_data()

        # Split by subjects
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_by_subjects(
            X, y, subjects
        )

        print("\nFinal split sizes:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

        # Train classical models
        classical_results = self.train_classical_models(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Train neural network
        nn_results = self.train_neural_network(
            X_train, X_val, X_test, y_train, y_val, y_test
        )

        # Combine results
        all_results = {**classical_results, "Neural Network": nn_results}

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: dict) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 WESAD BASELINE PERFORMANCE SUMMARY")
        print("=" * 60)

        print("\nTest Set Results (Subject-based split):")
        print("-" * 40)

        for model_name, result in results.items():
            test_metrics = result["test"]
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.3f}")
            print(f"  Precision: {test_metrics['precision']:.3f}")
            print(f"  Recall:    {test_metrics['recall']:.3f}")
            print(f"  F1-Score:  {test_metrics['f1']:.3f}")

        # Best model
        best_model = max(results.items(), key=lambda x: x[1]["test"]["accuracy"])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   Test Accuracy: {best_model[1]['test']['accuracy']:.3f}")

        print("\n💡 Key Insights:")
        print("   - Subject-based splitting prevents data leakage")
        print("   - Results represent realistic federated learning scenarios")
        print("   - Can compare with 3-node federated learning performance")


def main():
    """Main evaluation function."""
    evaluator = WESADBaselineEvaluator()
    results = evaluator.run_evaluation()

    # Save results
    results_file = "wesad_baseline_results.json"
    import json

    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, result in results.items():
        if model_name == "Neural Network":
            continue  # Skip model object
        json_results[model_name] = {
            "validation": result["validation"],
            "test": result["test"],
        }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")
    print("   Ready for federated learning comparison!")


if __name__ == "__main__":
    main()
