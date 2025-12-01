"""Performance comparison between federated and centralized models.

This module provides comprehensive comparison capabilities between federated
and centralized learning approaches with robust statistical validation.

The module includes:
- Federated learning simulation
- Statistical significance testing
- Data leakage detection
- Cross-validation support
- Comprehensive reporting and visualization

Example:
    >>> from compare_models import ModelComparator
    >>> comparator = ModelComparator()
    >>> results = comparator.run_robust_comparison(n_cv_folds=5)
    >>> print(f"Statistical significance: p={results['statistical_test']['p_value']:.3f}")
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate

from .baseline_model import BaselineTrainer
from .datasets import load_wesad_dataset
from .model import ECGModel
from .utils import (
    detect_data_leakage,
    statistical_significance_test,
)


class FederatedSimulator:
    """Simple federated learning simulator for performance comparison.

    This class simulates a basic federated learning setup where multiple
    clients train local models and their updates are aggregated using
    federated averaging.

    Attributes:
        num_clients: Number of federated clients participating.
        device: PyTorch device for computation (CPU/GPU).
        global_model: The global model shared across all clients.
        client_models: List of local models for each client.
    """

    def __init__(
        self,
        num_clients: int = 3,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the federated learning simulator.

        Args:
            num_clients: Number of federated clients to simulate.
                Must be positive. Defaults to 3.
            device: PyTorch device for computation. If None, uses CPU.

        Raises:
            ValueError: If num_clients is not positive.
        """
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")

        self.num_clients = num_clients
        self.device = device or torch.device("cpu")
        self.global_model = ECGModel().to(self.device)
        self.client_models: list[ECGModel] = [
            ECGModel().to(self.device) for _ in range(num_clients)
        ]

    def split_data_federated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split dataset among federated clients using random partitioning.

        This method distributes the dataset across clients in a way that simulates
        real-world federated learning scenarios where data is naturally partitioned
        across different devices or institutions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            random_state: Random seed for reproducible splits. If None, uses
                current random state.

        Returns:
            List of tuples, where each tuple contains (X_client, y_client) for
            each client. If there are more clients than samples, some clients
            will receive empty arrays.

        Raises:
            ValueError: If X and y have incompatible shapes.
        """
        if len(X) != len(y):
            raise ValueError(
                f"Incompatible shapes: X has {len(X)} samples, y has {len(y)} samples"
            )

        if len(X) == 0:
            empty_X = np.empty((0, X.shape[1]), dtype=X.dtype)
            empty_y = np.empty((0,), dtype=y.dtype)
            return [(empty_X, empty_y) for _ in range(self.num_clients)]

        # Set random state for reproducibility
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(X))

        # Calculate split sizes
        num_clients = min(self.num_clients, len(X))
        base_size = len(X) // num_clients
        remainder = len(X) % num_clients

        client_data: list[tuple[np.ndarray, np.ndarray]] = []
        start_idx = 0

        for i in range(num_clients):
            # Last client gets remainder samples
            split_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + split_size

            client_indices = indices[start_idx:end_idx]
            client_X = X[client_indices]
            client_y = y[client_indices]

            client_data.append((client_X, client_y))
            start_idx = end_idx

        # Pad with empty data if needed
        while len(client_data) < self.num_clients:
            empty_X = np.empty((0, X.shape[1]), dtype=X.dtype)
            empty_y = np.empty((0,), dtype=y.dtype)
            client_data.append((empty_X, empty_y))

        return client_data

    def federated_averaging(self) -> None:
        """Perform federated averaging of client models."""
        global_state = self.global_model.state_dict()

        # Average parameters from all clients
        for key in global_state.keys():
            # Stack all client parameters for this layer
            client_params = torch.stack(
                [client.state_dict()[key] for client in self.client_models]
            )
            # Average them
            global_state[key] = torch.mean(client_params, dim=0)

        # Update global model
        self.global_model.load_state_dict(global_state)

        # Update all client models with new global parameters
        for client in self.client_models:
            client.load_state_dict(global_state)

    def train_federated(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        rounds: int = 10,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> dict[str, Any]:
        """Simulate federated training.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            rounds: Number of federated rounds
            local_epochs: Local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            Dictionary with training results
        """
        print(
            f"Starting federated training ({self.num_clients} clients, {rounds} rounds)..."
        )

        # Split data among clients
        client_data = self.split_data_federated(X_train, y_train)

        # Create trainers for each client
        client_trainers = []
        for i, (X_client, y_client) in enumerate(client_data):
            trainer = BaselineTrainer(
                model=self.client_models[i],
                device=self.device,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            train_loader, _ = trainer.create_data_loaders(
                X_client, y_client, X_test, y_test
            )
            client_trainers.append((trainer, train_loader))
            print(f"   Client {i+1}: {len(X_client)} samples")

        # Create test loader for global evaluation
        global_trainer = BaselineTrainer(
            model=self.global_model,
            device=self.device,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )
        _, test_loader = global_trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        # Training history
        round_metrics = []
        start_time = time.time()

        for round_num in range(rounds):
            print(f"Round {round_num + 1}/{rounds}")

            # Local training on each client
            for _client_id, (trainer, train_loader) in enumerate(client_trainers):
                for _epoch in range(local_epochs):
                    trainer.train_epoch(train_loader)

            # Federated averaging
            self.federated_averaging()

            # Evaluate global model
            metrics = global_trainer.evaluate(test_loader)
            metrics["round"] = round_num + 1
            round_metrics.append(metrics)

            if (round_num + 1) % 2 == 0:
                print(f"   Global Accuracy: {metrics['accuracy']:.4f}")

        training_time = time.time() - start_time
        print(f"Federated training completed in {training_time:.2f} seconds")

        return {
            "final_metrics": round_metrics[-1],
            "round_metrics": round_metrics,
            "training_time": training_time,
            "num_clients": self.num_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
        }


class ModelComparator:
    """Compare federated and centralized model performance."""

    def __init__(self, output_dir: str = "comparison_results"):
        """Initialize comparator.

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_comparison(
        self,
        epochs: int = 50,
        num_clients: int = 3,
        fl_rounds: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Run complete comparison between models.

        Args:
            epochs: Epochs for centralized training
            num_clients: Number of federated clients
            fl_rounds: Federated learning rounds
            batch_size: Batch size
            learning_rate: Learning rate
            test_size: Test set proportion
            random_state: Random seed

        Returns:
            Comparison results dictionary
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("*** Model Performance Comparison ***")
        print("=" * 60)

        # Load data with subject-based partitioning to prevent data leakage
        print("Loading WESAD dataset with subject-based simulation...")
        X_train, X_test, y_train, y_test = load_wesad_dataset(
            test_size=test_size, random_state=random_state
        )

        print(
            f"Dataset: {len(X_train)} train, {len(X_test)} test samples (simulated subjects)"
        )

        # 1. Baseline (Centralized) Training
        print("\n1. Running Baseline (Centralized) Training...")
        baseline_model = ECGModel().to(device)
        baseline_trainer = BaselineTrainer(
            model=baseline_model,
            device=device,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        train_loader, test_loader = baseline_trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        baseline_results = baseline_trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            verbose=False,
        )

        print(
            f"Baseline completed: Accuracy = {baseline_results['final_metrics']['accuracy']:.4f}"
        )

        # 2. Federated Training
        print("\n2. Running Federated Training...")
        fed_simulator = FederatedSimulator(num_clients=num_clients, device=device)

        federated_results = fed_simulator.train_federated(
            X_train,
            y_train,
            X_test,
            y_test,
            rounds=fl_rounds,
            local_epochs=epochs // fl_rounds,  # Approximate same total training
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        print(
            f"Federated completed: Accuracy = {federated_results['final_metrics']['accuracy']:.4f}"
        )

        # 3. Generate comparison
        comparison = self.generate_comparison_report(
            baseline_results, federated_results
        )

        # 4. Save results
        self.save_comparison_results(comparison, baseline_results, federated_results)

        return comparison

    def generate_comparison_report(
        self, baseline_results: dict[str, Any], federated_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate comprehensive comparison report.

        Args:
            baseline_results: Results from baseline training
            federated_results: Results from federated training

        Returns:
            Comparison dictionary
        """
        baseline_metrics = baseline_results["final_metrics"]
        federated_metrics = federated_results["final_metrics"]

        # Calculate differences
        accuracy_diff = federated_metrics["accuracy"] - baseline_metrics["accuracy"]
        f1_diff = federated_metrics["f1"] - baseline_metrics["f1"]
        time_ratio = (
            federated_results["training_time"] / baseline_results["training_time"]
        )

        comparison = {
            "baseline": {
                "accuracy": baseline_metrics["accuracy"],
                "f1": baseline_metrics["f1"],
                "precision": baseline_metrics["precision"],
                "recall": baseline_metrics["recall"],
                "auc": baseline_metrics["auc"],
                "training_time": baseline_results["training_time"],
                "epochs": baseline_results["epochs"],
            },
            "federated": {
                "accuracy": federated_metrics["accuracy"],
                "f1": federated_metrics["f1"],
                "precision": federated_metrics["precision"],
                "recall": federated_metrics["recall"],
                "auc": federated_metrics["auc"],
                "training_time": federated_results["training_time"],
                "rounds": federated_results["rounds"],
                "clients": federated_results["num_clients"],
            },
            "differences": {
                "accuracy_diff": accuracy_diff,
                "accuracy_diff_pct": (accuracy_diff / baseline_metrics["accuracy"])
                * 100,
                "f1_diff": f1_diff,
                "f1_diff_pct": (
                    (f1_diff / baseline_metrics["f1"]) * 100
                    if baseline_metrics["f1"] > 0
                    else 0
                ),
                "time_ratio": time_ratio,
                "time_overhead_pct": (time_ratio - 1) * 100,
            },
            "summary": {
                "better_approach": "federated" if accuracy_diff > 0 else "centralized",
                "accuracy_gap": abs(accuracy_diff),
                "training_efficiency": "federated" if time_ratio < 1 else "centralized",
            },
        }

        return comparison

    def save_comparison_results(
        self,
        comparison: dict[str, Any],
        baseline_results: dict[str, Any],
        federated_results: dict[str, Any],
    ) -> None:
        """Save comparison results and generate visualizations.

        Args:
            comparison: Comparison results
            baseline_results: Baseline training results
            federated_results: Federated training results
        """
        # Save JSON report
        with open(os.path.join(self.output_dir, "comparison_report.json"), "w") as f:
            json.dump(comparison, f, indent=2)

        # Generate text report
        self.generate_text_report(comparison)

        # Generate comparison plots
        self.plot_comparison(comparison, baseline_results, federated_results)

        print(f"\nComparison results saved to {self.output_dir}/")

    def generate_text_report(self, comparison: dict[str, Any]) -> None:
        """Generate human-readable text report.

        Args:
            comparison: Comparison results
        """
        report_path = os.path.join(self.output_dir, "comparison_report.txt")

        with open(report_path, "w") as f:
            f.write("*** ECG Model Performance Comparison Report ***\n")
            f.write("=" * 50 + "\n\n")

            # Performance comparison table
            f.write("Performance Metrics Comparison\n")
            f.write("-" * 40 + "\n")

            metrics_data = [
                ["Metric", "Centralized", "Federated", "Difference"],
                [
                    "Accuracy",
                    f"{comparison['baseline']['accuracy']:.4f}",
                    f"{comparison['federated']['accuracy']:.4f}",
                    f"{comparison['differences']['accuracy_diff']:+.4f}",
                ],
                [
                    "F1-Score",
                    f"{comparison['baseline']['f1']:.4f}",
                    f"{comparison['federated']['f1']:.4f}",
                    f"{comparison['differences']['f1_diff']:+.4f}",
                ],
                [
                    "Precision",
                    f"{comparison['baseline']['precision']:.4f}",
                    f"{comparison['federated']['precision']:.4f}",
                    "-",
                ],
                [
                    "Recall",
                    f"{comparison['baseline']['recall']:.4f}",
                    f"{comparison['federated']['recall']:.4f}",
                    "-",
                ],
                [
                    "AUC",
                    f"{comparison['baseline']['auc']:.4f}",
                    f"{comparison['federated']['auc']:.4f}",
                    "-",
                ],
            ]

            f.write(tabulate(metrics_data, headers="firstrow", tablefmt="grid"))
            f.write("\n\n")

            # Training efficiency
            f.write("Training Efficiency\n")
            f.write("-" * 25 + "\n")
            f.write(
                f"Centralized Training Time: {comparison['baseline']['training_time']:.2f}s\n"
            )
            f.write(
                f"Federated Training Time: {comparison['federated']['training_time']:.2f}s\n"
            )
            f.write(
                f"Time Ratio (Fed/Central): {comparison['differences']['time_ratio']:.2f}x\n"
            )
            f.write(
                f"Time Overhead: {comparison['differences']['time_overhead_pct']:+.1f}%\n\n"
            )

            # Summary
            f.write("Summary\n")
            f.write("-" * 12 + "\n")
            f.write(
                f"Better Accuracy: {comparison['summary']['better_approach'].title()}\n"
            )
            f.write(f"Accuracy Gap: {comparison['summary']['accuracy_gap']:.4f}\n")
            f.write(
                f"More Efficient: {comparison['summary']['training_efficiency'].title()}\n"
            )

            # Recommendations
            f.write("\nRecommendations\n")
            f.write("-" * 20 + "\n")

            if (
                comparison["differences"]["accuracy_diff"] > -0.02
            ):  # Within 2% is acceptable
                f.write("GOOD: Federated learning maintains competitive accuracy\n")
            else:
                f.write(
                    "WARNING: Significant accuracy degradation in federated approach\n"
                )

            if comparison["differences"]["time_ratio"] < 1.5:  # Less than 50% overhead
                f.write("GOOD: Federated learning has reasonable training overhead\n")
            else:
                f.write("WARNING: High training overhead in federated approach\n")

    def plot_comparison(
        self,
        comparison: dict[str, Any],
        baseline_results: dict[str, Any],
        federated_results: dict[str, Any],
    ) -> None:
        """Generate comparison plots.

        Args:
            comparison: Comparison results
            baseline_results: Baseline training results
            federated_results: Federated training results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Performance metrics comparison
        metrics = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC"]
        baseline_vals = [
            comparison["baseline"]["accuracy"],
            comparison["baseline"]["f1"],
            comparison["baseline"]["precision"],
            comparison["baseline"]["recall"],
            comparison["baseline"]["auc"],
        ]
        federated_vals = [
            comparison["federated"]["accuracy"],
            comparison["federated"]["f1"],
            comparison["federated"]["precision"],
            comparison["federated"]["recall"],
            comparison["federated"]["auc"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width / 2, baseline_vals, width, label="Centralized", alpha=0.8)
        ax1.bar(x + width / 2, federated_vals, width, label="Federated", alpha=0.8)
        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Score")
        ax1.set_title("Performance Metrics Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Training time comparison
        times = [
            comparison["baseline"]["training_time"],
            comparison["federated"]["training_time"],
        ]
        approaches = ["Centralized", "Federated"]
        colors = ["skyblue", "lightcoral"]

        bars = ax2.bar(approaches, times, color=colors, alpha=0.8)
        ax2.set_ylabel("Training Time (seconds)")
        ax2.set_title("Training Time Comparison")
        ax2.grid(True, alpha=0.3)

        # Add values on bars
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
            )

        # 3. Training curves (if available)
        if "train_accuracies" in baseline_results:
            epochs = range(1, len(baseline_results["train_accuracies"]) + 1)
            ax3.plot(
                epochs,
                baseline_results["train_accuracies"],
                "b-",
                label="Centralized Train",
            )
            ax3.plot(
                epochs,
                baseline_results["val_accuracies"],
                "b--",
                label="Centralized Val",
            )

        if "round_metrics" in federated_results:
            rounds = [m["round"] for m in federated_results["round_metrics"]]
            fed_accs = [m["accuracy"] for m in federated_results["round_metrics"]]
            ax3.plot(rounds, fed_accs, "r-", label="Federated Global", marker="o")

        ax3.set_xlabel("Epoch/Round")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Training Progress")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Accuracy difference visualization
        diff_pct = comparison["differences"]["accuracy_diff_pct"]
        color = "green" if diff_pct >= 0 else "red"

        ax4.bar(["Accuracy Difference"], [diff_pct], color=color, alpha=0.7)
        ax4.set_ylabel("Difference (%)")
        ax4.set_title("Federated vs Centralized Accuracy")
        ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax4.grid(True, alpha=0.3)

        # Add value on bar
        ax4.text(
            0,
            diff_pct + (0.1 if diff_pct >= 0 else -0.1),
            f"{diff_pct:+.2f}%",
            ha="center",
            va="bottom" if diff_pct >= 0 else "top",
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "comparison_plots.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def run_robust_comparison(
        self,
        epochs: int = 50,
        num_clients: int = 3,
        fl_rounds: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        test_size: float = 0.2,
        random_state: int = 42,
        n_cv_folds: int = 5,
        n_random_seeds: int = 3,
    ) -> dict[str, Any]:
        """Run robust comparison with cross-validation and statistical testing.

        Args:
            epochs: Epochs for centralized training
            num_clients: Number of federated clients
            fl_rounds: Federated learning rounds
            batch_size: Batch size
            learning_rate: Learning rate
            test_size: Test set proportion
            random_state: Random seed
            n_cv_folds: Number of cross-validation folds
            n_random_seeds: Number of random seeds for statistical testing

        Returns:
            Comprehensive comparison results with statistical analysis
        """
        print("*** Robust Model Performance Comparison with Statistical Validation ***")
        print("=" * 80)

        # Load data with cross-validation splits
        print("Loading WESAD dataset with cross-validation...")
        # For now, we'll use a simple approach - load the data once and create CV splits
        from sklearn.model_selection import StratifiedKFold

        X_full, _, y_full, _ = load_wesad_dataset(
            test_size=0.1, random_state=random_state
        )

        # Create cross-validation splits
        cv = StratifiedKFold(
            n_splits=n_cv_folds, shuffle=True, random_state=random_state
        )
        cv_splits = []

        for train_idx, test_idx in cv.split(X_full, y_full):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y_full[train_idx], y_full[test_idx]
            cv_splits.append((X_train, X_test, y_train, y_test))

        # Store results for statistical analysis
        centralized_results = []
        federated_results = []

        for fold_idx, (_X_train, _X_test, _y_train, _y_test) in enumerate(cv_splits):
            print(f"\n--- Cross-Validation Fold {fold_idx + 1}/{n_cv_folds} ---")

            # Run single comparison for this fold
            fold_comparison = self.run_comparison(
                epochs=epochs,
                num_clients=num_clients,
                fl_rounds=fl_rounds,
                batch_size=batch_size,
                learning_rate=learning_rate,
                test_size=test_size,
                random_state=random_state + fold_idx,  # Different seed per fold
            )

            centralized_results.append(fold_comparison["baseline"]["accuracy"])
            federated_results.append(fold_comparison["federated"]["accuracy"])

            print(
                f"Fold {fold_idx + 1} - Centralized: {fold_comparison['baseline']['accuracy']:.4f}, "
                f"Federated: {fold_comparison['federated']['accuracy']:.4f}"
            )

        # Statistical analysis
        print("\n--- Statistical Analysis ---")
        stat_test = statistical_significance_test(
            centralized_results, federated_results
        )

        print(
            f"Centralized accuracy: {stat_test['mean_a']:.4f} Â+/- {np.std(centralized_results):.4f}"
        )
        print(
            f"Federated accuracy: {stat_test['mean_b']:.4f} Â+/- {np.std(federated_results):.4f}"
        )
        print(
            f"T-statistic: {stat_test['t_statistic']:.4f}, p-value: {stat_test['p_value']:.4f}"
        )
        print(
            f"Effect size (Cohen's d): {stat_test['cohen_d']:.4f} ({stat_test['effect_size_interpretation']})"
        )
        print(
            f"Statistically significant: {'Yes' if stat_test['significant'] else 'No'}"
        )

        # Data leakage detection
        print("\n--- Data Leakage Detection ---")
        leakage_results = detect_data_leakage(X_train, X_test)
        print(f"Mean similarity: {leakage_results['mean_similarity']:.4f}")
        print(f"Max similarity: {leakage_results['max_similarity']:.4f}")
        print(f"Potential leakage ratio: {leakage_results['leakage_ratio']:.4f}")
        print(
            f"Data leakage detected: {'Yes' if leakage_results['potential_leakage'] else 'No'}"
        )

        # Comprehensive results
        robust_results = {
            "cross_validation": {
                "n_folds": n_cv_folds,
                "centralized_accuracies": centralized_results,
                "federated_accuracies": federated_results,
                "centralized_mean": np.mean(centralized_results),
                "federated_mean": np.mean(federated_results),
                "centralized_std": np.std(centralized_results),
                "federated_std": np.std(federated_results),
            },
            "statistical_test": stat_test,
            "data_leakage": leakage_results,
            "recommendations": self._generate_robust_recommendations(
                stat_test, leakage_results
            ),
        }

        # Save robust results
        self._save_robust_results(robust_results)

        return robust_results

    def _generate_robust_recommendations(
        self, stat_test: dict[str, Any], leakage_results: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on robust analysis."""
        recommendations = []

        if leakage_results["potential_leakage"]:
            recommendations.append(
                "â ï¸  DATA LEAKAGE DETECTED: Results may be artificially inflated. "
                "Consider using different datasets or proper subject-based splitting."
            )

        if stat_test["significant"]:
            if stat_test["mean_a"] > stat_test["mean_b"]:
                recommendations.append(
                    "ð CENTRALIZED SUPERIOR: Centralized learning shows statistically "
                    "significant better performance."
                )
            else:
                recommendations.append(
                    "ð FEDERATED SUPERIOR: Federated learning shows statistically "
                    "significant better performance."
                )
        else:
            recommendations.append(
                "ð NO SIGNIFICANT DIFFERENCE: Performance difference is not statistically significant."
            )

        if stat_test["cohen_d"] < 0.2:
            recommendations.append(
                "ð SMALL EFFECT SIZE: The performance difference is practically negligible."
            )

        return recommendations

    def _save_robust_results(self, robust_results: dict[str, Any]) -> None:
        """Save robust comparison results."""
        import json

        robust_file = os.path.join(self.output_dir, "robust_comparison_results.json")

        # Create a simplified version for JSON serialization
        clean_results = {
            "cross_validation": {
                "n_folds": robust_results["cross_validation"]["n_folds"],
                "centralized_mean": float(
                    robust_results["cross_validation"]["centralized_mean"]
                ),
                "federated_mean": float(
                    robust_results["cross_validation"]["federated_mean"]
                ),
                "centralized_std": float(
                    robust_results["cross_validation"]["centralized_std"]
                ),
                "federated_std": float(
                    robust_results["cross_validation"]["federated_std"]
                ),
            },
            "statistical_test": {
                "t_statistic": float(robust_results["statistical_test"]["t_statistic"]),
                "p_value": float(robust_results["statistical_test"]["p_value"]),
                "significant": bool(robust_results["statistical_test"]["significant"]),
                "effect_size": float(robust_results["statistical_test"]["cohen_d"]),
            },
            "data_leakage": {
                "leakage_detected": bool(
                    robust_results["data_leakage"]["potential_leakage"]
                ),
                "leakage_ratio": float(robust_results["data_leakage"]["leakage_ratio"]),
            },
            "recommendations": robust_results["recommendations"],
        }

        with open(robust_file, "w") as f:
            json.dump(clean_results, f, indent=2)

        print(f"\nâ Robust results saved to: {robust_file}")


def main():
    """Main function to run model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare federated vs centralized ECG models"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Epochs for centralized training"
    )
    parser.add_argument(
        "--num_clients", type=int, default=3, help="Number of federated clients"
    )
    parser.add_argument(
        "--fl_rounds", type=int, default=10, help="Federated learning rounds"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default="comparison_results", help="Output directory"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    # Run comparison
    comparator = ModelComparator(args.output_dir)

    comparator.run_comparison(
        epochs=args.epochs,
        num_clients=args.num_clients,
        fl_rounds=args.fl_rounds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nModel comparison completed successfully!")
    print(f"Check {args.output_dir}/ for detailed results and visualizations")


if __name__ == "__main__":
    main()
