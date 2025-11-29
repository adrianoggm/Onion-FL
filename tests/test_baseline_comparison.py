"""Tests for baseline and comparison functionality."""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
from pathlib import Path

import pytest
import torch

from flower_basic.baseline_model import BaselineTrainer
from flower_basic.compare_models import FederatedSimulator, ModelComparator
from flower_basic.model import ECGModel


class TestBaselineTrainer:
    """Test baseline centralized trainer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.model = ECGModel()
        self.trainer = BaselineTrainer(
            model=self.model, device=self.device, learning_rate=0.001, batch_size=8
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.device == self.device
        assert self.trainer.batch_size == 8
        assert isinstance(self.trainer.criterion, torch.nn.BCEWithLogitsLoss)
        assert isinstance(self.trainer.optimizer, torch.optim.Adam)

    def test_create_data_loaders(self):
        """Test data loader creation."""
        # Create sample data
        X_train = np.random.randn(20, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 20).astype(np.int64)
        X_test = np.random.randn(10, 140).astype(np.float32)
        y_test = np.random.randint(0, 2, 10).astype(np.int64)

        train_loader, test_loader = self.trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        # Check loader properties
        assert len(train_loader.dataset) == 20  # type: ignore
        assert len(test_loader.dataset) == 10  # type: ignore
        assert train_loader.batch_size == 8

        # Check data shapes
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[1] == 1  # Channel dimension
        assert batch_x.shape[2] == 140  # Sequence length
        assert batch_y.shape[1] == 1  # Single output

    def test_train_epoch(self):
        """Test single epoch training."""
        # Create sample data loader
        X = torch.randn(16, 1, 140)
        y = torch.randint(0, 2, (16, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        # Train for one epoch
        loss, accuracy = self.trainer.train_epoch(loader)

        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert loss >= 0

    def test_evaluate(self):
        """Test model evaluation."""
        # Create sample data loader
        X = torch.randn(16, 1, 140)
        y = torch.randint(0, 2, (16, 1)).float()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)

        # Evaluate
        metrics = self.trainer.evaluate(loader)

        # Check all expected metrics are present
        expected_metrics = ["loss", "accuracy", "precision", "recall", "f1", "auc"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)

    def test_training_integration(self):
        """Test full training integration."""
        # Create sample data
        X_train = np.random.randn(32, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 32).astype(np.int64)
        X_test = np.random.randn(16, 140).astype(np.float32)
        y_test = np.random.randint(0, 2, 16).astype(np.int64)

        train_loader, test_loader = self.trainer.create_data_loaders(
            X_train, y_train, X_test, y_test
        )

        # Train for few epochs
        results = self.trainer.train(
            train_loader=train_loader, test_loader=test_loader, epochs=3, verbose=False
        )

        # Check results structure
        assert "final_metrics" in results
        assert "training_time" in results
        assert "epochs" in results
        assert results["epochs"] == 3
        assert len(self.trainer.train_losses) == 3
        assert len(self.trainer.val_accuracies) == 3

    def test_save_results(self):
        """Test saving training results."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock results
            results = {
                "final_metrics": {"accuracy": 0.85, "f1": 0.80},
                "training_time": 120.5,
                "epochs": 10,
                "train_losses": [0.7, 0.6, 0.5],
                "val_accuracies": [0.7, 0.75, 0.8],
            }

            # Save results
            self.trainer.save_results(results, temp_dir)

            # Check files were created
            assert os.path.exists(os.path.join(temp_dir, "baseline_metrics.json"))
            assert os.path.exists(os.path.join(temp_dir, "baseline_model.pth"))
            assert os.path.exists(os.path.join(temp_dir, "training_plots.png"))

            # Check JSON content
            with open(os.path.join(temp_dir, "baseline_metrics.json")) as f:
                saved_data = json.load(f)
                assert saved_data["epochs"] == 10
                assert saved_data["final_metrics"]["accuracy"] == 0.85


class TestFederatedSimulator:
    """Test federated learning simulator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.simulator = FederatedSimulator(num_clients=3, device=self.device)

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        assert self.simulator.num_clients == 3
        assert self.simulator.device == self.device
        assert isinstance(self.simulator.global_model, ECGModel)
        assert len(self.simulator.client_models) == 3

    def test_split_data_federated(self):
        """Test federated data splitting."""
        X = np.random.randn(100, 140)
        y = np.random.randint(0, 2, 100)

        client_data = self.simulator.split_data_federated(X, y)

        # Check correct number of clients
        assert len(client_data) == 3

        # Check data distribution
        total_samples = sum(len(X_client) for X_client, _ in client_data)
        assert total_samples == 100

        # Check each client has data
        for X_client, y_client in client_data:
            assert len(X_client) > 0
            assert len(y_client) > 0
            assert len(X_client) == len(y_client)

    def test_federated_averaging(self):
        """Test federated averaging mechanism."""
        # Initialize models with different parameters
        for i, client_model in enumerate(self.simulator.client_models):
            for param in client_model.parameters():
                param.data.fill_(i + 1)  # Fill with different values

        # Perform averaging
        self.simulator.federated_averaging()

        # Check that global model has averaged parameters
        global_params = list(self.simulator.global_model.parameters())
        client_params_list = [
            list(model.parameters()) for model in self.simulator.client_models
        ]

        # All client models should now have the same parameters as global
        for client_params in client_params_list:
            for global_param, client_param in zip(global_params, client_params):
                torch.testing.assert_close(global_param, client_param)

    @patch("flower_basic.compare_models.BaselineTrainer")
    def test_train_federated(self, mock_trainer_class):
        """Test federated training process."""
        # Mock trainer and its methods
        mock_trainer = MagicMock()
        mock_trainer.train_epoch.return_value = None
        mock_trainer.evaluate.return_value = {
            "accuracy": 0.85,
            "f1": 0.80,
            "precision": 0.82,
            "recall": 0.78,
            "auc": 0.88,
            "loss": 0.3,
        }
        # Mock create_data_loaders to return mock loaders
        mock_train_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_trainer.create_data_loaders.return_value = (
            mock_train_loader,
            mock_test_loader,
        )
        mock_trainer_class.return_value = mock_trainer

        # Create sample data
        X_train = np.random.randn(60, 140).astype(np.float32)
        y_train = np.random.randint(0, 2, 60).astype(np.int64)
        X_test = np.random.randn(20, 140).astype(np.float32)
        y_test = np.random.randint(0, 2, 20).astype(np.int64)

        # Run federated training
        results = self.simulator.train_federated(
            X_train, y_train, X_test, y_test, rounds=3, local_epochs=2
        )

        # Check results structure
        assert "final_metrics" in results
        assert "round_metrics" in results
        assert "training_time" in results
        assert "num_clients" in results
        assert results["num_clients"] == 3
        assert results["rounds"] == 3
        assert len(results["round_metrics"]) == 3


class TestModelComparator:
    """Test model comparison functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.comparator = ModelComparator(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_comparator_initialization(self):
        """Test comparator initialization."""
        assert self.comparator.output_dir == self.temp_dir
        assert os.path.exists(self.temp_dir)

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        # Mock results
        baseline_results = {
            "final_metrics": {
                "accuracy": 0.85,
                "f1": 0.80,
                "precision": 0.82,
                "recall": 0.78,
                "auc": 0.88,
            },
            "training_time": 120.0,
            "epochs": 50,
        }

        federated_results = {
            "final_metrics": {
                "accuracy": 0.83,
                "f1": 0.79,
                "precision": 0.81,
                "recall": 0.77,
                "auc": 0.86,
            },
            "training_time": 100.0,
            "rounds": 10,
            "num_clients": 3,
        }

        comparison = self.comparator.generate_comparison_report(
            baseline_results, federated_results
        )

        # Check structure
        assert "baseline" in comparison
        assert "federated" in comparison
        assert "differences" in comparison
        assert "summary" in comparison

        # Check calculations
        assert comparison["differences"]["accuracy_diff"] == pytest.approx(-0.02)
        assert comparison["differences"]["time_ratio"] == pytest.approx(100.0 / 120.0)
        assert comparison["summary"]["better_approach"] == "centralized"

    @patch("flower_basic.utils.load_ecg5000_subject_based")
    @patch("flower_basic.compare_models.BaselineTrainer")
    @patch("flower_basic.compare_models.FederatedSimulator")
    def test_run_comparison_integration(
        self, mock_fed_sim, mock_trainer_class, mock_load_data
    ):
        """Test full comparison integration."""
        # Mock data loading
        mock_load_data.return_value = (
            np.random.randn(80, 140),
            np.random.randn(20, 140),
            np.random.randint(0, 2, 80),
            np.random.randint(0, 2, 20),
        )

        # Mock baseline trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "final_metrics": {
                "accuracy": 0.85,
                "f1": 0.80,
                "precision": 0.82,
                "recall": 0.78,
                "auc": 0.88,
            },
            "training_time": 120.0,
            "epochs": 10,
        }
        # Mock create_data_loaders to return mock loaders
        mock_train_loader = MagicMock()
        mock_test_loader = MagicMock()
        mock_trainer.create_data_loaders.return_value = (
            mock_train_loader,
            mock_test_loader,
        )
        mock_trainer_class.return_value = mock_trainer

        # Mock federated simulator
        mock_simulator = MagicMock()
        mock_simulator.train_federated.return_value = {
            "final_metrics": {
                "accuracy": 0.83,
                "f1": 0.79,
                "precision": 0.81,
                "recall": 0.77,
                "auc": 0.86,
            },
            "training_time": 100.0,
            "rounds": 5,
            "num_clients": 3,
        }
        mock_fed_sim.return_value = mock_simulator

        # Run comparison
        if not Path("data/WESAD").exists():
            pytest.skip("WESAD dataset not available")
        comparison = self.comparator.run_comparison(epochs=10, num_clients=3, fl_rounds=5)

        # Check that comparison was generated
        assert "baseline" in comparison
        assert "federated" in comparison
        assert "differences" in comparison

    def test_save_comparison_results(self):
        """Test saving comparison results."""
        # Mock comparison data
        comparison = {
            "baseline": {
                "accuracy": 0.85,
                "f1": 0.80,
                "precision": 0.82,
                "recall": 0.78,
                "auc": 0.88,
                "training_time": 120.0,
            },
            "federated": {
                "accuracy": 0.83,
                "f1": 0.79,
                "precision": 0.81,
                "recall": 0.77,
                "auc": 0.86,
                "training_time": 100.0,
            },
            "differences": {
                "accuracy_diff": -0.02,
                "accuracy_diff_pct": -2.35,
                "f1_diff": -0.01,
                "f1_diff_pct": -1.25,
                "time_ratio": 0.83,
                "time_overhead_pct": -17.0,
            },
            "summary": {
                "better_approach": "centralized",
                "accuracy_gap": 0.02,
                "training_efficiency": "federated",
            },
        }

        baseline_results = {
            "train_accuracies": [0.7, 0.75, 0.8],
            "val_accuracies": [0.65, 0.7, 0.75],
        }

        federated_results = {
            "round_metrics": [
                {"round": 1, "accuracy": 0.7},
                {"round": 2, "accuracy": 0.75},
                {"round": 3, "accuracy": 0.8},
            ]
        }

        # Save results (should not crash)
        self.comparator.save_comparison_results(
            comparison, baseline_results, federated_results
        )

        # Check files were created
        assert os.path.exists(os.path.join(self.temp_dir, "comparison_report.json"))
        assert os.path.exists(os.path.join(self.temp_dir, "comparison_report.txt"))
        assert os.path.exists(os.path.join(self.temp_dir, "comparison_plots.png"))


class TestModelArchitectureConsistency:
    """Test that baseline and federated models use the same architecture."""

    def test_model_parameter_count_consistency(self):
        """Test that all models have the same number of parameters."""
        # Create different model instances
        baseline_model = ECGModel()
        federated_model = ECGModel()

        # Count parameters
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        federated_params = sum(p.numel() for p in federated_model.parameters())

        assert baseline_params == federated_params

    def test_model_architecture_consistency(self):
        """Test that model architectures are identical."""
        model1 = ECGModel()
        model2 = ECGModel()

        # Check layer names and types
        model1_layers = [
            (name, type(module)) for name, module in model1.named_modules()
        ]
        model2_layers = [
            (name, type(module)) for name, module in model2.named_modules()
        ]

        assert model1_layers == model2_layers

    def test_model_output_shape_consistency(self):
        """Test that models produce consistent output shapes."""
        model1 = ECGModel()
        model2 = ECGModel()

        # Create sample input
        x = torch.randn(4, 1, 140)

        # Get outputs
        output1 = model1(x)
        output2 = model2(x)

        assert output1.shape == output2.shape
        assert output1.shape == (4, 1)  # Batch size 4, single output
