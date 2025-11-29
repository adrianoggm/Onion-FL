"""Tests for dataset loaders.

This module tests the dataset loading functionality for WESAD and SWELL datasets,
ensuring proper data preprocessing, validation, and federated partitioning.

Test categories:
- Unit tests for individual loader functions (WESAD, SWELL)
- Integration tests for end-to-end data loading
- Performance tests for large dataset handling
- Error handling tests for various failure modes
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from flower_basic.datasets import load_swell_dataset, load_wesad_dataset
from flower_basic.datasets.swell import (
    SWELLDatasetError,
    get_swell_info,
    partition_swell_by_subjects,
)
from flower_basic.datasets.wesad import WESADDatasetError, partition_wesad_by_subjects


class TestWESADDataset:
    """Test suite for WESAD dataset loading functionality."""

    @pytest.fixture
    def mock_wesad_data_dir(self):
        """Create mock WESAD data directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "WESAD"

            # Create subject directories
            for subject_id in ["S2", "S3", "S4"]:
                subject_dir = data_dir / subject_id
                subject_dir.mkdir(parents=True)

                # Create mock pickle file with minimal structure
                mock_data = {
                    "signal": {
                        "wrist": {
                            "BVP": np.random.randn(1000),
                            "EDA": np.random.randn(1000),
                            "ACC": np.random.randn(1000, 3),
                            "TEMP": np.random.randn(1000),
                        }
                    },
                    "label": np.random.choice([1, 2], size=1000),  # baseline and stress
                }

                import pickle

                pickle_path = subject_dir / f"{subject_id}.pkl"
                with open(pickle_path, "wb") as f:
                    pickle.dump(mock_data, f)

            yield data_dir

    def test_load_wesad_dataset_validation(self, tmp_path: Path) -> None:
        """Test parameter validation for WESAD loader."""
        data_dir = tmp_path / "WESAD"
        data_dir.mkdir()
        # Test invalid sensor location
        with pytest.raises(ValueError, match="sensor_location must be"):
            load_wesad_dataset(sensor_location="invalid", data_dir=data_dir)

        # Test invalid signals
        with pytest.raises(ValueError, match="Invalid signals"):
            load_wesad_dataset(signals=["INVALID_SIGNAL"], data_dir=data_dir)

        # Test invalid conditions
        with pytest.raises(ValueError, match="Invalid conditions"):
            load_wesad_dataset(conditions=["invalid_condition"], data_dir=data_dir)

        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between"):
            load_wesad_dataset(test_size=0.0, data_dir=data_dir)

    def test_missing_data_directory_raises_error(self) -> None:
        """Test that missing data directory raises appropriate error."""
        with pytest.raises(FileNotFoundError, match="WESAD data directory not found"):
            load_wesad_dataset(data_dir="/nonexistent/path")

    def test_invalid_subjects_raises_error(self) -> None:
        """Test that invalid subject IDs raise appropriate errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid subjects"):
                load_wesad_dataset(
                    data_dir=temp_dir,
                    subjects=["S1", "S99"],  # S1 excluded, S99 doesn't exist
                )

    def test_wesad_dataset_error_handling(self, mock_wesad_data_dir: Path) -> None:
        """Test error handling for WESAD dataset loading failures."""
        # Test with empty data directory
        empty_dir = mock_wesad_data_dir.parent / "empty_wesad"
        empty_dir.mkdir()

        with pytest.raises(
            WESADDatasetError, match="No valid subject data could be loaded"
        ):
            load_wesad_dataset(data_dir=empty_dir, subjects=["S2"])

    @pytest.mark.slow
    def test_partition_wesad_by_subjects(self) -> None:
        """Test subject-based partitioning for WESAD dataset."""
        num_clients = 3

        # Test parameter validation
        with pytest.raises(ValueError, match="Cannot create .* clients"):
            partition_wesad_by_subjects(num_clients=20)  # More clients than subjects

        # Note: Full integration test would require actual WESAD data
        # This test validates the interface and error handling


class TestSWELLDataset:
    """Test suite for SWELL dataset loading functionality."""

    def test_load_swell_dataset_with_mock_data(self):
        """Test SWELL dataset loading with mocked data files."""
        with patch("pandas.read_csv") as mock_read_csv, patch(
            "pathlib.Path.exists", return_value=True
        ):

            # Mock computer interaction features
            computer_df = pd.DataFrame(
                {
                    "subject": [1, 1, 2, 2, 3, 3],
                    "condition": [
                        "No stress",
                        "Time pressure",
                        "No stress",
                        "Interruptions",
                        "Control",
                        "Combined",
                    ],
                    "mouse_clicks": [10, 20, 15, 25, 12, 30],
                    "keyboard_strokes": [100, 150, 120, 180, 110, 200],
                    "app_switches": [5, 8, 6, 10, 5, 12],
                }
            )

            # Mock physiology features
            physio_df = pd.DataFrame(
                {
                    "subject": [1, 1, 2, 2, 3, 3],
                    "condition": [
                        "No stress",
                        "Time pressure",
                        "No stress",
                        "Interruptions",
                        "Control",
                        "Combined",
                    ],
                    "hr_mean": [70.5, 85.2, 68.1, 82.7, 72.3, 88.9],
                    "hrv_rmssd": [45.2, 32.1, 50.3, 28.5, 48.7, 25.8],
                    "scl_mean": [2.1, 3.5, 1.9, 3.2, 2.3, 3.8],
                }
            )

            mock_read_csv.side_effect = [computer_df, physio_df]

            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=["computer", "physiology"], test_size=0.3, random_state=42
            )

            # Verify shapes
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert X_train.shape[1] > 0  # Should have features from both modalities
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]

            # Verify binary classification (stress vs no stress)
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            assert len(unique_labels) <= 2

    def test_load_swell_dataset_with_subject_info(self):
        """Test SWELL dataset loading with subject information return."""
        with patch("pandas.read_csv") as mock_read_csv, patch(
            "pathlib.Path.exists", return_value=True
        ):

            # Mock single modality data
            mock_df = pd.DataFrame(
                {
                    "subject": [1, 2, 3] * 4,
                    "condition": ["No stress"] * 6 + ["Time pressure"] * 6,
                    "feature1": np.random.randn(12),
                    "feature2": np.random.randn(12),
                    "feature3": np.random.randn(12),
                }
            )

            mock_read_csv.return_value = mock_df

            X_train, X_test, y_train, y_test, subject_info = load_swell_dataset(
                modalities=["computer"], return_subject_info=True
            )

            # Verify subject info structure
            assert isinstance(subject_info, dict)
            assert "n_subjects" in subject_info
            assert "n_samples" in subject_info
            assert "n_features" in subject_info
            assert "feature_names" in subject_info
            assert "modalities" in subject_info
            assert "class_distribution" in subject_info

            # Verify content
            assert subject_info["n_subjects"] == 3
            assert subject_info["n_samples"] == 12
            assert subject_info["modalities"] == ["computer"]

    def test_partition_swell_by_subjects(self):
        """Test SWELL dataset partitioning by subjects."""
        with patch("flower_basic.datasets.swell.load_swell_dataset") as mock_load:
            # Mock different partitions with different subjects
            def mock_load_side_effect(*args, **kwargs):
                subjects = kwargs.get("subjects", [1, 2, 3])
                n_samples = len(subjects) * 10  # 10 samples per subject
                n_features = 50

                X = np.random.randn(n_samples, n_features)
                y = np.random.randint(0, 2, n_samples)

                # Split into train/test
                split_idx = int(0.8 * n_samples)
                return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

            mock_load.side_effect = mock_load_side_effect

            # Test partitioning
            partitions = partition_swell_by_subjects(n_partitions=3, random_state=42)

            assert len(partitions) == 3

            for i, (X_train, X_test, y_train, y_test) in enumerate(partitions):
                assert X_train.shape[0] > 0
                assert X_test.shape[0] > 0
                assert len(y_train) == X_train.shape[0]
                assert len(y_test) == X_test.shape[0]

                # Verify binary classification
                all_labels = np.concatenate([y_train, y_test])
                assert len(np.unique(all_labels)) <= 2

    def test_swell_dataset_error_handling(self, tmp_path: Path):
        """Test SWELL dataset error handling for various failure modes."""

        if not Path("data/SWELL").exists():
            pytest.skip("SWELL dataset not available")

        # Test missing directory
        with pytest.raises(SWELLDatasetError, match="directory not found"):
            load_swell_dataset(data_dir="nonexistent/path")

        # Test invalid modalities
        data_dir = tmp_path / "SWELL"
        data_dir.mkdir()
        with pytest.raises(ValueError, match="Invalid modalities"):
            load_swell_dataset(modalities=["invalid_modality"], data_dir=data_dir)

        # Test invalid subjects
        with pytest.raises(ValueError, match="Subject IDs must be between 1 and 25"):
            load_swell_dataset(subjects=[0, 30], data_dir=data_dir)

        # Test invalid test_size
        with pytest.raises(ValueError, match="test_size must be between"):
            load_swell_dataset(test_size=1.5)

    def test_swell_modality_selection(self):
        """Test SWELL dataset loading with different modality combinations."""
        with patch("pandas.read_csv") as mock_read_csv, patch(
            "pathlib.Path.exists", return_value=True
        ):

            # Mock data for each modality
            base_data = {
                "subject": [1, 2, 3] * 2,
                "condition": ["No stress"] * 3 + ["Time pressure"] * 3,
            }

            computer_df = pd.DataFrame(
                {**base_data, "mouse_clicks": np.random.randn(6)}
            )
            facial_df = pd.DataFrame({**base_data, "emotion_joy": np.random.randn(6)})
            posture_df = pd.DataFrame({**base_data, "head_angle": np.random.randn(6)})
            physio_df = pd.DataFrame({**base_data, "hr_mean": np.random.randn(6)})

            # Test single modality
            mock_read_csv.return_value = computer_df
            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=["computer"], test_size=0.3
            )
            assert X_train.shape[1] == 1  # Only mouse_clicks feature

            # Test multiple modalities
            mock_read_csv.side_effect = [computer_df, physio_df]
            X_train, X_test, y_train, y_test = load_swell_dataset(
                modalities=["computer", "physiology"], test_size=0.3
            )
            assert X_train.shape[1] == 2  # Features from both modalities

    def test_get_swell_info(self):
        """Test SWELL dataset information retrieval."""
        with patch("flower_basic.datasets.swell.load_swell_dataset") as mock_load:
            mock_load.return_value = (
                np.random.randn(100, 50),  # X_train
                np.random.randn(25, 50),  # X_test
                np.random.randint(0, 2, 100),  # y_train
                np.random.randint(0, 2, 25),  # y_test
                {
                    "n_subjects": 25,
                    "n_samples": 125,
                    "n_features": 50,
                    "modalities": ["computer", "facial", "posture", "physiology"],
                },
            )

            info = get_swell_info()

            assert isinstance(info, dict)
            assert "description" in info
            assert "citation" in info
            assert "n_subjects" in info
            assert info["n_subjects"] == 25
            assert "SWELL" in info["description"]

    @pytest.mark.parametrize("n_partitions", [2, 5, 10])
    def test_swell_partition_sizes(self, n_partitions):
        """Test SWELL partitioning with different partition counts."""
        with patch("flower_basic.datasets.swell.load_swell_dataset") as mock_load:

            def mock_load_side_effect(*args, **kwargs):
                subjects = kwargs.get("subjects", list(range(1, 26)))
                n_samples = len(subjects) * 5
                return (
                    np.random.randn(n_samples, 20),
                    np.random.randn(n_samples // 4, 20),
                    np.random.randint(0, 2, n_samples),
                    np.random.randint(0, 2, n_samples // 4),
                )

            mock_load.side_effect = mock_load_side_effect

            partitions = partition_swell_by_subjects(n_partitions=n_partitions)
            assert len(partitions) == n_partitions

            # Verify all partitions have data
            for partition in partitions:
                X_train, X_test, y_train, y_test = partition
                assert X_train.shape[0] > 0
                assert X_test.shape[0] > 0
