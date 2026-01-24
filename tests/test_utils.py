"""Tests for utility functions."""

import numpy as np
import pytest
import torch

from flower_basic.utils import (
    _interpret_effect_size,
    cross_validate_models,
    detect_data_leakage,
    load_ecg5000_cross_validation,
    load_ecg5000_subject_based,
    load_wesad_dataset,
    numpy_to_state_dict,
    state_dict_to_numpy,
    statistical_significance_test,
)


class TestStateDictConversion:
    """Test cases for state dict conversion functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_state_dict = {
            "conv1.weight": torch.randn(16, 1, 5),
            "conv1.bias": torch.randn(16),
            "fc1.weight": torch.randn(64, 32),
            "fc1.bias": torch.randn(64),
        }

    def test_state_dict_to_numpy_conversion(self):
        """Test conversion from PyTorch state dict to numpy."""
        numpy_dict = state_dict_to_numpy(self.sample_state_dict)

        # Check that all values are lists (JSON serializable)
        assert isinstance(numpy_dict, dict)
        for key, value in numpy_dict.items():
            assert isinstance(value, list)
            assert key in self.sample_state_dict

    def test_numpy_to_state_dict_conversion(self):
        """Test conversion from numpy back to PyTorch state dict."""
        # Convert to numpy first
        numpy_dict = state_dict_to_numpy(self.sample_state_dict)

        # Convert back to state dict
        recovered_state_dict = numpy_to_state_dict(numpy_dict)

        # Check that keys match
        assert set(recovered_state_dict.keys()) == set(self.sample_state_dict.keys())

        # Check that tensors are recovered correctly
        for key in self.sample_state_dict:
            original = self.sample_state_dict[key]
            recovered = recovered_state_dict[key]

            assert isinstance(recovered, torch.Tensor)
            assert recovered.shape == original.shape
            torch.testing.assert_close(recovered, original)

    def test_conversion_roundtrip(self):
        """Test that conversion is lossless."""
        # Original -> numpy -> state_dict
        numpy_dict = state_dict_to_numpy(self.sample_state_dict)
        recovered_dict = numpy_to_state_dict(numpy_dict)

        # Check exact equality
        for key in self.sample_state_dict:
            torch.testing.assert_close(self.sample_state_dict[key], recovered_dict[key])

    def test_numpy_array_input(self):
        """Test handling of numpy arrays in state dict."""
        numpy_state_dict = {
            "param1": np.random.randn(10, 5).astype(np.float32),
            "param2": np.random.randn(5).astype(np.float32),
        }

        # Should handle numpy arrays correctly
        serializable_dict = state_dict_to_numpy(numpy_state_dict)

        for _key, value in serializable_dict.items():
            assert isinstance(value, list)

    def test_mixed_input_types(self):
        """Test handling of mixed PyTorch tensors and numpy arrays."""
        mixed_dict = {
            "torch_param": torch.randn(5, 3),
            "numpy_param": np.random.randn(3, 2).astype(np.float32),
            "list_param": [[1, 2], [3, 4]],
        }

        result = state_dict_to_numpy(mixed_dict)

        # All should be converted to lists
        for _key, value in result.items():
            assert isinstance(value, list)

    def test_device_handling(self):
        """Test that device information is handled correctly."""
        if torch.cuda.is_available():
            cuda_state_dict = {
                "param1": torch.randn(5, 3).cuda(),
                "param2": torch.randn(3).cuda(),
            }

            # Convert to numpy (should move to CPU)
            numpy_dict = state_dict_to_numpy(cuda_state_dict)

            # Convert back with CPU device
            cpu_state_dict = numpy_to_state_dict(numpy_dict, device=torch.device("cpu"))

            for key in cuda_state_dict:
                assert cpu_state_dict[key].device.type == "cpu"
                torch.testing.assert_close(
                    cuda_state_dict[key].cpu(), cpu_state_dict[key]
                )

    def test_empty_state_dict(self):
        """Test handling of empty state dict."""
        empty_dict = {}

        numpy_dict = state_dict_to_numpy(empty_dict)
        assert numpy_dict == {}

        recovered_dict = numpy_to_state_dict(numpy_dict)
        assert recovered_dict == {}

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test non-dict input
        with pytest.raises((TypeError, AttributeError)):
            state_dict_to_numpy("not a dict")  # type: ignore

        # Test invalid numpy dict
        invalid_dict = {"param": "not_a_tensor_or_array"}

        # Should either handle gracefully or raise appropriate error
        try:
            result = state_dict_to_numpy(invalid_dict)
            # If it doesn't raise an error, check that it handles the case
            assert isinstance(result, dict)
        except (TypeError, AttributeError):
            # Expected for invalid input
            pass


def test_detect_data_leakage_flags_duplicates() -> None:
    np.random.seed(0)
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X_test = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    stats = detect_data_leakage(X_train, X_test, threshold=0.9)
    assert bool(stats["leakage_detected"]) is True
    assert stats["potential_duplicates"] >= 1


def test_detect_data_leakage_no_duplicates() -> None:
    np.random.seed(1)
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    X_test = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    stats = detect_data_leakage(X_train, X_test, threshold=0.9)
    assert bool(stats["leakage_detected"]) is False


def test_statistical_significance_and_effect_size() -> None:
    results_a = [0.8, 0.81, 0.79, 0.8]
    results_b = [0.8, 0.81, 0.79, 0.8]

    stats = statistical_significance_test(results_a, results_b, alpha=0.05)
    assert bool(stats["significant"]) is False
    assert stats["effect_size_interpretation"] == "negligible"


def test_interpret_effect_size_boundaries() -> None:
    assert _interpret_effect_size(0.1) == "negligible"
    assert _interpret_effect_size(0.3) == "small"
    assert _interpret_effect_size(0.6) == "medium"
    assert _interpret_effect_size(1.0) == "large"


def test_cross_validate_models() -> None:
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)

    class DummyModel:
        def __init__(self, y_test):
            self._y_test = y_test

        def predict(self, _X):
            return self._y_test

    def model_fn(_X_train, _y_train, X_test, y_test, **_kwargs):
        return DummyModel(y_test)

    results = cross_validate_models(model_fn, X, y, n_splits=2, random_state=1)
    assert results["n_splits"] == 2
    assert len(results["fold_results"]["accuracy"]) == 2
    assert results["summary"]["accuracy"]["mean"] == 1.0


def test_load_wesad_dataset_mocked(monkeypatch) -> None:
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.array([1] * 5 + [2] * 5, dtype=np.int64)

    monkeypatch.setattr(
        "flower_basic.utils.fetch_openml",
        lambda *args, **kwargs: {"data": X, "target": y},
    )

    X_train, X_test, y_train, y_test = load_wesad_dataset(
        test_size=0.2, random_state=0, stratified=True
    )
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}


def test_load_ecg5000_subject_based_mocked(monkeypatch) -> None:
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.array([1] * 5 + [2] * 5, dtype=np.int64)

    monkeypatch.setattr(
        "flower_basic.utils.fetch_openml",
        lambda *args, **kwargs: {"data": X, "target": y},
    )

    X_train, X_test, y_train, y_test = load_ecg5000_subject_based(
        test_size=0.2, random_state=0, num_subjects=2, stratified=True
    )
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    assert y_train.dtype == np.int64
    assert y_test.dtype == np.int64


def test_load_ecg5000_cross_validation_mocked(monkeypatch) -> None:
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    y = np.array([1] * 10 + [2] * 10, dtype=np.int64)

    monkeypatch.setattr(
        "flower_basic.utils.fetch_openml",
        lambda *args, **kwargs: {"data": X, "target": y},
    )

    splits = load_ecg5000_cross_validation(n_splits=2, random_state=0, num_subjects=2)
    assert len(splits) == 2
    for X_train, X_test, y_train, y_test in splits:
        assert X_train.shape[1] == 2
        assert X_test.shape[1] == 2
        assert y_train.dtype == np.int64
        assert y_test.dtype == np.int64
