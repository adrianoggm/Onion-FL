"""Tests for MQTT components (client, broker_fog)."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestBrokerFog:
    """Test cases for the fog broker component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.test_region = "test_region"
        self.sample_weights = {
            "conv1.weight": np.random.randn(16, 1, 5).tolist(),
            "conv1.bias": np.random.randn(16).tolist(),
        }

    @patch("flower_basic.brokers.fog.weighted_average")
    @patch("flower_basic.brokers.fog.buffers")
    def test_on_update_accumulation(self, mock_buffers, mock_weighted_avg):
        """Test that updates are correctly accumulated in buffers."""
        from flower_basic.brokers.fog import on_update

        # Setup mock buffers to simulate a list
        test_buffer = []
        mock_buffers.__getitem__.return_value = test_buffer
        mock_buffers.__setitem__ = Mock()

        # Create test message
        test_payload = {
            "region": self.test_region,
            "weights": self.sample_weights,
            "client_id": "test_client_1",
            "num_samples": 100,
        }

        # Create mock message
        mock_msg = Mock()
        mock_msg.payload.decode.return_value = json.dumps(test_payload)

        # Test buffer not full (should not trigger aggregation)
        # Start with 1 item so after adding 1 more we have 2 < K=3
        test_buffer.append("existing_update")

        on_update(self.mock_client, None, mock_msg)

        # Should not call weighted_average since buffer is not full (2 < 3)
        mock_weighted_avg.assert_not_called()

        # Buffer should have the new weights added
        assert len(test_buffer) == 2  # 1 existing + 1 new

    @patch("flower_basic.brokers.fog.weighted_average")
    @patch("flower_basic.brokers.fog.buffers")
    def test_on_update_triggers_aggregation(self, mock_buffers, mock_weighted_avg):
        """Test that aggregation is triggered when buffer is full."""
        from flower_basic.brokers.fog import K, on_update

        # Create a realistic buffer with items that can be iterated
        buffer_items = [
            {"weights": {"param1": [1.0, 2.0]}, "num_samples": 50, "client_id": "c1"},
            {"weights": {"param1": [3.0, 4.0]}, "num_samples": 50, "client_id": "c2"},
        ]
        # Make sure we have K items
        while len(buffer_items) < K:
            buffer_items.append(
                {
                    "weights": {"param1": [5.0, 6.0]},
                    "num_samples": 50,
                    "client_id": f"c{len(buffer_items)+1}",
                }
            )

        # Use a real list that supports iteration
        buffer_list = buffer_items.copy()
        mock_buffers.__getitem__.return_value = buffer_list

        mock_weighted_avg.return_value = {"aggregated": "weights"}

        # Create test message
        test_payload = {
            "region": self.test_region,
            "weights": self.sample_weights,
            "client_id": "test_client_1",
            "num_samples": 100,
        }

        mock_msg = Mock()
        mock_msg.payload.decode.return_value = json.dumps(test_payload)

        # Call function
        on_update(self.mock_client, None, mock_msg)

        # Should trigger aggregation and publish
        mock_weighted_avg.assert_called_once()
        self.mock_client.publish.assert_called_once()

    def test_weighted_average_computation(self):
        """Test the weighted average computation."""
        from flower_basic.brokers.fog import weighted_average

        # Create test updates
        updates = [
            {"param1": [1.0, 2.0], "param2": [3.0]},
            {"param1": [2.0, 3.0], "param2": [4.0]},
            {"param1": [3.0, 4.0], "param2": [5.0]},
        ]

        # Test uniform weighting
        result = weighted_average(updates)

        # Expected: [2.0, 3.0] for param1, [4.0] for param2
        expected_param1 = [2.0, 3.0]
        expected_param2 = [4.0]

        np.testing.assert_array_almost_equal(result["param1"], expected_param1)
        np.testing.assert_array_almost_equal(result["param2"], expected_param2)

    def test_weighted_average_with_custom_weights(self):
        """Test weighted average with custom weights."""
        from flower_basic.brokers.fog import weighted_average

        updates = [{"param1": [1.0, 2.0]}, {"param1": [3.0, 4.0]}]
        weights = [0.7, 0.3]

        result = weighted_average(updates, weights)

        # Expected: 0.7 * [1.0, 2.0] + 0.3 * [3.0, 4.0] = [1.6, 2.6]
        expected = [1.6, 2.6]
        np.testing.assert_array_almost_equal(result["param1"], expected)

    def test_malformed_message_handling(self):
        """Test handling of malformed MQTT messages."""
        from flower_basic.brokers.fog import on_update

        # Test with invalid JSON
        mock_msg = Mock()
        mock_msg.payload.decode.return_value = "invalid json"

        # Should not raise exception
        try:
            on_update(self.mock_client, None, mock_msg)
        except Exception as e:
            pytest.fail(f"Should handle malformed JSON gracefully, but raised: {e}")

    def test_missing_fields_handling(self):
        """Test handling of messages with missing required fields."""
        from flower_basic.brokers.fog import on_update

        # Test with missing weights
        test_payload = {
            "region": self.test_region,
            "client_id": "test_client_1",
            # Missing 'weights' field
        }

        mock_msg = Mock()
        mock_msg.payload.decode.return_value = json.dumps(test_payload)

        # Should handle gracefully
        try:
            on_update(self.mock_client, None, mock_msg)
        except Exception as e:
            pytest.fail(f"Should handle missing fields gracefully, but raised: {e}")


class TestClientMQTT:
    """Test cases for the MQTT client component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_mqtt_client = Mock()
        # Create complete sample weights matching ECGModel structure
        # Note: ECGModel is for binary classification (output size = 1)
        self.sample_global_weights = {
            "conv1.weight": np.random.randn(16, 1, 5).tolist(),
            "conv1.bias": np.random.randn(16).tolist(),
            "conv2.weight": np.random.randn(32, 16, 5).tolist(),
            "conv2.bias": np.random.randn(32).tolist(),
            "fc1.weight": np.random.randn(
                64, 32 * 32
            ).tolist(),  # 32 channels * 32 length after conv
            "fc1.bias": np.random.randn(64).tolist(),
            "fc2.weight": np.random.randn(
                1, 64
            ).tolist(),  # 1 output for binary classification
            "fc2.bias": np.random.randn(1).tolist(),
        }

    @patch("flower_basic.client.mqtt.Client")
    @patch("flower_basic.client.load_wesad_dataset")
    def test_client_initialization(self, mock_load_data, mock_mqtt_class):
        """Test client initialization."""
        # Mock data loading
        mock_load_data.return_value = (
            np.random.randn(100, 140),  # X_train
            np.random.randn(20, 140),  # X_test
            np.random.randint(0, 2, 100),  # y_train
            np.random.randint(0, 2, 20),  # y_test
        )

        mock_mqtt_class.return_value = self.mock_mqtt_client

        from flower_basic.client import FLClientMQTT

        client = FLClientMQTT()

        # Check that MQTT client was configured
        mock_mqtt_class.assert_called_once()
        self.mock_mqtt_client.connect.assert_called_once()
        self.mock_mqtt_client.loop_start.assert_called_once()

        # Check model initialization
        assert hasattr(client, "model")
        assert hasattr(client, "train_loader")

    @patch("flower_basic.client.mqtt.Client")
    @patch("flower_basic.client.load_wesad_dataset")
    def test_global_model_reception(self, mock_load_data, mock_mqtt_class):
        """Test reception and processing of global model updates."""
        # Setup mocks
        mock_load_data.return_value = (
            np.random.randn(100, 140),
            np.random.randn(20, 140),
            np.random.randint(0, 2, 100),
            np.random.randint(0, 2, 20),
        )
        mock_mqtt_class.return_value = self.mock_mqtt_client

        from flower_basic.client import FLClientMQTT

        client = FLClientMQTT()

        # Create mock message with global model
        test_payload = {"round": 1, "global_weights": self.sample_global_weights}

        mock_msg = Mock()
        mock_msg.topic = "fl/global_model"
        mock_msg.payload.decode.return_value = json.dumps(test_payload)

        # Test message processing
        client._on_message(None, None, mock_msg)

        # Should set the flag
        assert client._got_global

    @patch("flower_basic.client.mqtt.Client")
    @patch("flower_basic.client.load_wesad_dataset")
    def test_training_and_publishing(self, mock_load_data, mock_mqtt_class):
        """Test local training and weight publishing."""
        # Setup mocks
        mock_load_data.return_value = (
            np.random.randn(100, 140),
            np.random.randn(20, 140),
            np.random.randint(0, 2, 100),
            np.random.randint(0, 2, 20),
        )
        mock_mqtt_class.return_value = self.mock_mqtt_client

        from flower_basic.client import FLClientMQTT

        client = FLClientMQTT()

        # Mock the wait for global model
        client._got_global = False

        # Mock publish method
        with patch.object(client, "_got_global", True):
            # This should complete without waiting
            client.train_one_round()

        # Check that publish was called
        self.mock_mqtt_client.publish.assert_called()

        # Verify the published message structure
        publish_calls = self.mock_mqtt_client.publish.call_args_list
        assert len(publish_calls) > 0

        topic, payload = publish_calls[-1][0]
        assert topic == "fl/updates"

        # Parse the payload
        parsed_payload = json.loads(payload)
        assert "client_id" in parsed_payload
        assert "weights" in parsed_payload
        assert "region" in parsed_payload

    def test_invalid_global_model_message(self):
        """Test handling of invalid global model messages."""
        from flower_basic.client import FLClientMQTT

        with patch("flower_basic.client.mqtt.Client"), patch(
            "flower_basic.client.load_wesad_dataset"
        ) as mock_load:
            mock_load.return_value = (
                np.random.randn(100, 140),
                np.random.randn(20, 140),
                np.random.randint(0, 2, 100),
                np.random.randint(0, 2, 20),
            )

            client = FLClientMQTT()

            # Test with invalid JSON
            mock_msg = Mock()
            mock_msg.topic = "fl/global_model"
            mock_msg.payload.decode.return_value = "invalid json"

            # Should not raise exception
            try:
                client._on_message(None, None, mock_msg)
            except Exception as e:
                pytest.fail(f"Should handle invalid JSON gracefully, but raised: {e}")

            # Should not set the flag
            assert not client._got_global


class TestMQTTIntegration:
    """Integration tests for MQTT components."""

    @pytest.mark.integration
    def test_broker_client_communication_flow(self):
        """Test the communication flow between broker and client."""
        # This would be an integration test that requires actual MQTT setup
        # For now, we'll test the message format compatibility

        # Test that broker output matches client input expectations
        broker_output = {
            "region": "test_region",
            "partial_weights": {
                "conv1.weight": [[1.0, 2.0], [3.0, 4.0]],
                "conv1.bias": [0.5, 0.6],
            },
            "timestamp": time.time(),
        }

        # This should be parseable as client input format
        serialized = json.dumps(broker_output)
        parsed = json.loads(serialized)

        assert parsed["region"] == "test_region"
        assert "partial_weights" in parsed
        assert isinstance(parsed["partial_weights"], dict)

    def test_message_format_consistency(self):
        """Test that message formats are consistent across components."""
        # Client update format
        client_update = {
            "client_id": "client_123",
            "region": "region_0",
            "weights": {"param1": [1.0, 2.0], "param2": [3.0]},
            "num_samples": 100,
            "loss": 0.15,
        }

        # Should be JSON serializable
        serialized = json.dumps(client_update)
        parsed = json.loads(serialized)

        # Check required fields
        required_fields = ["client_id", "region", "weights"]
        for field in required_fields:
            assert field in parsed

        # Global model format
        global_model = {
            "round": 1,
            "global_weights": {"param1": [1.5, 2.5], "param2": [3.5]},
        }

        # Should be JSON serializable
        serialized = json.dumps(global_model)
        parsed = json.loads(serialized)

        assert "round" in parsed
        assert "global_weights" in parsed
