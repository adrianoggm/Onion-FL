"""Integration tests for the complete fog federated learning system."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSystemIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_message_flow(self):
        """Test the complete message flow from client to server."""
        # This tests the data format compatibility across all components

        # 1. Client update format (client.py -> broker_fog.py)
        client_update = {
            "client_id": "client_123",
            "region": "region_0",
            "weights": {
                "conv1.weight": np.random.randn(16, 1, 5).tolist(),
                "conv1.bias": np.random.randn(16).tolist(),
                "conv2.weight": np.random.randn(32, 16, 5).tolist(),
                "conv2.bias": np.random.randn(32).tolist(),
            },
            "num_samples": 100,
            "loss": 0.15,
        }

        # Test JSON serialization
        serialized = json.dumps(client_update)
        parsed = json.loads(serialized)
        assert parsed == client_update

        # 2. Broker aggregation (broker_fog.py -> fog_flower_client.py)
        broker_output = {
            "region": "region_0",
            "partial_weights": client_update["weights"],  # Simplified for test
            "timestamp": time.time(),
        }

        # Test JSON serialization
        serialized = json.dumps(broker_output)
        parsed = json.loads(serialized)
        assert "region" in parsed
        assert "partial_weights" in parsed

        # 3. Server global model (server.py -> client.py)
        server_output = {
            "round": 1,
            "global_weights": client_update["weights"],  # Simplified for test
        }

        # Test JSON serialization
        serialized = json.dumps(server_output)
        parsed = json.loads(serialized)
        assert "round" in parsed
        assert "global_weights" in parsed

        # Test that weights can be converted back to numpy
        for _param_name, param_values in parsed["global_weights"].items():
            np.array(param_values)  # Should not raise exception

    def test_parameter_shape_consistency(self):
        """Test that parameter shapes remain consistent across components."""
        from flower_basic.model import ECGModel

        model = ECGModel()
        original_state = model.state_dict()

        # Simulate the flow: model -> numpy -> JSON -> numpy -> model
        # 1. Extract parameters (as in client.py)
        client_weights = {
            k: v.cpu().numpy().tolist() for k, v in original_state.items()
        }

        # 2. JSON serialization (MQTT transport)
        serialized = json.dumps(client_weights)
        deserialized = json.loads(serialized)

        # 3. Convert back to tensors (as in client._on_message)
        recovered_state = {k: torch.tensor(v) for k, v in deserialized.items()}

        # 4. Load into model
        model.load_state_dict(recovered_state)

        # Check that shapes are preserved
        final_state = model.state_dict()
        for key in original_state:
            assert original_state[key].shape == final_state[key].shape

    def test_aggregation_mathematics(self):
        """Test that the aggregation mathematics work correctly."""
        from flower_basic.brokers.fog import weighted_average

        # Create realistic model parameters
        num_clients = 3
        param_shapes = {
            "conv1.weight": (16, 1, 5),
            "conv1.bias": (16,),
            "conv2.weight": (32, 16, 5),
            "conv2.bias": (32,),
        }

        # Generate client updates
        client_updates = []
        for _i in range(num_clients):
            update = {}
            for param_name, shape in param_shapes.items():
                # Generate random parameters
                param = np.random.randn(*shape).astype(np.float32)
                update[param_name] = param.tolist()
            client_updates.append(update)

        # Compute weighted average - now returns (weights, stats) tuple
        aggregated, stats = weighted_average(client_updates)

        # Verify shapes are preserved
        for param_name, shape in param_shapes.items():
            aggregated_param = np.array(aggregated[param_name])
            assert aggregated_param.shape == shape

        # Verify it's actually an average (for uniform weights)
        for param_name in param_shapes:
            manual_avg = np.mean(
                [np.array(update[param_name]) for update in client_updates], axis=0
            )
            aggregated_param = np.array(aggregated[param_name])
            np.testing.assert_array_almost_equal(
                aggregated_param, manual_avg, decimal=5
            )

        # Verify stats are returned
        assert "norm" in stats
        assert "mean" in stats
        assert "std" in stats

    def test_multiple_rounds_simulation(self):
        """Simulate multiple rounds of federated learning."""
        from flower_basic.brokers.fog import weighted_average
        from flower_basic.model import ECGModel, get_parameters, set_parameters

        # Initialize models for multiple clients
        num_clients = 3
        num_rounds = 2

        client_models = [ECGModel() for _ in range(num_clients)]
        server_model = ECGModel()

        for round_num in range(num_rounds):
            # 1. Clients get global model (except first round)
            if round_num > 0:
                global_params = get_parameters(server_model)
                for client_model in client_models:
                    set_parameters(client_model, global_params)

            # 2. Clients train locally (simulated by adding noise)
            client_updates = []
            for _i, client_model in enumerate(client_models):
                # Simulate training by adding small random noise
                params = get_parameters(client_model)
                noisy_params = [
                    p + np.random.normal(0, 0.01, p.shape).astype(np.float32)
                    for p in params
                ]
                set_parameters(client_model, noisy_params)

                # Create update message
                state_dict = client_model.state_dict()
                update = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
                client_updates.append(update)

            # 3. Broker aggregates updates - now returns (weights, stats) tuple
            aggregated_weights, _stats = weighted_average(client_updates)

            # 4. Server updates global model
            state_dict = {k: torch.tensor(v) for k, v in aggregated_weights.items()}
            server_model.load_state_dict(state_dict)

        # Test completed without errors
        assert True

    @pytest.mark.integration
    def test_mqtt_topic_consistency(self):
        """Test that MQTT topics are used consistently across components."""
        # Expected topics from the system

        # Verify topic constants in each file
        from flower_basic.brokers.fog import PARTIAL_TOPIC, UPDATE_TOPIC
        from flower_basic.client import TOPIC_GLOBAL_MODEL, TOPIC_UPDATES
        from flower_basic.fog_flower_client import PARTIAL_TOPIC as FOG_PARTIAL_TOPIC

        # Check consistency
        assert TOPIC_UPDATES == UPDATE_TOPIC == "fl/updates"
        assert PARTIAL_TOPIC == FOG_PARTIAL_TOPIC == "fl/partial"
        assert TOPIC_GLOBAL_MODEL == "fl/global_model"

    def test_configuration_consistency(self):
        """Test that configuration is consistent across components."""
        # MQTT broker configuration
        from flower_basic.brokers.fog import MQTT_BROKER as BROKER_BROKER
        from flower_basic.brokers.fog import MQTT_PORT as BROKER_PORT
        from flower_basic.client import MQTT_BROKER as CLIENT_BROKER
        from flower_basic.client import MQTT_PORT as CLIENT_PORT
        from flower_basic.fog_flower_client import MQTT_BROKER as FOG_BROKER
        from flower_basic.fog_flower_client import MQTT_PORT as FOG_PORT

        # All should use the same broker
        assert CLIENT_BROKER == BROKER_BROKER == FOG_BROKER == "localhost"
        assert CLIENT_PORT == BROKER_PORT == FOG_PORT == 1883

        # Aggregation configuration
        from flower_basic.brokers.fog import K as BROKER_K

        assert BROKER_K >= 2  # Default aggregation threshold (K can be configured)

    def test_error_propagation_and_recovery(self):
        """Test how errors propagate through the system and recovery mechanisms."""
        from flower_basic.brokers.fog import on_update

        # Test with malformed client update
        mock_client = Mock()
        mock_msg = Mock()

        # Test 1: Invalid JSON
        mock_msg.payload.decode.return_value = "invalid json"

        # Should not raise exception
        try:
            on_update(mock_client, None, mock_msg)
        except Exception as e:
            pytest.fail(f"System should handle invalid JSON gracefully: {e}")

        # Test 2: Missing required fields
        mock_msg.payload.decode.return_value = json.dumps({"incomplete": "data"})

        try:
            on_update(mock_client, None, mock_msg)
        except Exception as e:
            pytest.fail(f"System should handle missing fields gracefully: {e}")

        # Test 3: Invalid weight data
        invalid_update = {
            "region": "test_region",
            "client_id": "test_client",
            "weights": "not_a_dict",
            "num_samples": 100,
        }
        mock_msg.payload.decode.return_value = json.dumps(invalid_update)

        try:
            on_update(mock_client, None, mock_msg)
        except Exception as e:
            pytest.fail(f"System should handle invalid weights gracefully: {e}")


class TestPerformanceCharacteristics:
    """Test performance characteristics of the system."""

    def test_aggregation_performance(self):
        """Test performance of aggregation operations."""

        from flower_basic.brokers.fog import weighted_average

        # Create large parameter sets
        large_updates = []
        for _i in range(10):  # 10 clients
            update = {
                "large_param": np.random.randn(1000, 1000).tolist()  # Large parameter
            }
            large_updates.append(update)

        # Time the aggregation - now returns (weights, stats) tuple
        start_time = time.time()
        result, stats = weighted_average(large_updates)
        end_time = time.time()

        aggregation_time = end_time - start_time

        # Should complete in reasonable time (adjust threshold as needed)
        assert (
            aggregation_time < 10.0
        ), f"Aggregation took too long: {aggregation_time}s"

        # Result should be correct shape
        assert len(result["large_param"]) == 1000
        assert len(result["large_param"][0]) == 1000

        # Verify stats are returned
        assert "norm" in stats

    def test_memory_usage_patterns(self):
        """Test memory usage during normal operations."""
        from flower_basic.model import ECGModel, get_parameters, set_parameters

        model = ECGModel()

        # Test repeated parameter operations
        for _i in range(100):
            params = get_parameters(model)
            set_parameters(model, params)

        # Should complete without memory issues
        assert True

    def test_serialization_performance(self):
        """Test JSON serialization performance."""

        from flower_basic.model import ECGModel

        model = ECGModel()
        state_dict = model.state_dict()

        # Convert to serializable format
        large_update = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}

        # Test serialization performance
        start_time = time.time()
        for _i in range(10):
            serialized = json.dumps(large_update)
            json.loads(serialized)
        end_time = time.time()

        serialization_time = end_time - start_time

        # Should be reasonably fast
        assert (
            serialization_time < 5.0
        ), f"Serialization took too long: {serialization_time}s"
