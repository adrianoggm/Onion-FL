from __future__ import annotations

import json
from collections import defaultdict
from unittest.mock import Mock, patch

import numpy as np


class TestSweetBroker:
    def setup_method(self):
        self.mock_client = Mock()
        self.test_region = "fog_0"
        self.sample_weights = {
            "layer.weight": np.random.randn(4, 4).tolist(),
            "layer.bias": np.random.randn(4).tolist(),
        }

    def test_on_update_accumulates(self):
        from flower_basic.brokers import sweet_fog

        original_k = sweet_fog.K
        try:
            sweet_fog.K = 2
            with patch.object(sweet_fog, "buffers", defaultdict(list)), patch.object(
                sweet_fog, "clients_per_region", defaultdict(set)
            ):
                payload = {
                    "region": self.test_region,
                    "weights": self.sample_weights,
                    "client_id": "client_1",
                    "num_samples": 20,
                }
                msg = Mock()
                msg.payload.decode.return_value = json.dumps(payload)

                sweet_fog.on_update(self.mock_client, None, msg)

                assert len(sweet_fog.buffers[self.test_region]) == 1
                self.mock_client.publish.assert_not_called()
        finally:
            sweet_fog.K = original_k

    def test_on_update_publishes_partial(self):
        from flower_basic.brokers import sweet_fog

        original_k = sweet_fog.K
        try:
            sweet_fog.K = 1
            with patch.object(sweet_fog, "buffers", defaultdict(list)), patch.object(
                sweet_fog, "clients_per_region", defaultdict(set)
            ):
                payload = {
                    "region": self.test_region,
                    "weights": self.sample_weights,
                    "client_id": "client_1",
                    "num_samples": 20,
                }
                msg = Mock()
                msg.payload.decode.return_value = json.dumps(payload)

                sweet_fog.on_update(self.mock_client, None, msg)

                self.mock_client.publish.assert_called_once()
                _topic, published = self.mock_client.publish.call_args[0]
                parsed = json.loads(published)

                assert parsed["region"] == self.test_region
                assert parsed["total_samples"] == 20
                assert "partial_weights" in parsed
        finally:
            sweet_fog.K = original_k
