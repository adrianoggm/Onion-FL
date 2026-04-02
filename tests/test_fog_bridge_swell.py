from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from flower_basic.clients.fog_bridge_swell import FogClientSwell
from flower_basic.swell_model import get_parameters


def _mock_mqtt_init(self, tag, mqtt_broker, mqtt_port, subscriptions=None) -> None:
    self.tag = tag
    self._subscriptions = list(subscriptions or [])
    self.mqtt = Mock()


def test_on_message_filters_region(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSwell(
        server_address="localhost:8080", input_dim=4, region="fog_0"
    )
    msg = SimpleNamespace(
        payload=json.dumps(
            {"region": "fog_1", "partial_weights": {"x": [1, 2]}}
        ).encode("utf-8")
    )

    client.on_message(None, None, msg)
    assert client.partial_weights is None


def test_on_message_sets_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSwell(
        server_address="localhost:8080", input_dim=4, region="fog_0"
    )
    msg = SimpleNamespace(
        payload=json.dumps(
            {
                "region": "fog_0",
                "partial_weights": {"a": [1, 2]},
                "expected_round": 3,
                "round_min": 2,
                "round_max": 3,
                "stale_update_count": 1,
                "future_update_count": 0,
                "max_delay_seconds": 4.5,
                "mean_delay_seconds": 2.0,
                "stale_policy": "accept",
                "trace_context": {},
            }
        ).encode("utf-8")
    )

    client.on_message(None, None, msg)
    assert client.partial_weights == {"a": [1, 2]}
    assert client.partial_trace_context == {}
    assert client.partial_metadata["expected_round"] == 3
    assert client.partial_metadata["stale_update_count"] == 1


def test_fit_forwards_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSwell(
        server_address="localhost:8080", input_dim=4, region="fog_0"
    )
    params = get_parameters(client.model)
    client.partial_weights = {
        name: np.zeros_like(param) for name, param in client.model.state_dict().items()
    }
    client.partial_metadata = {
        "expected_round": 2,
        "round_min": 1,
        "round_max": 2,
        "stale_update_count": 1,
        "future_update_count": 0,
        "max_delay_seconds": 3.0,
        "mean_delay_seconds": 2.0,
        "stale_policy": "accept",
    }

    out, num_samples, metrics = client.fit(params, {})
    assert len(out) == len(params)
    assert num_samples == 1000
    assert client.partial_weights is None
    assert client.partial_metadata == {}
    assert metrics["expected_round"] == 2
    assert metrics["stale_update_count"] == 1


def test_fit_times_out_without_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client = FogClientSwell(
        server_address="localhost:8080", input_dim=4, region="fog_0"
    )
    params = get_parameters(client.model)
    out, num_samples, _metrics = client.fit(params, {})

    assert len(out) == len(params)
    assert num_samples == 1


def test_fit_handles_forwarding_error_without_exiting(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSwell(
        server_address="localhost:8080", input_dim=4, region="fog_0"
    )
    params = get_parameters(client.model)
    client.partial_weights = {"missing_param": [0.0]}
    client.partial_metadata = {"expected_round": 1}

    out, num_samples, metrics = client.fit(params, {})

    assert len(out) == len(params)
    assert num_samples == 1
    assert metrics["bridge_error"] is True
    assert metrics["bridge_error_type"] == "KeyError"
    assert client.partial_weights is None
    assert client.partial_metadata == {}
