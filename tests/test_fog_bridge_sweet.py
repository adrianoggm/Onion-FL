from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from flower_basic.clients.fog_bridge_sweet import FogClientSweet
from flower_basic.sweet_model import get_parameters


def _mock_mqtt_init(self, tag, mqtt_broker, mqtt_port, subscriptions=None) -> None:
    self.tag = tag
    self._subscriptions = list(subscriptions or [])
    self.mqtt = Mock()


def test_on_message_filters_region(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSweet(
        server_address="localhost:8080",
        input_dim=4,
        hidden_dims=[8, 4],
        num_classes=3,
        region="fog_0",
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

    client = FogClientSweet(
        server_address="localhost:8080",
        input_dim=4,
        hidden_dims=[8, 4],
        num_classes=3,
        region="fog_0",
    )
    msg = SimpleNamespace(
        payload=json.dumps(
            {
                "region": "fog_0",
                "partial_weights": {"a": [1, 2]},
                "trace_context": {},
            }
        ).encode("utf-8")
    )

    client.on_message(None, None, msg)
    assert client.partial_weights == {"a": [1, 2]}
    assert client.partial_trace_context == {}
    assert client.partial_metadata == {}


def test_fit_forwards_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    client = FogClientSweet(
        server_address="localhost:8080",
        input_dim=4,
        hidden_dims=[8, 4],
        num_classes=3,
        region="fog_0",
    )
    params = get_parameters(client.model)
    client.partial_weights = {
        name: np.zeros_like(param) for name, param in client.model.state_dict().items()
    }

    out, num_samples, metrics = client.fit(params, {})
    assert len(out) == len(params)
    assert num_samples == 1000
    assert metrics == {}
    assert client.partial_weights is None


def test_fit_times_out_without_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client = FogClientSweet(
        server_address="localhost:8080",
        input_dim=4,
        hidden_dims=[8, 4],
        num_classes=3,
        region="fog_0",
    )
    params = get_parameters(client.model)
    out, num_samples, metrics = client.fit(params, {})

    assert len(out) == len(params)
    assert num_samples == 1
    assert metrics == {}
