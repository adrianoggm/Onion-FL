from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np

from flower_basic.fog_flower_client import FogClient
from flower_basic.model import get_parameters


class _DummyMQTT:
    def __init__(self, *args, **kwargs):
        self.on_connect = None
        self.on_message = None
        self.subscriptions: list[str] = []

    def connect(self, *_args, **_kwargs):
        return 0

    def loop_start(self):
        return None

    def subscribe(self, topic):
        self.subscriptions.append(topic)


def _patch_mqtt(monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.fog_flower_client.mqtt.Client",
        lambda *_args, **_kwargs: _DummyMQTT(),
    )


def test_on_connect_subscribes(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    client = FogClient(server_address="localhost:8080")

    dummy = _DummyMQTT()
    client._on_connect(dummy, None, None, 0)
    assert client.partial_topic in dummy.subscriptions


def test_on_connect_error(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    client = FogClient(server_address="localhost:8080")

    dummy = _DummyMQTT()
    client._on_connect(dummy, None, None, 1)
    assert dummy.subscriptions == []


def test_on_partial_sets_weights(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    client = FogClient(server_address="localhost:8080")

    payload = {"region": "fog_0", "partial_weights": {"w": [1, 2, 3]}}
    msg = SimpleNamespace(payload=json.dumps(payload).encode("utf-8"))
    client._on_partial(None, None, msg)

    assert client.partial_weights == {"w": [1, 2, 3]}


def test_on_partial_invalid_json(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    client = FogClient(server_address="localhost:8080")

    msg = SimpleNamespace(payload=b"not-json")
    client._on_partial(None, None, msg)
    assert client.partial_weights is None


def test_fit_timeout(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client = FogClient(server_address="localhost:8080")
    params = get_parameters(client.model)
    out, num_samples, metrics = client.fit(params, {})

    assert len(out) == len(params)
    assert num_samples == 1
    assert metrics == {}


def test_fit_with_partial(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client = FogClient(server_address="localhost:8080")
    params = get_parameters(client.model)
    client.partial_weights = {
        name: np.zeros_like(param) for name, param in client.model.state_dict().items()
    }

    out, num_samples, metrics = client.fit(params, {})
    assert len(out) == len(params)
    assert num_samples == 1000
    assert metrics == {}
    assert client.partial_weights is None


def test_evaluate_returns_zero(monkeypatch) -> None:
    _patch_mqtt(monkeypatch)
    client = FogClient(server_address="localhost:8080")
    loss, num, metrics = client.evaluate([], {})
    assert loss == 0.0
    assert num == 0
    assert metrics == {}
