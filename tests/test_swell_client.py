from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from flower_basic.clients.swell import SwellFLClientMQTT


def _write_split(path: Path, n_samples: int, n_features: int, subject: str = "1"):
    if n_samples > 0:
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, size=n_samples).astype(np.int64)
        subjects = np.array([subject] * n_samples, dtype=object)
    else:
        X = np.empty((0, n_features), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        subjects = np.empty((0,), dtype=object)
    np.savez(path, X=X, y=y, subjects=subjects)


def _mock_mqtt_init(self, tag, mqtt_broker, mqtt_port, subscriptions=None) -> None:
    self.tag = tag
    self._subscriptions = list(subscriptions or [])
    self.mqtt = Mock()


def test_init_raises_on_empty_train(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=0, n_features=4)

    with pytest.raises(RuntimeError, match="Train split is empty"):
        SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")


def test_val_and_test_counts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=6)
    _write_split(node_dir / "val.npz", n_samples=2, n_features=6)
    _write_split(node_dir / "test.npz", n_samples=1, n_features=6)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0", client_id="c1")
    assert client.num_samples == 3
    assert client.num_val_samples == 2
    assert client.num_test_samples == 1


def test_evaluate_val_empty_without_val(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=5)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    assert client.evaluate_val() == {}


def test_publish_update_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=5)

    client = SwellFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        client_id="client_1",
    )
    client.publish_update(avg_loss=0.25, val_acc=0.5, round_num=2)

    assert client.mqtt.publish.called
    publish_call = client.mqtt.publish.call_args
    topic, payload = publish_call[0]
    parsed = json.loads(payload)

    assert topic == client.topic_updates
    assert parsed["client_id"] == "client_1"
    assert parsed["region"] == "fog_0"
    assert parsed["round"] == 2
    assert parsed["num_samples"] == 4
    assert parsed["sent_at"] > 0
    assert "weights" in parsed
    assert "trace_context" in parsed


def test_train_one_round_returns_loss(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=6, n_features=4)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    loss = client.train_one_round()
    assert isinstance(loss, float)


def test_on_message_sets_pending_global(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=3)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    state_dict = client.model.state_dict()
    weights = {k: v.detach().cpu().numpy().tolist() for k, v in state_dict.items()}
    weights["extra"] = [1, 2, 3]

    payload = {"round": 1, "global_weights": weights, "trace_context": {}}
    msg = SimpleNamespace(
        payload=json.dumps(payload).encode("utf-8"), topic=client.topic_global
    )

    client.on_message(None, None, msg)
    assert client._pending_global_state is not None
    assert "extra" not in client._pending_global_state
    assert client._got_global is True


def test_on_message_ignores_invalid_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=3)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    msg = SimpleNamespace(payload=b"not-json", topic=client.topic_global)

    client.on_message(None, None, msg)
    assert client._pending_global_state is None
    assert client._got_global is False


def test_wait_for_global_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=3)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    assert client.wait_for_global(timeout_s=0.5) is False


def test_wait_for_global_immediate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=3)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    with client._lock:
        client._pending_global_state = {"dummy": torch.tensor([1.0])}
    assert client.wait_for_global(timeout_s=1.0) is True


def test_evaluate_val_returns_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=6, n_features=4)
    _write_split(node_dir / "val.npz", n_samples=4, n_features=4)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    metrics = client.evaluate_val()
    assert "val_loss" in metrics
    assert "val_acc" in metrics


def test_run_applies_pending_state(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=2)
    _write_split(node_dir / "val.npz", n_samples=2, n_features=2)

    client = SwellFLClientMQTT(node_dir=str(node_dir), region="fog_0")
    pending = {k: torch.zeros_like(v) for k, v in client.model.state_dict().items()}
    client._pending_global_state = pending
    client._got_global = True

    monkeypatch.setattr(client, "train_one_round", lambda: 0.5)
    monkeypatch.setattr(client, "evaluate_val", lambda: {"val_acc": 0.1})

    client.run(rounds=2, delay=0.0)

    assert client._pending_global_state is None
    assert client._got_global is False
