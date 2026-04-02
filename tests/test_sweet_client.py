from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from flower_basic.clients.sweet import SweetFLClientMQTT


def _write_split(path: Path, n_samples: int, n_features: int, subject: str = "user1"):
    if n_samples > 0:
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, size=n_samples).astype(np.int64)
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
    _write_split(node_dir / "train.npz", n_samples=0, n_features=5)

    with pytest.raises(RuntimeError, match="Train split is empty"):
        SweetFLClientMQTT(
            node_dir=str(node_dir),
            region="fog_0",
            input_dim=5,
            hidden_dims=[16, 8],
            num_classes=3,
        )


def test_init_counts_for_aggregated_node(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=6)
    _write_split(node_dir / "val.npz", n_samples=2, n_features=6)
    _write_split(node_dir / "test.npz", n_samples=1, n_features=6)

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=6,
        hidden_dims=[16, 8],
        num_classes=3,
    )

    assert client.num_samples == 4
    assert client.num_val_samples == 2
    assert client.num_test_samples == 1


def test_init_counts_for_subject_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    subject_dir = node_dir / "subject_user0001"
    subject_dir.mkdir(parents=True)
    _write_split(
        subject_dir / "train.npz", n_samples=5, n_features=4, subject="user0001"
    )
    _write_split(subject_dir / "val.npz", n_samples=3, n_features=4, subject="user0001")

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=4,
        hidden_dims=[16, 8],
        num_classes=3,
        subject_id="user0001",
    )

    assert client.num_samples == 5
    assert client.num_val_samples == 3
    assert client.client_id == "fog_0_client_user0001"


def test_evaluate_val_empty_without_val(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=3, n_features=4)

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=4,
        hidden_dims=[16, 8],
        num_classes=3,
    )

    assert client.evaluate_val() == {}


def test_publish_update_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=5)

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=5,
        hidden_dims=[16, 8],
        num_classes=3,
    )
    client.publish_update(avg_loss=0.2, val_acc=0.4, round_num=2)

    assert client.mqtt.publish.called
    topic, payload = client.mqtt.publish.call_args[0]
    parsed = json.loads(payload)

    assert topic == client.topic_updates
    assert parsed["region"] == "fog_0"
    assert parsed["round"] == 2
    assert parsed["num_samples"] == 4
    assert "weights" in parsed
    assert "sent_at" in parsed


def test_on_message_sets_pending_global(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=3)

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=3,
        hidden_dims=[16, 8],
        num_classes=3,
    )
    state_dict = client.model.state_dict()
    weights = {k: v.detach().cpu().numpy().tolist() for k, v in state_dict.items()}
    payload = {"round": 1, "global_weights": weights, "trace_context": {}}
    msg = SimpleNamespace(
        payload=json.dumps(payload).encode("utf-8"), topic=client.topic_global
    )

    client.on_message(None, None, msg)

    assert client._pending_global_state is not None
    assert client._got_global is True


def test_waits_for_global_from_first_round(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "flower_basic.clients.baseclient.BaseMQTTComponent.__init__", _mock_mqtt_init
    )

    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_split(node_dir / "train.npz", n_samples=4, n_features=3)

    client = SweetFLClientMQTT(
        node_dir=str(node_dir),
        region="fog_0",
        input_dim=3,
        hidden_dims=[16, 8],
        num_classes=3,
    )

    assert client.should_wait_for_global(1) is True
    assert client.global_wait_timeout_seconds(1) == 60.0
