from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import flwr as fl
import numpy as np

from flower_basic.servers.sweet import (
    MQTTFedAvgSweet,
    _evaluate_global,
    _load_eval_data,
)
from flower_basic.sweet_model import SweetMLP


def _write_split(path: Path, n_samples: int, n_features: int) -> None:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 3, size=n_samples).astype(np.int64)
    subjects = np.array(["u1"] * n_samples, dtype=object)
    np.savez(path, X=X, y=y, subjects=subjects)


def test_load_eval_data(tmp_path: Path) -> None:
    base = tmp_path / "run"
    (base / "fog_0").mkdir(parents=True)
    (base / "fog_1").mkdir(parents=True)
    _write_split(base / "fog_0" / "test.npz", n_samples=2, n_features=5)
    _write_split(base / "fog_1" / "test.npz", n_samples=3, n_features=5)

    manifest = {"nodes": {"fog_0": [], "fog_1": []}}
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    X, y = _load_eval_data(manifest_path)
    assert X.shape == (5, 5)
    assert y.shape == (5,)


def test_load_eval_data_no_test_files(tmp_path: Path) -> None:
    base = tmp_path / "run"
    (base / "fog_0").mkdir(parents=True)
    manifest = {"nodes": {"fog_0": []}}
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _load_eval_data(manifest_path)
    assert result is None


def test_evaluate_global() -> None:
    model = SweetMLP(input_dim=5, hidden_dims=[8, 4], num_classes=3)
    X = np.random.randn(6, 5).astype(np.float32)
    y = np.random.randint(0, 3, size=6).astype(np.int64)

    loss, acc = _evaluate_global(model, (X, y))

    assert loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_mqtt_fedavg_sweet_aggregate_fit_evaluates_last_round() -> None:
    model = SweetMLP(input_dim=5, hidden_dims=[8, 4], num_classes=3)
    eval_data = (
        np.random.randn(4, 5).astype(np.float32),
        np.random.randint(0, 3, size=4).astype(np.int64),
    )
    mqtt_client = Mock()

    strategy = MQTTFedAvgSweet(
        model=model,
        mqtt_client=mqtt_client,
        eval_data=eval_data,
        total_rounds=2,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    params = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    fl_params = fl.common.ndarrays_to_parameters(params)

    with patch.object(
        fl.server.strategy.FedAvg, "aggregate_fit", return_value=fl_params
    ):
        strategy.aggregate_fit(server_round=1, results=[(Mock(), Mock())], failures=[])
        assert strategy.history["round"] == []
        assert mqtt_client.publish.called

        mqtt_client.publish.reset_mock()
        strategy.aggregate_fit(server_round=2, results=[(Mock(), Mock())], failures=[])

    assert strategy.history["round"] == [2]
    assert mqtt_client.publish.called
