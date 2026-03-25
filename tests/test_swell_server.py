from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import flwr as fl
import numpy as np

from flower_basic.servers.swell import (
    MQTTFedAvgSwell,
    _evaluate_global,
    _load_eval_data,
    _load_manifest_split_counts,
)
from flower_basic.swell_model import SwellMLP


def _write_split(path: Path, n_samples: int, n_features: int) -> None:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    subjects = np.array(["1"] * n_samples, dtype=object)
    np.savez(path, X=X, y=y, subjects=subjects)


def test_load_eval_data(tmp_path: Path) -> None:
    base = tmp_path / "run"
    (base / "fog_0").mkdir(parents=True)
    (base / "fog_1").mkdir(parents=True)
    _write_split(base / "fog_0" / "test.npz", n_samples=2, n_features=4)
    _write_split(base / "fog_1" / "test.npz", n_samples=3, n_features=4)

    manifest = {"nodes": {"fog_0": [], "fog_1": []}}
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    X, y = _load_eval_data(manifest_path)
    assert X.shape == (5, 4)
    assert y.shape == (5,)


def test_load_eval_data_no_test_files(tmp_path: Path) -> None:
    base = tmp_path / "run"
    (base / "fog_0").mkdir(parents=True)
    manifest = {"nodes": {"fog_0": []}}
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _load_eval_data(manifest_path)
    assert result is None


def test_load_manifest_split_counts(tmp_path: Path) -> None:
    base = tmp_path / "run"
    (base / "fog_0").mkdir(parents=True)
    (base / "fog_1").mkdir(parents=True)
    _write_split(base / "fog_0" / "train.npz", n_samples=2, n_features=4)
    _write_split(base / "fog_0" / "val.npz", n_samples=1, n_features=4)
    _write_split(base / "fog_0" / "test.npz", n_samples=3, n_features=4)
    _write_split(base / "fog_1" / "train.npz", n_samples=4, n_features=4)
    _write_split(base / "fog_1" / "val.npz", n_samples=2, n_features=4)
    _write_split(base / "fog_1" / "test.npz", n_samples=1, n_features=4)

    manifest = {"nodes": {"fog_0": [], "fog_1": []}}
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    train_n, val_n, test_n = _load_manifest_split_counts(manifest_path)
    assert train_n == 6
    assert val_n == 3
    assert test_n == 4


def test_evaluate_global() -> None:
    model = SwellMLP(input_dim=4)
    X = np.random.randn(6, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=6).astype(np.int64)

    loss, acc, cm, labels = _evaluate_global(model, (X, y))

    assert 0.0 <= acc <= 1.0
    assert loss >= 0.0
    assert cm.shape == (len(labels), len(labels))
    assert len(labels) > 0


def test_mqtt_fedavg_swell_aggregate_fit_evaluates_last_round() -> None:
    model = SwellMLP(input_dim=4)
    eval_data = (
        np.random.randn(4, 4).astype(np.float32),
        np.random.randint(0, 2, size=4).astype(np.int64),
    )
    mqtt_client = Mock()

    strategy = MQTTFedAvgSwell(
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


def test_mqtt_fedavg_swell_accepts_tuple_parameters() -> None:
    model = SwellMLP(input_dim=4)
    mqtt_client = Mock()

    strategy = MQTTFedAvgSwell(
        model=model,
        mqtt_client=mqtt_client,
        eval_data=None,
        total_rounds=1,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    params = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    fl_params = fl.common.ndarrays_to_parameters(params)
    payload = (fl_params, {"extra": True})

    with patch.object(fl.server.strategy.FedAvg, "aggregate_fit", return_value=payload):
        result = strategy.aggregate_fit(
            server_round=1, results=[(Mock(), Mock())], failures=[]
        )

    assert result is not None
    assert mqtt_client.publish.called


def test_mqtt_fedavg_swell_tracks_staleness_metadata() -> None:
    model = SwellMLP(input_dim=4)
    mqtt_client = Mock()

    strategy = MQTTFedAvgSwell(
        model=model,
        mqtt_client=mqtt_client,
        eval_data=None,
        total_rounds=1,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    params = [v.detach().cpu().numpy() for v in model.state_dict().values()]
    fl_params = fl.common.ndarrays_to_parameters(params)
    fit_res = Mock()
    fit_res.metrics = {
        "stale_update_count": 2,
        "future_update_count": 0,
        "round_min": 1,
        "round_max": 2,
        "max_delay_seconds": 1.25,
    }

    with patch.object(
        fl.server.strategy.FedAvg, "aggregate_fit", return_value=fl_params
    ):
        strategy.aggregate_fit(server_round=1, results=[(Mock(), fit_res)], failures=[])

    assert strategy.history["stale_updates"] == [2]
    assert strategy.history["future_updates"] == [0]
    assert strategy.history["round_span"] == [1]
    assert strategy.history["max_delay_seconds"] == [1.25]
