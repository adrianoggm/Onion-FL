from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flower_basic.federated_architecture import (
    ClientSpec,
    DatasetConfig,
    FederatedArchitecture,
    FogNodeSpec,
    ModelConfig,
    MQTTConfig,
    MQTTTopics,
    OrchestratorSpec,
    build_runtime_plan,
    materialize_swell_partitions,
)


def _write_train_npz(path: Path, n_samples: int, n_features: int) -> None:
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    subjects = np.array(["1"] * n_samples, dtype=object)
    np.savez(path, X=X, y=y, subjects=subjects)


def _make_arch(client_dir: str | None) -> FederatedArchitecture:
    return FederatedArchitecture(
        orchestrator=OrchestratorSpec(
            address="localhost:8080",
            rounds=2,
            protocol="MQTT",
            mqtt=MQTTConfig(
                broker="localhost",
                port=1883,
                topics=MQTTTopics(
                    updates="fl/updates",
                    partial="fl/partial",
                    global_model="fl/global_model",
                ),
            ),
        ),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                k=1,
                clients=[
                    ClientSpec(
                        id="c1",
                        dataset="swell",
                        workflow="swell",
                        rounds=2,
                        data_dir=client_dir,
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=None,
        workflow="swell",
    )


def test_build_runtime_plan_infers_input_dim(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_train_npz(node_dir / "train.npz", n_samples=4, n_features=10)

    arch = _make_arch(str(node_dir))
    repo_root = Path(__file__).resolve().parents[1]
    commands = build_runtime_plan(arch, repo_root=repo_root)

    assert arch.model.input_dim == 10
    roles = [cmd.role for cmd in commands]
    assert "server" in roles


def test_build_runtime_plan_requires_input_dim(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()

    arch = _make_arch(str(node_dir))
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="model.input_dim"):
        build_runtime_plan(arch, repo_root=repo_root)


def test_build_runtime_plan_includes_manifest(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_train_npz(node_dir / "train.npz", n_samples=2, n_features=5)

    arch = _make_arch(str(node_dir))
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    commands = build_runtime_plan(
        arch, repo_root=repo_root, manifest_path=manifest_path
    )

    server_cmd = next(cmd for cmd in commands if cmd.role == "server")
    assert "--manifest" in server_cmd.cmd
    assert str(manifest_path) in server_cmd.cmd


def test_materialize_swell_partitions_updates_arch(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "runs"
    run_name = "run_test"
    expected_out = output_dir / run_name

    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                clients=[
                    ClientSpec(
                        id="c1", dataset="swell", workflow="swell", rounds=1
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=DatasetConfig(
            data_dir=str(tmp_path / "data"),
            output_dir=str(output_dir),
            run_name=run_name,
        ),
        workflow="swell",
    )

    captured = {}

    def _fake_plan(config_path: str):
        captured["config_path"] = Path(config_path)
        return {
            "output_dir": str(expected_out),
            "manifest": {"meta": {"n_features": 7}},
        }

    monkeypatch.setattr(
        "flower_basic.datasets.swell_federated.plan_and_materialize_swell_federated",
        _fake_plan,
    )

    manifest_path = materialize_swell_partitions(arch, repo_root=tmp_path)

    assert manifest_path == expected_out / "manifest.json"
    assert arch.model.input_dim == 7
    assert captured["config_path"].exists()
    cfg = json.loads(captured["config_path"].read_text(encoding="utf-8"))
    assert cfg["dataset"]["data_dir"] == str(tmp_path / "data")
    for client in arch.fog_nodes[0].clients:
        assert client.data_dir == str(expected_out / "fog_0")
