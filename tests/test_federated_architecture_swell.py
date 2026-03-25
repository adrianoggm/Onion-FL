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


def test_build_runtime_plan_skips_empty_fogs(tmp_path: Path) -> None:
    active_dir = tmp_path / "active_node"
    active_dir.mkdir()
    _write_train_npz(active_dir / "train.npz", n_samples=2, n_features=5)

    arch = FederatedArchitecture(
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
                        data_dir=str(active_dir),
                    )
                ],
            ),
            FogNodeSpec(id="fog_empty", k=3, clients=[]),
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=None,
        workflow="swell",
    )

    repo_root = Path(__file__).resolve().parents[1]
    commands = build_runtime_plan(arch, repo_root=repo_root)

    roles = [cmd.role for cmd in commands]
    assert "fog_bridge_fog_0" in roles
    assert "fog_bridge_fog_empty" not in roles
    assert "client_c1" in roles

    server_cmd = next(cmd for cmd in commands if cmd.role == "server")
    assert server_cmd.cmd[server_cmd.cmd.index("--min-fit-clients") + 1] == "1"
    assert server_cmd.cmd[server_cmd.cmd.index("--min-available-clients") + 1] == "1"

    broker_cmd = next(cmd for cmd in commands if cmd.role == "broker")
    assert broker_cmd.env.get("FOG_K_MAP") in (None, '{"fog_0": 1}')


def test_build_runtime_plan_matches_exact_multispawn_topology(tmp_path: Path) -> None:
    node_a = tmp_path / "fog_0_client_1"
    node_b = tmp_path / "fog_0_client_2"
    node_c = tmp_path / "fog_1_client_1"
    node_d = tmp_path / "fog_2_client_1"
    for node_dir in (node_a, node_b, node_c, node_d):
        node_dir.mkdir()
        _write_train_npz(node_dir / "train.npz", n_samples=2, n_features=6)

    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(
            address="localhost:8080",
            rounds=3,
            protocol="MQTT",
            stale_update_policy="strict",
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
                k=2,
                clients=[
                    ClientSpec(
                        id="fog_0_client_1",
                        dataset="swell",
                        workflow="swell",
                        rounds=3,
                        data_dir=str(node_a),
                    ),
                    ClientSpec(
                        id="fog_0_client_2",
                        dataset="swell",
                        workflow="swell",
                        rounds=3,
                        data_dir=str(node_b),
                    ),
                ],
            ),
            FogNodeSpec(
                id="fog_1",
                k=1,
                clients=[
                    ClientSpec(
                        id="fog_1_client_1",
                        dataset="swell",
                        workflow="swell",
                        rounds=3,
                        data_dir=str(node_c),
                    )
                ],
            ),
            FogNodeSpec(
                id="fog_2",
                k=1,
                clients=[
                    ClientSpec(
                        id="fog_2_client_1",
                        dataset="swell",
                        workflow="swell",
                        rounds=3,
                        data_dir=str(node_d),
                    )
                ],
            ),
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=None,
        workflow="swell",
    )

    repo_root = Path(__file__).resolve().parents[1]
    commands = build_runtime_plan(arch, repo_root=repo_root)

    bridge_roles = [cmd.role for cmd in commands if cmd.role.startswith("fog_bridge_")]
    client_roles = [cmd.role for cmd in commands if cmd.role.startswith("client_")]

    assert len(bridge_roles) == 3
    assert len(client_roles) == 4
    assert set(bridge_roles) == {
        "fog_bridge_fog_0",
        "fog_bridge_fog_1",
        "fog_bridge_fog_2",
    }
    assert set(client_roles) == {
        "client_fog_0_client_1",
        "client_fog_0_client_2",
        "client_fog_1_client_1",
        "client_fog_2_client_1",
    }

    server_cmd = next(cmd for cmd in commands if cmd.role == "server")
    assert server_cmd.cmd[server_cmd.cmd.index("--min-fit-clients") + 1] == "3"
    assert server_cmd.cmd[server_cmd.cmd.index("--min-available-clients") + 1] == "3"

    broker_cmd = next(cmd for cmd in commands if cmd.role == "broker")
    assert broker_cmd.cmd[broker_cmd.cmd.index("--stale-update-policy") + 1] == "strict"
    assert broker_cmd.env["FOG_K_MAP"] == '{"fog_0": 2, "fog_1": 1, "fog_2": 1}'


def test_materialize_swell_partitions_updates_arch(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "runs"
    run_name = "run_test"
    expected_out = output_dir / run_name

    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                k=5,
                clients=[
                    ClientSpec(
                        id="template_client",
                        dataset="swell",
                        workflow="swell",
                        rounds=1,
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=DatasetConfig(
            data_dir=str(tmp_path / "data"),
            output_dir=str(output_dir),
            run_name=run_name,
            split_strategy="global",
            test_assignments={"fog_0": [2]},
        ),
        workflow="swell",
    )

    captured = {}

    def _fake_plan(config_path: str):
        captured["config_path"] = Path(config_path)
        subject_1 = expected_out / "fog_0" / "subject_1"
        subject_2 = expected_out / "fog_0" / "subject_2"
        subject_1.mkdir(parents=True, exist_ok=True)
        subject_2.mkdir(parents=True, exist_ok=True)
        _write_train_npz(subject_1 / "train.npz", n_samples=3, n_features=7)
        _write_train_npz(subject_2 / "train.npz", n_samples=0, n_features=7)

        manifest = {
            "nodes": {"fog_0": ["1", "2"]},
            "clients": {
                "fog_0": {
                    "fog_0_client_1": "1",
                    "fog_0_client_2": "2",
                }
            },
            "global_subjects": {"train": ["1"], "all": ["1", "2"]},
            "config": {"split": {"strategy": "global"}},
            "meta": {"n_features": 7},
        }
        (expected_out / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return {"output_dir": str(expected_out), "manifest": manifest}

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
    assert cfg["split"]["strategy"] == "global"
    assert cfg["federation"]["test_assignments"] == {"fog_0": [2]}
    assert [client.id for client in arch.fog_nodes[0].clients] == ["fog_0_client_1"]
    assert arch.fog_nodes[0].clients[0].data_dir == str(
        expected_out / "fog_0" / "subject_1"
    )
    assert arch.fog_nodes[0].k == 1
