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
    apply_manifest_paths,
    build_distribution_payloads,
    build_runtime_plan,
    distribute_architecture,
    materialize_swell_partitions,
    plan_runtime_commands,
    plan_swell_materialization,
    resolve_runtime_architecture,
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
    runtime_plan = build_runtime_plan(arch, repo_root=repo_root)

    assert arch.model.input_dim is None
    assert runtime_plan.architecture.model.input_dim == 10
    roles = [cmd.role for cmd in runtime_plan.commands]
    assert "server" in roles


def test_build_runtime_plan_requires_input_dim(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()

    arch = _make_arch(str(node_dir))
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="model.input_dim"):
        build_runtime_plan(arch, repo_root=repo_root)


def test_resolve_runtime_architecture_returns_independent_copy(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_train_npz(node_dir / "train.npz", n_samples=3, n_features=6)

    arch = _make_arch(str(node_dir))
    resolved = resolve_runtime_architecture(arch, inferred_input_dim=6)
    resolved.fog_nodes[0].clients[0].rounds = 99

    assert resolved.model.input_dim == 6
    assert arch.model.input_dim is None
    assert arch.fog_nodes[0].clients[0].rounds == 2


def test_build_runtime_plan_includes_manifest(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_train_npz(node_dir / "train.npz", n_samples=2, n_features=5)

    arch = _make_arch(str(node_dir))
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "manifest.json"
    runtime_plan = build_runtime_plan(
        arch, repo_root=repo_root, manifest_path=manifest_path
    )

    server_cmd = next(cmd for cmd in runtime_plan.commands if cmd.role == "server")
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
    runtime_plan = build_runtime_plan(arch, repo_root=repo_root)
    commands = runtime_plan.commands

    roles = [cmd.role for cmd in commands]
    assert "fog_bridge_fog_0" in roles
    assert "fog_bridge_fog_empty" not in roles
    assert "client_c1" in roles

    server_cmd = next(cmd for cmd in commands if cmd.role == "server")
    assert server_cmd.cmd[server_cmd.cmd.index("--min-fit-clients") + 1] == "1"
    assert server_cmd.cmd[server_cmd.cmd.index("--min-available-clients") + 1] == "1"

    broker_cmd = next(cmd for cmd in commands if cmd.role == "broker")
    assert broker_cmd.env.get("FOG_K_MAP") in (None, '{"fog_0": 1}')


def test_plan_runtime_commands_rejects_mixed_workflow(tmp_path: Path) -> None:
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    _write_train_npz(node_dir / "train.npz", n_samples=2, n_features=5)

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
                        workflow="wesad",
                        rounds=2,
                        data_dir=str(node_dir),
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=5),
        dataset=None,
        workflow="swell",
    )

    with pytest.raises(ValueError, match="Workflow mixto"):
        plan_runtime_commands(
            arch,
            Path(__file__).resolve().parents[1],
            python_exec="python3",
        )


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
    runtime_plan = build_runtime_plan(arch, repo_root=repo_root)
    commands = runtime_plan.commands

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

    bridge_cmd = next(cmd for cmd in commands if cmd.role == "fog_bridge_fog_0")
    assert bridge_cmd.cmd[bridge_cmd.cmd.index("--server") + 1] == "localhost:8080"

    broker_cmd = next(cmd for cmd in commands if cmd.role == "broker")
    assert broker_cmd.cmd[broker_cmd.cmd.index("--stale-update-policy") + 1] == "strict"
    assert broker_cmd.env["FOG_K_MAP"] == '{"fog_0": 2, "fog_1": 1, "fog_2": 1}'


def test_plan_swell_materialization_requires_run_name(tmp_path: Path) -> None:
    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                clients=[ClientSpec(id="c1", dataset="swell", workflow="swell")],
            )
        ],
        model=ModelConfig(type="swell_mlp"),
        dataset=DatasetConfig(
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "runs"),
            run_name=None,
        ),
        workflow="swell",
    )

    with pytest.raises(ValueError, match="run_name"):
        plan_swell_materialization(arch, repo_root=tmp_path)


def test_plan_swell_materialization_detaches_mutable_inputs(tmp_path: Path) -> None:
    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                clients=[ClientSpec(id="c1", dataset="swell", workflow="swell")],
            )
        ],
        model=ModelConfig(type="swell_mlp"),
        dataset=DatasetConfig(
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "runs"),
            run_name="detached",
            modalities=["hrv"],
            subjects=[1, 2],
            manual_assignments={"fog_0": [1]},
            per_node_percentages=[1.0],
            test_assignments={"fog_0": [2]},
        ),
        workflow="swell",
    )

    plan = plan_swell_materialization(arch, repo_root=tmp_path)
    plan.config["dataset"]["modalities"].append("posture")
    plan.config["federation"]["manual_assignments"]["fog_0"].append(2)

    assert arch.dataset is not None
    assert arch.dataset.modalities == ["hrv"]
    assert arch.dataset.manual_assignments == {"fog_0": [1]}


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
        "flower_basic.federated_architecture._run_swell_materialization",
        _fake_plan,
    )

    plan = plan_swell_materialization(arch, repo_root=tmp_path)
    result = materialize_swell_partitions(arch, repo_root=tmp_path)

    assert plan.config_path == expected_out / "_arch_auto_config.json"
    assert result.manifest_path == expected_out / "manifest.json"
    assert arch.model.input_dim is None
    assert result.architecture.model.input_dim == 7
    assert captured["config_path"].exists()
    cfg = json.loads(captured["config_path"].read_text(encoding="utf-8"))
    assert cfg["dataset"]["data_dir"] == str(tmp_path / "data")
    assert cfg["split"]["strategy"] == "global"
    assert cfg["federation"]["test_assignments"] == {"fog_0": [2]}
    assert [client.id for client in arch.fog_nodes[0].clients] == ["template_client"]
    assert [client.id for client in result.architecture.fog_nodes[0].clients] == [
        "fog_0_client_1"
    ]
    assert result.architecture.fog_nodes[0].clients[0].data_dir == str(
        expected_out / "fog_0" / "subject_1"
    )
    assert arch.fog_nodes[0].k == 5
    assert result.architecture.fog_nodes[0].k == 1


def test_apply_manifest_paths_emits_expected_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "run"
    subject_1 = base / "fog_0" / "subject_1"
    subject_2 = base / "fog_0" / "subject_2"
    subject_1.mkdir(parents=True)
    subject_2.mkdir(parents=True)
    _write_train_npz(subject_1 / "train.npz", n_samples=2, n_features=4)
    _write_train_npz(subject_2 / "train.npz", n_samples=0, n_features=4)

    manifest = {
        "nodes": {"fog_0": ["1", "2"]},
        "clients": {"fog_0": {"c1": "1", "c2": "2"}},
        "global_subjects": {"train": ["1", "2"], "all": ["1", "2"]},
        "config": {"split": {"strategy": "global"}},
        "meta": {"n_features": 4},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    updated_arch = apply_manifest_paths(_make_arch(None), manifest_path)

    captured = capsys.readouterr()
    assert "Strategy: global" in captured.out
    assert "Skipping c2 (train.npz is empty)" in captured.out
    assert [client.id for client in updated_arch.fog_nodes[0].clients] == ["c1"]


def test_apply_manifest_paths_can_disable_output(tmp_path: Path) -> None:
    base = tmp_path / "run"
    subject_1 = base / "fog_0" / "subject_1"
    subject_1.mkdir(parents=True)
    _write_train_npz(subject_1 / "train.npz", n_samples=2, n_features=4)

    manifest = {
        "nodes": {"fog_0": ["1"]},
        "clients": {"fog_0": {"c1": "1"}},
        "global_subjects": {"train": ["1"], "all": ["1"]},
        "config": {"split": {"strategy": "global"}},
        "meta": {"n_features": 4},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    messages: list[str] = []
    updated_arch = apply_manifest_paths(
        _make_arch(None), manifest_path, emit=messages.append
    )

    assert any("fog_0: 1 clientes" in message for message in messages)
    assert updated_arch.model.input_dim == 4


def test_build_distribution_payloads_detach_client_params() -> None:
    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                clients=[
                    ClientSpec(
                        id="c1",
                        dataset="swell",
                        workflow="swell",
                        params={"seed": 7},
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=4),
        workflow="swell",
    )

    payloads = build_distribution_payloads(arch)
    payloads[0]["clients"][0]["params"]["seed"] = 99

    assert arch.fog_nodes[0].clients[0].params["seed"] == 7


def test_distribute_architecture_is_best_effort_on_publish_failure() -> None:
    arch = FederatedArchitecture(
        orchestrator=OrchestratorSpec(),
        fog_nodes=[
            FogNodeSpec(
                id="fog_0",
                clients=[ClientSpec(id="c1", dataset="swell", workflow="swell")],
            ),
            FogNodeSpec(
                id="fog_1",
                clients=[ClientSpec(id="c2", dataset="swell", workflow="swell")],
            ),
        ],
        model=ModelConfig(type="swell_mlp", input_dim=4),
        workflow="swell",
    )

    class _FakeMQTT:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def publish(self, topic: str, _payload: str) -> None:
            self.calls.append(topic)
            if topic.endswith("fog_0"):
                raise RuntimeError("offline")

    fake = _FakeMQTT()
    payloads = distribute_architecture(arch, mqtt_client=fake)

    assert len(payloads) == 2
    assert fake.calls == ["fl/ctrl/plan/fog_0", "fl/ctrl/plan/fog_1"]
