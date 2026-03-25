from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _add_scripts_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _write_npz(path: Path, n_samples: int, n_features: int, subject: str) -> None:
    if n_samples > 0:
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, size=n_samples).astype(np.int64)
        subjects = np.array([subject] * n_samples, dtype=object)
    else:
        X = np.empty((0, n_features), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        subjects = np.empty((0,), dtype=object)
    np.savez(path, X=X, y=y, subjects=subjects)


def test_run_swell_demo_builds_current_module_plan(tmp_path: Path) -> None:
    _add_scripts_to_path()
    import run_swell_federated_demo as swell_demo

    base = tmp_path / "swell_run"
    fog0_subj1 = base / "fog_0" / "subject_1"
    fog0_subj2 = base / "fog_0" / "subject_2"
    fog1_subj3 = base / "fog_1" / "subject_3"
    for path in (fog0_subj1, fog0_subj2, fog1_subj3):
        path.mkdir(parents=True)

    _write_npz(fog0_subj1 / "train.npz", n_samples=4, n_features=5, subject="1")
    _write_npz(fog0_subj2 / "train.npz", n_samples=3, n_features=5, subject="2")
    _write_npz(fog1_subj3 / "train.npz", n_samples=2, n_features=5, subject="3")

    manifest = {
        "nodes": {"fog_0": ["1", "2"], "fog_1": ["3"]},
        "clients": {
            "fog_0": {"fog_0_client_1": "1", "fog_0_client_2": "2"},
            "fog_1": {"fog_1_client_3": "3"},
        },
        "global_subjects": {"train": ["1", "2", "3"], "all": ["1", "2", "3"]},
        "config": {"split": {"strategy": "global"}},
        "meta": {"n_features": 5},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    commands, active_nodes = swell_demo.build_launch_plan(
        manifest_path=manifest_path,
        rounds=3,
        server_addr="localhost:8080",
        max_clients_per_node=0,
        k_per_region=2,
        mqtt_broker="localhost",
        mqtt_port=1883,
        topic_updates="fl/updates",
        topic_partial="fl/partial",
        topic_global="fl/global_model",
        python_exec="python",
    )

    roles = [cmd.role for cmd in commands]
    assert roles[:4] == ["server", "fog_bridge_fog_0", "fog_bridge_fog_1", "broker"]
    assert "client_fog_0_client_1" in roles
    assert "client_fog_0_client_2" in roles
    assert "client_fog_1_client_3" in roles
    assert set(active_nodes.keys()) == {"fog_0", "fog_1"}

    server_cmd = commands[0].cmd
    assert server_cmd[:3] == ["python", "-m", "flower_basic.servers.swell"]
    assert server_cmd[server_cmd.index("--server_addr") + 1] == "localhost:8080"
    assert "--manifest" in server_cmd

    broker_cmd = next(cmd.cmd for cmd in commands if cmd.role == "broker")
    assert broker_cmd[:3] == ["python", "-m", "flower_basic.brokers.fog"]
    assert "--k-map" in broker_cmd

    client_cmd = next(
        cmd.cmd for cmd in commands if cmd.role == "client_fog_0_client_1"
    )
    assert client_cmd[:3] == ["python", "-m", "flower_basic.clients.swell"]
    assert str(fog0_subj1) in client_cmd
    assert "fog_0_client_1" in client_cmd


def test_run_sweet_demo_builds_current_module_plan(tmp_path: Path) -> None:
    _add_scripts_to_path()
    import run_sweet_federated_demo as sweet_demo

    base = tmp_path / "sweet_run"
    fog0_subj1 = base / "fog_0" / "subject_user0001"
    fog1_subj2 = base / "fog_1" / "subject_user0002"
    for path in (fog0_subj1, fog1_subj2):
        path.mkdir(parents=True)

    _write_npz(fog0_subj1 / "train.npz", n_samples=5, n_features=7, subject="user0001")
    _write_npz(fog1_subj2 / "train.npz", n_samples=0, n_features=7, subject="user0002")

    manifest = {
        "clients": {
            "fog_0": {"fog_0_client_user0001": "user0001"},
            "fog_1": {"fog_1_client_user0002": "user0002"},
        },
        "meta": {"n_features": 7},
        "output_dir": str(base),
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    commands, active_nodes, total_clients = sweet_demo.build_launch_plan(
        manifest_path=manifest_path,
        config_data={"dataset": {"label_strategy": "binary"}},
        num_rounds=5,
        server_addr="localhost:8080",
        mqtt_broker="localhost",
        mqtt_port=1883,
        enable_telemetry=True,
        enable_prometheus=True,
        hidden_dims=[32, 16],
        num_classes=None,
        python_exec="python",
    )

    roles = [cmd.role for cmd in commands]
    assert roles[:3] == ["server", "broker", "fog_bridge_fog_0"]
    assert "fog_bridge_fog_1" not in roles
    assert "client_fog_0_user0001" in roles
    assert "client_fog_1_user0002" not in roles
    assert set(active_nodes.keys()) == {"fog_0"}
    assert total_clients == 1

    server_cmd = commands[0].cmd
    assert server_cmd[:3] == ["python", "-m", "flower_basic.servers.sweet"]
    assert "--num-classes" in server_cmd
    assert server_cmd[server_cmd.index("--num-classes") + 1] == "2"
    assert server_cmd[server_cmd.index("--min-fit-clients") + 1] == "1"

    broker_cmd = commands[1].cmd
    assert broker_cmd[:3] == ["python", "-m", "flower_basic.brokers.sweet_fog"]

    client_cmd = next(
        cmd.cmd for cmd in commands if cmd.role == "client_fog_0_user0001"
    )
    assert client_cmd[:3] == ["python", "-m", "flower_basic.clients.sweet"]
    assert "--enable-telemetry" in client_cmd
    assert "--enable-prometheus" in client_cmd


def test_run_sweet_architecture_skips_empty_nodes(tmp_path: Path, monkeypatch) -> None:
    _add_scripts_to_path()
    import run_sweet_architecture as sweet_arch

    manifest = {
        "nodes": {"fog_0": ["user0001"], "fog_1": ["user0002"]},
        "clients": {
            "fog_0": {"fog_0_client_user0001": "user0001"},
            "fog_1": {},
        },
        "meta": {"n_features": 6},
        "output_dir": str(tmp_path / "sweet_run"),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    popen_calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            return None

    def _fake_popen(cmd, *args, **kwargs):
        popen_calls.append(cmd)
        return _FakeProc(pid=len(popen_calls))

    monkeypatch.setattr(sweet_arch.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(sweet_arch.time, "sleep", lambda _s: None)

    config = {
        "federated_architecture": {
            "orchestrator": {
                "address": "0.0.0.0:9090",
                "rounds": 4,
                "mqtt": {"broker": "localhost", "port": 1883},
            },
            "model": {"num_classes": 2, "hidden_dims": [32, 16], "lr": 0.001},
        }
    }

    procs = sweet_arch.launch_federated_system(
        config=config,
        manifest_path=manifest_path,
        mqtt_check=False,
    )

    assert len(procs) == 4
    assert popen_calls[0][:3] == [sys.executable, "-m", "flower_basic.servers.sweet"]
    assert popen_calls[0][popen_calls[0].index("--server-addr") + 1] == "0.0.0.0:9090"
    assert popen_calls[1][:3] == [
        sys.executable,
        "-m",
        "flower_basic.brokers.sweet_fog",
    ]
    assert popen_calls[2][:3] == [
        sys.executable,
        "-m",
        "flower_basic.clients.fog_bridge_sweet",
    ]
    assert popen_calls[2][popen_calls[2].index("--server") + 1] == "localhost:9090"
    assert "fog_0" in popen_calls[2]
    assert popen_calls[3][:3] == [sys.executable, "-m", "flower_basic.clients.sweet"]
    assert "user0001" in popen_calls[3]
    assert "fog_1" not in " ".join(" ".join(cmd) for cmd in popen_calls[2:])
