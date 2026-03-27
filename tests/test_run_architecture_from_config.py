from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flower_basic.federated_architecture import (
    ClientSpec,
    FederatedArchitecture,
    FogNodeSpec,
    ModelConfig,
    MQTTConfig,
    MQTTTopics,
    OrchestratorSpec,
)


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


def _make_arch(k: int = 0) -> FederatedArchitecture:
    return FederatedArchitecture(
        orchestrator=OrchestratorSpec(
            address="localhost:8080",
            rounds=3,
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
                k=k,
                clients=[
                    ClientSpec(
                        id="c1",
                        dataset="swell",
                        workflow="swell",
                        rounds=3,
                        data_dir=None,
                    )
                ],
            )
        ],
        model=ModelConfig(type="swell_mlp", input_dim=None),
        dataset=None,
        workflow="swell",
    )


def test_apply_manifest_paths_global_strategy(tmp_path: Path) -> None:
    _add_scripts_to_path()
    import run_architecture_from_config as rac

    base = tmp_path / "run"
    subject_1 = base / "fog_0" / "subject_1"
    subject_2 = base / "fog_0" / "subject_2"
    subject_1.mkdir(parents=True)
    subject_2.mkdir(parents=True)
    _write_npz(subject_1 / "train.npz", n_samples=4, n_features=5, subject="1")
    _write_npz(subject_2 / "train.npz", n_samples=0, n_features=5, subject="2")

    manifest = {
        "nodes": {"fog_0": ["1", "2"]},
        "clients": {"fog_0": {"c1": "1", "c2": "2"}},
        "global_subjects": {"train": ["1"], "all": ["1", "2"]},
        "config": {"split": {"strategy": "global"}},
        "meta": {"n_features": 5},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    arch = _make_arch()
    updated_arch = rac._apply_manifest_paths(arch, manifest_path)

    fog = updated_arch.fog_nodes[0]
    assert len(fog.clients) == 1
    assert fog.clients[0].id == "c1"
    assert fog.clients[0].data_dir == str(subject_1)
    assert fog.k == 1
    assert updated_arch.model.input_dim == 5
    assert arch.model.input_dim is None


def test_apply_manifest_paths_per_subject_strategy(tmp_path: Path) -> None:
    _add_scripts_to_path()
    import run_architecture_from_config as rac

    base = tmp_path / "run"
    subject_1 = base / "fog_0" / "subject_1"
    subject_2 = base / "fog_0" / "subject_2"
    subject_1.mkdir(parents=True)
    subject_2.mkdir(parents=True)
    _write_npz(subject_1 / "train.npz", n_samples=2, n_features=4, subject="1")
    _write_npz(subject_2 / "train.npz", n_samples=0, n_features=4, subject="2")

    manifest = {
        "nodes": {"fog_0": ["1", "2"]},
        "clients": {"fog_0": {"c1": "1", "c2": "2"}},
        "global_subjects": {"all": ["1", "2"]},
        "config": {"split": {"strategy": "per_subject"}},
        "meta": {"n_features": 4},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    arch = _make_arch()
    updated_arch = rac._apply_manifest_paths(arch, manifest_path)

    fog = updated_arch.fog_nodes[0]
    assert [c.id for c in fog.clients] == ["c1"]
    assert fog.clients[0].data_dir == str(subject_1)
    assert fog.k == 1
    assert updated_arch.model.input_dim == 4
    assert arch.model.input_dim is None


def test_apply_manifest_paths_clamps_k_to_spawned_clients(tmp_path: Path) -> None:
    _add_scripts_to_path()
    import run_architecture_from_config as rac

    base = tmp_path / "run"
    subject_1 = base / "fog_0" / "subject_1"
    subject_1.mkdir(parents=True)
    _write_npz(subject_1 / "train.npz", n_samples=4, n_features=5, subject="1")

    manifest = {
        "nodes": {"fog_0": ["1"]},
        "clients": {"fog_0": {"c1": "1"}},
        "global_subjects": {"train": ["1"], "all": ["1"]},
        "config": {"split": {"strategy": "global"}},
        "meta": {"n_features": 5},
    }
    manifest_path = base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    arch = _make_arch(k=5)
    updated_arch = rac._apply_manifest_paths(arch, manifest_path)

    fog = updated_arch.fog_nodes[0]
    assert fog.k == 1
    assert arch.fog_nodes[0].k == 5
