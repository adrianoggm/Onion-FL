import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flower_basic.federated_architecture import (
    build_runtime_plan,
    infer_primary_workflow,
    load_architecture_config,
)


def _make_config(workflow: str = "swell") -> dict:
    return {
        "workflow": workflow,
        "orchestrator": {
            "address": "localhost:8080",
            "rounds": 2,
            "protocol": "MQTT",
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "topics": {"updates": "fl/updates", "partial": "fl/partial", "global_model": "fl/global_model"},
            },
        },
        "model": {"type": "swell_mlp", "input_dim": 178},
        "fog_nodes": [
            {
                "id": "fog_0",
                "k": 2,
                "clients": [
                    {"id": "c1", "dataset": workflow, "rounds": 2, "data_dir": "data/fog_0"},
                ],
            },
            {
                "id": "fog_1",
                "k": 3,
                "clients": [
                    {"id": "c2", "dataset": workflow, "rounds": 2, "data_dir": "data/fog_1"},
                ],
            },
        ],
    }


def test_load_architecture_config_from_json(tmp_path: Path):
    cfg = {"federated_architecture": _make_config()}
    cfg_path = tmp_path / "arch.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    arch = load_architecture_config(cfg_path)

    assert arch.orchestrator.rounds == 2
    assert len(arch.fog_nodes) == 2
    assert infer_primary_workflow(arch) == "swell"


def test_runtime_plan_includes_k_map(tmp_path: Path):
    cfg = {"federated_architecture": _make_config()}
    cfg_path = tmp_path / "arch.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    arch = load_architecture_config(cfg_path)

    repo_root = Path(__file__).resolve().parents[1]
    commands = build_runtime_plan(arch, repo_root=repo_root, manifest_path=None)

    roles = [c.role for c in commands]
    assert "server" in roles and "broker" in roles
    assert any(r.startswith("fog_bridge") for r in roles)
    broker_cmd = next(c for c in commands if c.role == "broker")
    assert "FOG_K_MAP" in broker_cmd.env
    client_roles = [r for r in roles if r.startswith("client_")]
    assert len(client_roles) == 2
