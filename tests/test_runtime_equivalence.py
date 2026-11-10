"""Simulation and real runtime give the same final model (spec §9.3, issue #100).

With full quorum and the same seed, the coordinator's final model must match
in SimRuntime, in RealRuntime over the memory bus, gRPC and Flower, and over
MQTT, gRPC and Flower with one process per group. Flower needs the ``flower``
extra, and its tests skip without it. The edges use the stub trainer with
per-node noise: it
learns nothing, but every edge sends a different, seeded update, so any
difference in participants, rng streams or aggregation order would show.
The dataset is a CSV format fixture (docs/RULES.md).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.runner import _scenario_data, build_scenario, run_experiment
from onion_fl.experiment.sweep import scenarios
from onion_fl.runtime.real import RealRuntime

BROKER = os.environ.get("ONIONFL_MQTT", "localhost:1883")


def _broker_up() -> bool:
    host, _, port = BROKER.partition(":")
    try:
        with socket.create_connection((host, int(port or 1883)), timeout=0.5):
            return True
    except OSError:
        return False


def experiment(tmp_path: Path, transport, mode: str = "sim"):
    rows = ["pp,cond,f1,f2"] + [
        f"{s},{'NT'[i % 2]},{s + i},{s * 3 - i}" for s in range(1, 11) for i in range(4)
    ]
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {(tmp_path / "raw").as_posix()}
            source: {{reader: csv, path: t.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    link = {"transport": transport}
    return parse_experiment(
        {
            "name": "equivalence",
            "topology": {
                "name": "two_fogs",
                "levels": ["global", "fog", "edge"],
                "root": {"id": "cloud"},
                "fog": {
                    "defaults": {"link_up": link},
                    "nodes": [{"id": "fog_a"}, {"id": "fog_b"}],
                },
                "edge": {"link_up": link},
            },
            "data": {
                "datasets": {"demo": {"descriptor": str(tmp_path / "demo.yaml")}},
                "roles": {"test": 0.2},
                "placement": {"name": "mixing", "alpha": 1.0},
            },
            "learning": {
                "model": {
                    "name": "modular_mlp",
                    "adapter_width": 4,
                    "trunk_hidden": [4],
                },
                "trainer": {"name": "stub", "shift": 0.0, "noise": 0.1},
            },
            "rounds": 3,
            "evaluation": {"global": {"every": None}},
            "runtime": {"mode": mode, "timeout": 60, "heartbeat": None},
            "paths": {"runs": str(tmp_path / "runs"), "cache": str(tmp_path / "cache")},
        }
    )


def simulated(config) -> dict[str, np.ndarray]:
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    federation = build_scenario(scenario, topology, split, placement)
    federation.run()
    assert federation.coordinator.finished
    return federation.coordinator.state


def assert_same_model(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-6, atol=1e-7, err_msg=key)


def test_the_noise_makes_the_check_meaningful(tmp_path: Path) -> None:
    config = experiment(tmp_path, "memory")
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    initial = build_scenario(scenario, topology, split, placement).coordinator.state

    final = simulated(config)

    assert any(not np.allclose(final[k], initial[k]) for k in final)


def test_the_real_runtime_in_one_process_matches_the_simulation(tmp_path: Path) -> None:
    config = experiment(tmp_path, {"name": "memory", "bus": "equivalence"})
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    runtime = RealRuntime("equivalence", seed=scenario.seed, heartbeat_s=None)
    federation = build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    assert federation.coordinator.finished
    assert_same_model(simulated(config), federation.coordinator.state)


DIRECT = ["grpc", "flower"]  # transports with direct links, no broker


def _direct(tmp_path: Path, name: str = "grpc") -> dict:
    if name == "flower":
        pytest.importorskip("flwr", minversion="1.21")
    pytest.importorskip("grpc")
    return {"name": name, "directory": str(tmp_path / f"peers_{name}")}


@pytest.mark.parametrize("name", DIRECT)
def test_the_real_runtime_over_a_direct_transport_matches_the_simulation(
    tmp_path: Path, name: str
) -> None:
    config = experiment(tmp_path, _direct(tmp_path, name))
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    runtime = RealRuntime(f"equivalence-{name}", seed=scenario.seed, heartbeat_s=None)
    federation = build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    assert federation.coordinator.finished
    assert_same_model(simulated(config), federation.coordinator.state)


@pytest.mark.parametrize("name", DIRECT)
def test_a_direct_transport_with_one_process_per_group_matches_the_simulation(
    tmp_path: Path, name: str
) -> None:
    transport = _direct(tmp_path, name)
    expected = simulated(experiment(tmp_path, transport))

    (path,) = run_experiment(experiment(tmp_path, transport, mode="real"))

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})


def test_local_over_grpc_and_over_flower_give_the_same_model_and_bytes(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    for name in DIRECT:
        raw = experiment(tmp_path, _direct(tmp_path, name)).dump()
        raw = raw | {"backend": "local"}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{name}")}
        run_experiment(parse_experiment(raw))

    result = fidelity(
        tmp_path / "runs_grpc", tmp_path / "runs_flower", metrics=["time"]
    )

    (pair,) = result["pairs"]
    assert pair["functional"]["model"]["bitwise"], pair["functional"]
    assert result["functional"]["complete"], pair["functional"]
    assert pair["functional"]["bytes"]["a"] == pair["functional"]["bytes"]["b"] > 0


def _round_bytes(events) -> dict:
    out: dict = {}
    for e in events:
        if e["name"] == "message.sent" and e["tags"]["round"] is not None:
            key = (e["node"], e["tags"]["dst"], e["tags"]["round"])
            out[key] = out.get(key, 0) + e["value"]
    return out


def test_the_real_runtime_counts_the_bytes_the_simulation_counts(
    tmp_path: Path,
) -> None:
    config = experiment(tmp_path, {"name": "memory", "bus": "bytes"})
    (scenario,) = scenarios(config)
    topology, split, placement, _ = _scenario_data(scenario)
    sim = build_scenario(scenario, topology, split, placement)
    sim.run()
    runtime = RealRuntime("bytes", seed=scenario.seed, heartbeat_s=None)
    build_scenario(scenario, topology, split, placement, runtime=runtime)

    runtime.run(timeout=30)

    expected = _round_bytes(sim.runtime.events)
    assert expected and _round_bytes(runtime.events) == expected
    framed = [e for e in runtime.events if e["name"] == "message.sent"]
    assert all(e["tags"]["frame"] > 0 for e in framed)


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_mqtt_with_one_process_per_group_matches_the_simulation(tmp_path: Path) -> None:
    mqtt = {"name": "mqtt", "broker": BROKER}
    expected = simulated(experiment(tmp_path, mqtt))

    (path,) = run_experiment(experiment(tmp_path, mqtt, mode="real"))

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_the_fidelity_of_local_against_sim_is_functionally_complete(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    mqtt = {"name": "mqtt", "broker": BROKER}
    folders = {}
    for backend in ("sim", "local"):
        raw = experiment(tmp_path, mqtt).dump() | {"backend": backend}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{backend}")}
        run_experiment(parse_experiment(raw))
        folders[backend] = tmp_path / f"runs_{backend}"

    result = fidelity(folders["local"], folders["sim"], metrics=["time"])

    assert result["backends"] == {"a": ["local"], "b": ["sim"]}
    assert result["functional"]["complete"], result["pairs"][0]["functional"]
    assert {"round_time", "link_latency", "duration"} <= set(result["performance"])


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_a_real_run_feeds_its_sinks_and_labels_its_traffic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onion_fl.experiment import runner
    from onion_fl.observability.events import read_events

    seen = []

    class Recorder:
        def write(self, event) -> None:
            seen.append(event["name"])

        def close(self) -> None:
            pass

    monkeypatch.setattr(runner, "_sinks", lambda config: [Recorder()])
    mqtt = {"name": "mqtt", "broker": BROKER}

    (path,) = run_experiment(experiment(tmp_path, mqtt, mode="real"))

    traffic = [
        e
        for e in read_events(path / "events.jsonl")
        if e["name"] == "diagnostic.communication"
    ]
    assert traffic and all(e["level"] and e["role"] for e in traffic)
    assert {"run.finished", "diagnostic.communication"} <= set(seen)


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_local_over_mqtt_and_over_grpc_give_the_same_model_and_measure_the_rest(
    tmp_path: Path,
) -> None:
    from onion_fl.experiment.fidelity import fidelity

    transports = {"mqtt": {"name": "mqtt", "broker": BROKER}, "grpc": _direct(tmp_path)}
    for name, transport in transports.items():
        raw = experiment(tmp_path, transport).dump() | {"backend": "local"}
        raw["paths"] = raw["paths"] | {"runs": str(tmp_path / f"runs_{name}")}
        run_experiment(parse_experiment(raw))

    result = fidelity(tmp_path / "runs_mqtt", tmp_path / "runs_grpc", metrics=["time"])

    (pair,) = result["pairs"]
    assert pair["functional"]["model"]["bitwise"], pair["functional"]
    assert result["functional"]["complete"], pair["functional"]
    assert pair["functional"]["bytes"]["a"] == pair["functional"]["bytes"]["b"] > 0
    assert {"round_time", "link_latency", "duration"} <= set(result["performance"])


# --- the emulated backend (issue #195) -------------------------------------------

EMULATED = {
    "name": "emulated",
    "image": os.environ.get("ONIONFL_EMULATED_IMAGE", "onionfl-emulated:latest"),
    "project": os.environ.get("ONIONFL_EMULATED_PROJECT", "onionfl-test"),
    "subnet": os.environ.get("ONIONFL_EMULATED_SUBNET", "10.195.0.0/16"),
}


def _emulator_up() -> bool:
    """Docker answers and has the host image: the emulator is there."""
    try:
        found = subprocess.run(
            ["docker", "image", "inspect", EMULATED["image"]],
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return found.returncode == 0


LAB = {
    "name": "emulated_lab",
    "profiles": {
        "cloud": {"cpus": 2, "memory": "2GB"},
        "fog": {"cpus": 1, "memory": "1GB"},
        "device": {"cpus": 1, "memory": "1GB"},
    },
    "sites": {
        "dc": {"hosts": {"cloud-1": "cloud"}},
        "fog_a": {"hosts": {"fog-a": "fog"}},
        "fog_b": {"hosts": {"fog-b": "fog"}},
        "field": {"pool": {"device": 4}},
    },
    "network": {
        "links": [
            {"between": ["field", "fog_a"], "latency": "5ms", "bandwidth": "50Mbps"},
            {"between": ["field", "fog_b"], "latency": "10ms", "bandwidth": "20Mbps"},
            {"between": ["fog_a", "dc"], "latency": "20ms", "bandwidth": "10Mbps"},
            {"between": ["fog_b", "dc"], "latency": "30ms", "bandwidth": "10Mbps"},
        ]
    },
    "services": {"mqtt_broker": "cloud-1"},
}


def on_the_lab(config, backend, runs: Path):
    """``config`` on ``LAB``, two edges per device, run by ``backend``."""
    raw = config.dump() | {
        "backend": backend,
        "infrastructure": {
            "name": LAB,
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-a", "fog_b": "fog-b"},
                "edges": {"name": "colocated", "per_host": 2},
            },
        },
    }
    raw["paths"] = raw["paths"] | {"runs": str(runs)}
    return parse_experiment(raw)


@pytest.mark.skipif(not _emulator_up(), reason="no Docker or no emulated host image")
def test_the_emulated_backend_matches_the_simulation(tmp_path: Path) -> None:
    from onion_fl.experiment.fidelity import fidelity
    from onion_fl.observability.events import read_events

    mqtt = {"name": "mqtt", "broker": BROKER}  # the emulation brings its own broker
    base = experiment(tmp_path, mqtt)
    sim = on_the_lab(base, "sim", tmp_path / "runs_sim")
    emulated = on_the_lab(base, EMULATED, tmp_path / "runs_emulated")

    (sim_path,) = run_experiment(sim)
    (path,) = run_experiment(emulated)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished", meta.get("processes")
    assert meta["backend"] == "emulated" and meta["infrastructure_id"]
    assert meta["emulation"]["containers"] == 12  # 7 hosts, 4 routers, the broker
    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(simulated(sim), {k: saved[k] for k in saved.files})
    expected = _round_bytes(read_events(sim_path / "events.jsonl"))
    assert expected and _round_bytes(read_events(path / "events.jsonl")) == expected
    result = fidelity(
        tmp_path / "runs_emulated", tmp_path / "runs_sim", metrics=["time"]
    )
    functional = result["pairs"][0]["functional"]
    assert result["functional"]["complete"], functional
    assert functional["model"]["bitwise"], functional["model"]
