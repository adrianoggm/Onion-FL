"""Simulation and real runtime give the same final model (spec §9.3, issue #100).

With full quorum and the same seed, the coordinator's final model must match
in SimRuntime, in RealRuntime over the memory bus, and over MQTT with one
process per group. The edges use the stub trainer with per-node noise: it
learns nothing, but every edge sends a different, seeded update, so any
difference in participants, rng streams or aggregation order would show.
The dataset is a CSV format fixture (docs/RULES.md).
"""

from __future__ import annotations

import os
import socket
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


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_mqtt_with_one_process_per_group_matches_the_simulation(tmp_path: Path) -> None:
    mqtt = {"name": "mqtt", "broker": BROKER}
    expected = simulated(experiment(tmp_path, mqtt))

    (path,) = run_experiment(experiment(tmp_path, mqtt, mode="real"))

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert_same_model(expected, {k: saved[k] for k in saved.files})


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
