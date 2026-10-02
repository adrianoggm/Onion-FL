"""Tests for run identity, the event schema and runs/<run_id>/ (issue #93).

The federations use the stub trainer; nothing is trained (docs/RULES.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from onion_fl.core.ids import code_version, config_id, data_id, new_run_id
from onion_fl.core.topology import parse_topology
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.observability.events import SCHEMA, event_fingerprint, kind_of
from onion_fl.observability.run import Run, verify_run
from onion_fl.roles import EdgeSpec, build_federation

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)


def model() -> ModularMLP:
    return ModularMLP(CONFIG, [A], seed=0)


TOPOLOGY = parse_topology(
    {
        "name": "t",
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud"},
        "fog": {
            "defaults": {"participation": {"name": "fraction", "p": 0.5}},
            "nodes": [{"id": "fog_0"}],
        },
    }
)


def federation(seed: int = 0):
    edges = {
        "fog_0": [
            EdgeSpec(f"e{i}", model(), trainer=trainers.create("stub"))
            for i in range(4)
        ]
    }
    return build_federation(
        TOPOLOGY, edges, initial_state=state_arrays(model()), rounds=3, seed=seed
    )


def recorded(tmp_path: Path, seed: int = 0, config: dict | None = None) -> Path:
    fed = federation(seed)
    run = Run(tmp_path, config=config or {"rounds": 3}, topology=TOPOLOGY, seed=seed)
    run.attach(fed)
    fed.run()
    run.finish()
    return run.path


# --- identifiers ----------------------------------------------------------------------------


def test_run_ids_sort_by_date_and_never_repeat() -> None:
    ids = {new_run_id("cfg", seed=0) for _ in range(50)}

    assert len(ids) == 50
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{12}", i) for i in ids)


def test_config_id_ignores_key_order_and_changes_with_values() -> None:
    a = config_id({"rounds": 3, "model": {"width": 8, "depth": 2}})
    b = config_id({"model": {"depth": 2, "width": 8}, "rounds": 3})

    assert a == b and len(a) == 64
    assert config_id({"rounds": 4, "model": {"width": 8, "depth": 2}}) != a


def test_data_id_ignores_the_order_of_the_caches() -> None:
    assert data_id({"swell": "aa", "sweet": "bb"}) == data_id(
        {"sweet": "bb", "swell": "aa"}
    )
    assert data_id({"swell": "aa"}) != data_id({"swell": "ab"})


def test_code_version_names_the_commit() -> None:
    version = code_version()

    assert set(version) == {"commit", "dirty"}
    assert version["commit"] == "unknown" or re.fullmatch(
        r"[0-9a-f]{40}", version["commit"]
    )


# --- events -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, kind",
    [
        ("message.sent", "message"),
        ("link.dropped", "message"),
        ("eval.accuracy", "metric"),
        ("round.train_loss", "metric"),
        ("edge.trained", "metric"),
        ("round.closed", "lifecycle"),
        ("run.finished", "lifecycle"),
        ("diagnostic.divergence", "diagnostic"),
        ("data.composition", "data"),
    ],
)
def test_events_are_classified_by_name(name: str, kind: str) -> None:
    assert kind_of(name) == kind


def test_each_event_follows_the_schema(tmp_path: Path) -> None:
    path = recorded(tmp_path)

    lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert events and all(set(e) == set(SCHEMA) for e in events)
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert {e["run_id"] for e in events} == {run["run_id"]}
    assert {e["topology_id"] for e in events} == {TOPOLOGY.topology_id}


def test_events_carry_level_role_and_round(tmp_path: Path) -> None:
    events = [
        json.loads(line)
        for line in (recorded(tmp_path) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    def first(**match):
        return next(e for e in events if all(e[k] == v for k, v in match.items()))

    assert first(name="round.started")["level"] == "global"
    assert first(name="round.started")["role"] == "coordinator"
    assert first(name="round.participants", node="fog_0")["level"] == "fog"
    assert first(name="round.participants", node="fog_0")["round"] == 1
    trained = first(name="edge.trained")
    assert (trained["level"], trained["role"], trained["kind"]) == (
        "edge",
        "edge",
        "metric",
    )


# --- runs/<run_id>/ ----------------------------------------------------------------------------


def test_a_run_writes_its_folder(tmp_path: Path) -> None:
    path = recorded(tmp_path, config={"rounds": 3})

    assert path.parent == tmp_path
    assert sorted(p.name for p in path.iterdir()) == [
        "events.jsonl",
        "run.json",
        "summary.json",
    ]
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert run["run_id"] == path.name
    assert run["topology_id"] == TOPOLOGY.topology_id
    assert run["config_id"] == config_id({"rounds": 3})
    assert run["status"] == "finished"
    assert run["started_at"] <= run["finished_at"]
    assert {"host", "pid", "code_version", "seed", "data_id", "run_hash"} <= set(run)


def test_run_json_exists_while_running(tmp_path: Path) -> None:
    run = Run(tmp_path, config={}, topology=TOPOLOGY, seed=0)

    status = json.loads((run.path / "run.json").read_text(encoding="utf-8"))["status"]

    assert status == "running"


def test_the_summary_reports_rounds_and_traffic(tmp_path: Path) -> None:
    summary = json.loads(
        (recorded(tmp_path) / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["rounds"] == 3 and summary["finished"] is True
    assert summary["messages"]["sent"] > 0 and summary["messages"]["bytes"] > 0
    assert summary["messages"]["dropped"] == 0
    assert summary["quorum_failed"] == 0


def test_the_run_hash_detects_any_change(tmp_path: Path) -> None:
    path = recorded(tmp_path)

    assert verify_run(path)
    events = path / "events.jsonl"
    events.write_text(events.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert not verify_run(path)


def test_same_config_and_seed_give_the_same_event_fingerprint(tmp_path: Path) -> None:
    a = recorded(tmp_path / "a", seed=3)
    b = recorded(tmp_path / "b", seed=3)
    c = recorded(tmp_path / "c", seed=4)

    assert a.name != b.name  # different runs ...
    assert event_fingerprint(a / "events.jsonl") == event_fingerprint(
        b / "events.jsonl"
    )
    assert event_fingerprint(a / "events.jsonl") != event_fingerprint(
        c / "events.jsonl"
    )


def test_runner_level_events_can_be_recorded(tmp_path: Path) -> None:
    run = Run(tmp_path, config={}, topology=TOPOLOGY, seed=0)

    run.record("data.composition", None, leaf="fog_0", samples=12)
    run.finish()

    (event,) = [
        json.loads(line)
        for line in (run.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert event["kind"] == "data" and event["node"] == "experiment"
    assert event["tags"] == {"leaf": "fog_0", "samples": 12}


def test_the_summary_keeps_the_last_global_scores(tmp_path: Path) -> None:
    def score(model, data):
        return {"accuracy": float(data)}, 1

    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", "eval": {"every": 1}},
            "fog": {"nodes": [{"id": "fog_0"}]},
        }
    )
    edges = {
        "fog_0": [EdgeSpec("e1", model(), trainer=trainers.create("stub"))],
        "cloud": [
            EdgeSpec("t1", model(), data=0.75, train=False, tags={"dataset": "a"})
        ],
    }
    fed = build_federation(
        topology, edges, initial_state=state_arrays(model()), rounds=2, evaluate=score
    )
    run = Run(tmp_path, config={}, topology=topology, seed=0)
    run.attach(fed)
    fed.run()
    run.finish()

    final = json.loads((run.path / "summary.json").read_text(encoding="utf-8"))["final"]
    assert final["global/a"] == {"round": 2, "accuracy": 0.75}
    assert final["global/*"]["round"] == 2


def test_code_version_without_git(monkeypatch) -> None:
    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)

    assert code_version() == {"commit": "unknown", "dirty": None}


def test_numpy_values_are_written_as_plain_json() -> None:
    import numpy as np

    from onion_fl.observability.events import dumps

    line = dumps({"a": np.float32(1.5), "b": np.arange(2), "c": Path("x")})

    assert json.loads(line) == {"a": 1.5, "b": [0, 1], "c": "x"}


def test_the_final_model_is_saved_and_signed(tmp_path: Path) -> None:
    import numpy as np

    from onion_fl.observability.run import save_model

    fed = federation(0)
    run = Run(tmp_path, config={}, topology=TOPOLOGY, seed=0)
    run.attach(fed)
    fed.run()
    save_model(run.path, fed.coordinator.state)
    run.finish()

    with np.load(run.path / "model.npz", allow_pickle=False) as saved:
        for key, value in fed.coordinator.state.items():
            np.testing.assert_array_equal(saved[key], value)
    assert verify_run(run.path)
    np.savez(
        run.path / "model.npz", **{k: v + 1 for k, v in fed.coordinator.state.items()}
    )
    assert not verify_run(run.path)
