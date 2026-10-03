"""The model bundle: a federation's state on disk, and continuing from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from onion_fl.continuum.bundle import load_bundle, save_bundle
from onion_fl.core.topology import parse_topology
from onion_fl.learning.aggregators import server_optimizers
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import (
    EdgeSpec,
    build_federation,
    restore_federation,
    snapshot_federation,
)

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
INITIAL = state_arrays(ModularMLP(CONFIG, [A], seed=0))
TREE = parse_topology(
    {
        "name": "t",
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud"},
        "fog": {"nodes": [{"id": "fog_0"}, {"id": "fog_1"}]},
    }
)


def federation(rounds: int):
    def noisy(node_id: str, name: str, shift: float) -> EdgeSpec:
        trainer = (
            trainers.create(name, {"shift": shift, "noise": 0.1})
            if name == "stub"
            else trainers.create(name)
        )
        return EdgeSpec(node_id, ModularMLP(CONFIG, [A], seed=0), trainer=trainer)

    edges = {
        "fog_0": [noisy("e1", "stub", 1.0), noisy("e2", "stub", 2.0)],
        "fog_1": [noisy("e3", "stub", 3.0)],
    }
    return build_federation(
        TREE,
        edges,
        initial_state=INITIAL,
        rounds=rounds,
        server_optimizer=server_optimizers.create("fedadam", {"server_lr": 0.1}),
    )


LINEAGE = {"version": 2, "run_id": "r1", "parent": None}
PREPROCESSING = {
    "a": {
        "scaler": "global",
        "features": ["f0", "f1"],
        "fill": [0, 0],
        "mean": [0, 0],
        "std": [1, 1],
    }
}


def test_a_bundle_on_disk_holds_the_whole_snapshot(tmp_path: Path) -> None:
    first = federation(2)
    first.run()
    snapshot = snapshot_federation(first)

    save_bundle(
        tmp_path / "bundle",
        snapshot,
        preprocessing=PREPROCESSING,
        schema={"a": {"task": "t", "n_classes": 2}},
        lineage=LINEAGE,
        config={"name": "demo"},
    )
    bundle = load_bundle(tmp_path / "bundle")

    assert bundle.snapshot.round == snapshot.round
    assert bundle.snapshot.nodes.keys() == snapshot.nodes.keys()
    for node_id, node in snapshot.nodes.items():
        loaded = bundle.snapshot.nodes[node_id]
        assert loaded.meta == node.meta, node_id
        assert loaded.arrays.keys() == node.arrays.keys(), node_id
        for key, value in node.arrays.items():
            np.testing.assert_array_equal(loaded.arrays[key], value, err_msg=key)
    assert bundle.preprocessing == PREPROCESSING and bundle.lineage == LINEAGE
    assert bundle.config == {"name": "demo"}
    assert not list((tmp_path / "bundle").rglob("*.pkl"))


def test_continuing_from_a_bundle_on_disk_equals_never_stopping(tmp_path: Path) -> None:
    straight = federation(4)
    straight.run()
    first = federation(2)
    first.run()
    save_bundle(
        tmp_path / "bundle",
        snapshot_federation(first),
        preprocessing={},
        schema={},
        lineage=LINEAGE,
        config={},
    )

    second = federation(4)
    restore_federation(second, load_bundle(tmp_path / "bundle").snapshot)
    second.run()

    for key, value in straight.coordinator.state.items():
        np.testing.assert_array_equal(second.coordinator.state[key], value, err_msg=key)
