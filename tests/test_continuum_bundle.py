"""The model bundle: a federation's state on disk, and continuing from it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


# --- real data: SCAFFOLD trains, so it needs the real extract (docs/RULES.md) ----------

SWELL_SAMPLE = Path("data/samples/swell_real_sample.pkl")
SWELL = DataShape(dataset="swell", task="stress_binary", n_features=16, n_classes=2)


@pytest.mark.skipif(not SWELL_SAMPLE.exists(), reason="swell sample not available")
def test_scaffold_continues_from_a_bundle_as_if_it_never_stopped(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from onion_fl.core.context import rng_state
    from onion_fl.datasets.samples import load_swell_sample_features

    X, y = load_swell_sample_features(SWELL_SAMPLE)
    X = ((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)).astype(np.float32)
    parts = np.array_split(np.arange(len(y)), 3)
    config = ModularMLPConfig(adapter_width=8, trunk_hidden=[4], dropout=0.0)
    initial = state_arrays(ModularMLP(config, [SWELL], seed=0))

    def scaffold(rounds: int):
        def edge(i: int) -> EdgeSpec:
            data = SimpleNamespace(X=X[parts[i]], y=y[parts[i]])
            trainer = trainers.create("scaffold", {"lr": 0.05, "local_epochs": 1})
            model = ModularMLP(config, [SWELL], seed=0)
            return EdgeSpec(f"e{i}", model, data=data, trainer=trainer)

        return build_federation(
            TREE,
            {"fog_0": [edge(0), edge(1)], "fog_1": [edge(2)]},
            initial_state=initial,
            rounds=rounds,
            server_optimizer=server_optimizers.create("scaffold"),
        )

    straight = scaffold(4)
    straight.run()
    first = scaffold(2)
    first.run()
    save_bundle(
        tmp_path / "bundle",
        snapshot_federation(first),
        preprocessing={},
        schema={},
        lineage=LINEAGE,
        config={},
    )
    second = scaffold(4)
    restore_federation(second, load_bundle(tmp_path / "bundle").snapshot)
    second.run()

    assert any(k.startswith("scaffold/") for k in straight.coordinator.state)  # c
    for key, value in straight.coordinator.state.items():
        np.testing.assert_array_equal(second.coordinator.state[key], value, err_msg=key)
    for edge_id, edge in straight.edges.items():
        other = second.edges[edge_id]
        assert edge.trainer._c_i.keys() == other.trainer._c_i.keys() != set()
        for key, value in edge.trainer._c_i.items():
            np.testing.assert_array_equal(other.trainer._c_i[key], value, err_msg=key)
        theirs = state_arrays(other.model)
        for key, value in state_arrays(edge.model).items():
            np.testing.assert_array_equal(theirs[key], value, err_msg=edge_id + key)
    for node_id in [straight.coordinator.id, *straight.aggregators, *straight.edges]:
        assert rng_state(second.runtime._rng(node_id)) == rng_state(
            straight.runtime._rng(node_id)
        ), node_id
