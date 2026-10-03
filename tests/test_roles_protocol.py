"""Protocol tests for the roles on SimRuntime (issue #91).

Every edge uses the stub trainer, which adds a constant to each weight and
declares a number of examples, so the expected global model is plain
arithmetic. No data is involved (docs/RULES.md).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from onion_fl.core.message import Message, Payload
from onion_fl.core.registry import PluginError
from onion_fl.core.topology import parse_topology
from onion_fl.learning.aggregators import aggregators
from onion_fl.learning.attacks import attacks
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.roles.policies import (
    parse_duration,
    participations,
    quorum_needed,
    stale_weightings,
    stalenesses,
)
from onion_fl.runtime.devices import availability_models, compute_models

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
B = DataShape(dataset="b", task="t", n_features=3, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)


def model(*shapes: DataShape) -> ModularMLP:
    return ModularMLP(CONFIG, list(shapes), seed=0)


INITIAL = state_arrays(model(A, B))


def tree(n_fogs: int = 2, fog: dict | None = None, root: dict | None = None):
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"} | (root or {}),
            "fog": {
                "defaults": fog or {},
                "nodes": [{"id": f"fog_{i}"} for i in range(n_fogs)],
            },
        }
    )


def edge(
    node_id: str, shift: float = 1.0, examples: int = 10, shape=A, **kw
) -> EdgeSpec:
    trainer = trainers.create("stub", {"shift": shift, "examples": examples})
    return EdgeSpec(id=node_id, model=model(shape), trainer=trainer, **kw)


class Broken:
    def train(self, *args, **kwargs):
        raise RuntimeError("out of memory")


class AuxStub:
    """The stub trainer plus one auxiliary array: what it last received, plus 1."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub", {"shift": 1.0})
        self.seen: list[np.ndarray | None] = []

    def train(self, model, data=None, received=None, ctx=None):
        result = self.stub.train(model, data, received, ctx)
        last = (received or {}).get("algo/trunk.0.weight")
        self.seen.append(None if last is None else np.asarray(last).copy())
        base = np.zeros_like(received["trunk.0.weight"]) if last is None else last
        return dataclasses.replace(result, aux={"algo/trunk.0.weight": base + 1})


def run(topology, edges, rounds: int = 1, sharing: str = "fedavg", **kw):
    federation = build_federation(
        topology,
        edges,
        initial_state=INITIAL,
        rounds=rounds,
        sharing=sharing,
        **kw,
    )
    federation.run()
    return federation


def names(federation, name: str, node: str | None = None) -> list[dict]:
    return [
        e
        for e in federation.runtime.events
        if e["name"] == name and (node is None or e["node"] == node)
    ]


def shifted(key: str, by: float) -> np.ndarray:
    return INITIAL[key] + np.float32(by)


def assert_global(federation, key: str, by: float) -> None:
    np.testing.assert_allclose(
        federation.coordinator.state[key], shifted(key, by), rtol=1e-6, atol=1e-6
    )


# --- rounds and hierarchical weights ---------------------------------------------------


def test_one_round_moves_the_global_by_the_edges_shift() -> None:
    federation = run(
        tree(), {"fog_0": [edge("e1"), edge("e2")], "fog_1": [edge("e3"), edge("e4")]}
    )

    for key in ("trunk.0.weight", "adapter.a.0.weight", "head.t.bias"):
        assert_global(federation, key, 1.0)
    assert len(names(federation, "run.finished")) == 1


def test_hierarchical_weights_equal_a_flat_fedavg() -> None:
    edges = {
        "fog_0": [edge("e1", shift=1, examples=1), edge("e2", shift=3, examples=3)],
        "fog_1": [edge("e3", shift=0, examples=4)],
    }

    federation = run(tree(), edges)

    assert_global(federation, "trunk.0.weight", (1 * 1 + 3 * 3 + 0 * 4) / 8)


def test_rounds_accumulate() -> None:
    federation = run(tree(1), {"fog_0": [edge("e1"), edge("e2")]}, rounds=3)

    assert_global(federation, "trunk.0.weight", 3.0)
    edge_state = state_arrays(federation.edges["e1"].model)
    np.testing.assert_allclose(
        edge_state["trunk.0.weight"], shifted("trunk.0.weight", 3.0)
    )
    assert [e["value"] for e in names(federation, "round.started", "cloud")] == [
        1,
        2,
        3,
    ]


def test_each_key_is_weighted_only_by_the_edges_that_hold_it() -> None:
    edges = {
        "fog_0": [edge("ea", shift=1, examples=1, shape=A)],
        "fog_1": [edge("eb", shift=3, examples=3, shape=B)],
    }

    federation = run(tree(), edges)

    assert_global(federation, "adapter.a.0.weight", 1.0)
    assert_global(federation, "adapter.b.0.weight", 3.0)
    assert_global(federation, "trunk.0.weight", (1 + 9) / 4)


def test_train_metrics_travel_up_weighted_by_examples() -> None:
    federation = run(
        tree(1), {"fog_0": [edge("e1", examples=1), edge("e2", examples=3)]}
    )

    (loss,) = names(federation, "round.train_loss", "cloud")
    assert loss["value"] == 0.0 and loss["tags"]["examples"] == 4


# --- registration ---------------------------------------------------------------------------


def test_no_model_is_sent_before_the_tree_has_registered() -> None:
    federation = run(tree(), {"fog_0": [edge("e1")], "fog_1": [edge("e2")]})

    hellos = [
        e["t"]
        for e in names(federation, "message.delivered", "cloud")
        if e["tags"]["kind"] == "hello"
    ]
    first_model = min(
        e["t"]
        for e in names(federation, "message.sent", "cloud")
        if e["tags"]["kind"] == "global_model"
    )
    assert len(hellos) == 2 and first_model >= max(hellos)


def test_an_edge_that_never_comes_up_is_left_out_after_the_register_timeout() -> None:
    dead = availability_models.create("crash_at", {"t": 0.0})
    edges = {"fog_0": [edge("e1", shift=2), edge("e2", shift=10, availability=dead)]}

    federation = run(tree(1, fog={"register_timeout": 5}), edges)

    assert_global(federation, "trunk.0.weight", 2.0)
    registered = {
        e["tags"]["child"] for e in names(federation, "node.registered", "fog_0")
    }
    assert registered == {"e1"}


# --- quorum, deadline and failures ----------------------------------------------------------


def test_the_deadline_closes_a_round_with_quorum() -> None:
    edges = {
        "fog_0": [
            edge("e1", shift=2, examples=1),
            EdgeSpec("e2", model(A), trainer=Broken()),
        ],
        "fog_1": [edge("e3", shift=0, examples=1)],
    }

    federation = run(tree(fog={"quorum": 0.5, "deadline": "10s"}), edges)

    (closed,) = names(federation, "round.closed", "fog_0")
    assert closed["tags"]["responded"] == 1
    assert_global(federation, "trunk.0.weight", 1.0)
    (failed,) = names(federation, "edge.train_failed", "e2")
    assert "out of memory" in failed["tags"]["error"]


def test_without_quorum_an_empty_update_goes_up() -> None:
    edges = {
        "fog_0": [edge("e1", shift=2), EdgeSpec("e2", model(A), trainer=Broken())],
        "fog_1": [edge("e3", shift=5)],
    }

    federation = run(
        tree(fog={"quorum": 1.0, "deadline": 10}, root={"quorum": 0.5}), edges
    )

    assert len(names(federation, "round.quorum_failed", "fog_0")) == 1
    assert_global(federation, "trunk.0.weight", 5.0)


def test_a_failed_round_at_the_root_keeps_the_global_model() -> None:
    edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=Broken())]}

    federation = run(
        tree(1, fog={"deadline": 5}, root={"deadline": 20}), edges, rounds=2
    )

    assert len(names(federation, "round.quorum_failed", "cloud")) == 2
    assert_global(federation, "trunk.0.weight", 0.0)
    assert len(names(federation, "run.finished")) == 1


def test_a_count_quorum() -> None:
    assert quorum_needed(2, 5) == 2
    assert quorum_needed(9, 5) == 5
    assert quorum_needed(0.5, 5) == 3
    assert quorum_needed(1.0, 0) == 0


# --- staleness -----------------------------------------------------------------------------------


def slow(node_id: str, seconds: float = 100.0, **kw) -> EdgeSpec:
    # 10 examples trained in ``seconds``: past a 10 s deadline
    speed = {"samples_per_second": 10 / seconds}
    return edge(
        node_id, compute=compute_models.create("samples_per_second", speed), **kw
    )


def test_late_updates_are_dropped_by_default() -> None:
    edges = {"fog_0": [edge("fast", shift=1), slow("slow", shift=7)]}

    federation = run(tree(1, fog={"quorum": 0.5, "deadline": 10}), edges, rounds=2)

    late = names(federation, "update.late", "fog_0")
    assert late and all(e["tags"]["action"] == "drop" for e in late)
    assert_global(federation, "trunk.0.weight", 2.0)  # only the fast edge, twice


def test_next_round_staleness_keeps_late_updates_for_the_next_round() -> None:
    # 15 s of training: its round-1 update lands during round 2
    edges = {"fog_0": [edge("fast", shift=1), slow("slow", seconds=15, shift=7)]}
    fog = {
        "quorum": 0.5,
        "deadline": 10,
        "staleness": {
            "name": "next_round",
            "weighting": {"name": "constant", "factor": 0.5},
        },
    }

    federation = run(tree(1, fog=fog), edges, rounds=3)

    late = names(federation, "update.late", "fog_0")
    assert late and late[0]["tags"]["action"] == "buffered"
    assert any(
        e["tags"]["stale"] == 1 for e in names(federation, "round.closed", "fog_0")
    )


def test_stale_weightings() -> None:
    assert stale_weightings.create("constant", {"factor": 0.5}).factor(3) == 0.5
    assert stale_weightings.create("polynomial", {"a": 1.0}).factor(1) == 0.5
    assert stalenesses.create("drop").weight(1) is None
    weighting = {"name": "polynomial", "a": 1.0}
    assert stalenesses.create("next_round", {"weighting": weighting}).weight(3) == 0.25


# --- participation ------------------------------------------------------------------------------


def test_fraction_participation_picks_a_seeded_subset_each_round() -> None:
    edges = {"fog_0": [edge(f"e{i}") for i in range(4)]}

    federation = run(
        tree(1, fog={"participation": {"name": "fraction", "p": 0.5}}), edges, rounds=4
    )

    chosen = names(federation, "round.participants", "fog_0")
    assert [e["value"] for e in chosen] == [2, 2, 2, 2]
    picks = {tuple(e["tags"]["children"]) for e in chosen}
    assert len(picks) > 1


def test_participation_plugins() -> None:
    rng = np.random.default_rng(0)
    children = ["a", "b", "c", "d"]

    assert participations.create("all").select(children, 1, rng) == children
    picked = participations.create("fraction", {"p": 0.25}).select(children, 1, rng)
    assert len(picked) == 1 and picked[0] in children
    with pytest.raises(PluginError, match="p"):
        participations.create("fraction", {"p": 0})


# --- sharing across the hierarchy ---------------------------------------------------------------


def test_fedper_keeps_each_head_on_its_edge() -> None:
    edges = {"fog_0": [edge("e1", shift=1)], "fog_1": [edge("e2", shift=3)]}

    federation = run(tree(), edges, rounds=2, sharing="fedper")

    assert_global(federation, "head.t.weight", 0.0)
    assert_global(federation, "trunk.0.weight", 4.0)  # (1 + 3) / 2 per round
    e1 = state_arrays(federation.edges["e1"].model)
    np.testing.assert_allclose(e1["head.t.weight"], shifted("head.t.weight", 2.0))


def test_zone_heads_are_aggregated_and_kept_per_fog() -> None:
    edges = {
        "fog_0": [edge("e1", shift=1), edge("e2", shift=3)],
        "fog_1": [edge("e3", shift=3), edge("e4", shift=3)],
    }

    federation = run(tree(), edges, rounds=2, sharing={"name": "zone", "level": "fog"})

    zone_0 = federation.aggregators["fog_0"].zone["head.t.weight"]
    zone_1 = federation.aggregators["fog_1"].zone["head.t.weight"]
    np.testing.assert_allclose(zone_0, shifted("head.t.weight", 4.0))
    np.testing.assert_allclose(zone_1, shifted("head.t.weight", 6.0))
    # round 2 starts from the zone head (2), not from the edge's own head (1)
    e1 = state_arrays(federation.edges["e1"].model)
    np.testing.assert_allclose(e1["head.t.weight"], shifted("head.t.weight", 3.0))
    assert_global(federation, "head.t.weight", 0.0)
    assert_global(federation, "trunk.0.weight", 5.0)


class Recording:
    """The stub trainer, remembering which keys each round brought."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub")
        self.received: list[set[str]] = []

    def train(self, model, data=None, received=None, ctx=None):
        self.received.append(set(received or {}))
        return self.stub.train(model, data, received, ctx)


def test_round_one_sends_the_full_state_and_later_rounds_only_what_crosses() -> None:
    recorder = Recording()

    run(
        tree(1),
        {"fog_0": [EdgeSpec("e1", model(A), trainer=recorder)]},
        rounds=2,
        sharing="fedper",
    )

    first, second = recorder.received
    assert "head.t.weight" in first and "adapter.b.0.weight" in first
    assert not any(k.startswith("head.") for k in second)
    assert "trunk.0.weight" in second


def test_an_edge_sends_up_only_what_its_link_lets_through() -> None:
    federation = build_federation(
        tree(1),
        {"fog_0": [edge("e1")]},
        initial_state=INITIAL,
        rounds=1,
        sharing="fedper",
    )
    ctx = FakeContext("e1")

    federation.edges["e1"].on_message(
        Message(
            kind="global_model",
            src="fog_0",
            dst="e1",
            round=1,
            payload=Payload(state=dict(INITIAL)),
        ),
        ctx,
    )

    (sent,) = ctx.sent
    assert "trunk.0.weight" in sent.payload.state
    assert not any(k.startswith("head.") for k in sent.payload.state)
    assert set(sent.payload.weights) == set(sent.payload.state)


# --- errors -----------------------------------------------------------------------------------------


class FakeContext:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.rng = np.random.default_rng(0)
        self.sent: list[Message] = []
        self.events: list[tuple[str, dict]] = []

    def send(self, msg):
        self.sent.append(msg)

    def set_timer(self, delay, name):
        pass

    def cancel_timer(self, name):
        pass

    def now(self):
        return 0.0

    def emit(self, name, value=None, **tags):
        self.events.append((name, tags))

    def compute(self, samples):
        pass


def test_messages_from_strangers_are_rejected() -> None:
    federation = build_federation(
        tree(1), {"fog_0": [edge("e1")]}, initial_state=INITIAL, rounds=1
    )
    ctx = FakeContext("fog_0")

    federation.aggregators["fog_0"].on_message(
        Message(kind="update", src="intruder", dst="fog_0", round=1), ctx
    )
    federation.edges["e1"].on_message(
        Message(kind="global_model", src="cloud", dst="e1", round=1), ctx
    )

    rejected = [tags for name, tags in ctx.events if name == "message.rejected"]
    assert [t["src"] for t in rejected] == ["intruder", "cloud"]
    assert not ctx.sent


def test_an_update_nobody_asked_for_is_rejected() -> None:
    federation = build_federation(
        tree(1), {"fog_0": [edge("e1")]}, initial_state=INITIAL, rounds=1
    )
    ctx = FakeContext("fog_0")

    federation.aggregators["fog_0"].on_message(
        Message(kind="update", src="e1", dst="fog_0", round=4), ctx
    )

    assert [name for name, _ in ctx.events] == ["message.rejected"]


# --- determinism and decoupling ----------------------------------------------------------------------


def test_runs_are_deterministic() -> None:
    def once():
        edges = {"fog_0": [edge(f"e{i}") for i in range(3)], "fog_1": [slow("s")]}
        fog = {
            "participation": {"name": "fraction", "p": 0.7},
            "quorum": 0.5,
            "deadline": 10,
        }
        return run(tree(fog=fog), edges, rounds=3, seed=5)

    a, b = once(), once()

    assert a.runtime.events == b.runtime.events
    for key, value in a.coordinator.state.items():
        np.testing.assert_array_equal(value, b.coordinator.state[key])


def test_settings_reject_unknown_plugins() -> None:
    with pytest.raises(PluginError, match="tree_mean"):
        build_federation(
            tree(1, fog={"aggregator": "tree_mean"}),
            {"fog_0": [edge("e1")]},
            initial_state=INITIAL,
            rounds=1,
        )


def test_durations() -> None:
    assert parse_duration("30s") == 30.0
    assert parse_duration("500ms") == 0.5
    assert parse_duration("2m") == 120.0
    assert parse_duration(1.5) == 1.5
    assert parse_duration(None) is None
    with pytest.raises(ValueError, match="10 parsecs"):
        parse_duration("10 parsecs")


def test_no_dataset_names_in_the_generic_layers() -> None:
    pattern = re.compile(r"swell|sweet|wesad", re.IGNORECASE)
    offenders = [
        str(path)
        for package in ("roles", "learning", "runtime", "core")
        for path in Path("src/onion_fl", package).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


# --- edges of the protocol -------------------------------------------------------------------


def test_evaluators_register_but_never_train() -> None:
    evaluator = EdgeSpec("ev", model(A), train=False)

    federation = run(tree(1), {"fog_0": [edge("e1", shift=2), evaluator]})

    roles = {
        e["tags"]["child"]: e["tags"]["role"]
        for e in names(federation, "node.registered", "fog_0")
    }
    assert roles == {"e1": "edge", "ev": "evaluator"}
    (chosen,) = names(federation, "round.participants", "fog_0")
    assert chosen["tags"]["children"] == ["e1"]
    assert_global(federation, "trunk.0.weight", 2.0)


def test_a_fog_without_live_edges_is_left_out() -> None:
    dead = availability_models.create("crash_at", {"t": 0.0})
    edges = {
        "fog_0": [edge("e1", availability=dead)],
        "fog_1": [edge("e2", shift=4)],
    }

    federation = run(tree(fog={"register_timeout": 5}), edges)

    (chosen,) = names(federation, "round.participants", "cloud")
    assert chosen["tags"]["children"] == ["fog_1"]
    assert_global(federation, "trunk.0.weight", 4.0)


def test_a_round_with_nobody_to_ask_fails_at_once() -> None:
    dead = availability_models.create("crash_at", {"t": 0.0})

    federation = run(
        tree(1, fog={"register_timeout": 5}), {"fog_0": [edge("e1", availability=dead)]}
    )

    assert len(names(federation, "round.quorum_failed", "cloud")) == 1
    assert len(names(federation, "run.finished")) == 1


def test_edges_must_hang_from_leaf_aggregators() -> None:
    with pytest.raises(ValueError, match="cloud"):
        build_federation(
            tree(1), {"cloud": [edge("e1")]}, initial_state=INITIAL, rounds=1
        )


def test_a_sharing_policy_object_is_accepted() -> None:
    from onion_fl.learning.sharing import SharingPolicy

    policy = SharingPolicy(rules={"head.*": "local"})

    federation = run(tree(1), {"fog_0": [edge("e1", shift=1)]}, sharing=policy)

    assert_global(federation, "head.t.weight", 0.0)
    assert_global(federation, "trunk.0.weight", 1.0)


def test_fraction_of_nobody_is_nobody() -> None:
    picked = participations.create("fraction", {"p": 0.5}).select(
        [], 1, np.random.default_rng(0)
    )

    assert picked == []


# --- driving one aggregator by hand -------------------------------------------------------------


KEY = "trunk.0.weight"


def fog_by_hand(**settings):
    federation = build_federation(
        tree(1, fog=settings),
        {"fog_0": [edge("e1"), edge("e2")]},
        initial_state=INITIAL,
        rounds=3,
    )
    fog, ctx = federation.aggregators["fog_0"], FakeContext("fog_0")
    for child in ("e1", "e2"):
        fog.on_message(
            Message(
                kind="hello", src=child, dst="fog_0", meta={"role": "edge", "edges": 1}
            ),
            ctx,
        )
    return fog, ctx


def model_msg(round: int) -> Message:
    return Message(
        kind="global_model",
        src="cloud",
        dst="fog_0",
        round=round,
        payload=Payload(state={KEY: INITIAL[KEY]}),
        meta={"bootstrap": round == 1},
    )


def update_msg(src: str, round: int, value: float, weight: float) -> Message:
    state = {KEY: np.full_like(INITIAL[KEY], value)}
    return Message(
        kind="update",
        src=src,
        dst="fog_0",
        round=round,
        payload=Payload(state=state, weights={KEY: weight}),
    )


STALE = {
    "quorum": 0.5,
    "deadline": 10,
    "staleness": {
        "name": "next_round",
        "weighting": {"name": "constant", "factor": 0.5},
    },
}


def test_a_stale_update_joins_the_next_round_with_its_factor() -> None:
    fog, ctx = fog_by_hand(**STALE)
    fog.on_message(model_msg(1), ctx)
    fog.on_message(update_msg("e1", 1, 1.0, 1), ctx)
    fog.on_timer("deadline", ctx)
    fog.on_message(model_msg(2), ctx)

    fog.on_message(update_msg("e2", 1, 5.0, 2), ctx)  # late: weight 2 * 0.5
    fog.on_message(update_msg("e1", 2, 3.0, 1), ctx)
    fog.on_timer("deadline", ctx)

    sent = ctx.sent[-1]
    np.testing.assert_allclose(sent.payload.state[KEY], 4.0)  # (3 * 1 + 5 * 1) / 2
    assert sent.payload.weights[KEY] == 2.0


def test_a_fresh_update_replaces_a_stale_one_from_the_same_edge() -> None:
    fog, ctx = fog_by_hand(**STALE)
    fog.on_message(model_msg(1), ctx)
    fog.on_message(update_msg("e1", 1, 1.0, 1), ctx)
    fog.on_timer("deadline", ctx)
    fog.on_message(model_msg(2), ctx)

    fog.on_message(update_msg("e2", 1, 5.0, 2), ctx)
    fog.on_message(update_msg("e1", 2, 3.0, 1), ctx)
    fog.on_message(update_msg("e2", 2, 7.0, 2), ctx)  # everyone answered: closes now

    sent = ctx.sent[-1]
    np.testing.assert_allclose(sent.payload.state[KEY], (3 * 1 + 7 * 2) / 3, rtol=1e-6)
    assert sent.payload.weights[KEY] == 3.0


# --- registration over lossy links ------------------------------------------------------------


def lossy_tree(**settings):
    lossy = {"profile": {"preset": "lan", "loss": 0.5}}
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {"link_up": lossy, **settings},
                "nodes": [{"id": "fog_0"}],
            },
            "edge": {"link_up": lossy, **settings},
        }
    )


def test_hellos_are_repeated_until_the_parent_acknowledges_them() -> None:
    edges = {"fog_0": [edge(f"e{i}") for i in range(4)]}

    federation = run(lossy_tree(), edges, seed=1)

    assert any(
        e["name"] == "link.dropped" and e["tags"]["kind"] == "hello"
        for e in federation.runtime.events
    )
    assert len(names(federation, "federation.registered")) == 1
    acks = [
        e
        for e in names(federation, "message.sent", "fog_0")
        if e["tags"]["kind"] == "control"
    ]
    assert acks  # the fog answers every hello it gets


def test_without_retries_a_lost_hello_stalls_the_registration() -> None:
    edges = {"fog_0": [edge(f"e{i}") for i in range(4)]}

    federation = run(lossy_tree(hello_retry=None), edges, seed=1)

    assert names(federation, "federation.registered") == []


def test_auxiliary_arrays_go_up_and_come_back_down() -> None:
    trainer = AuxStub()
    edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=trainer)]}

    federation = run(tree(1), edges, rounds=3)

    assert trainer.seen[0] is None
    assert float(trainer.seen[1].mean()) == pytest.approx(1.0)
    assert float(trainer.seen[2].mean()) == pytest.approx(2.0)
    final = federation.coordinator.state["algo/trunk.0.weight"]
    assert float(final.mean()) == pytest.approx(3.0)


def test_auxiliary_arrays_do_not_reach_the_diagnostics() -> None:
    def diagnostics(trainer) -> list:
        other = trainers.create("stub", {"shift": 2.0})
        edges = {
            "fog_0": [
                EdgeSpec("e1", model(A), trainer=trainer),
                EdgeSpec("e2", model(A), trainer=other),
            ]
        }
        federation = run(tree(1), edges, rounds=2)
        return sorted(
            (e["node"], e["name"], e["tags"].get("round"), round(float(e["value"]), 9))
            for e in federation.runtime.events
            if e["name"].startswith(("diagnostic.divergence", "diagnostic.drift"))
        )

    with_aux = diagnostics(AuxStub())

    assert with_aux and with_aux == diagnostics(trainers.create("stub", {"shift": 1.0}))


class StatsRecorder:
    """A server optimizer that keeps the statistics of every round and replaces."""

    def __init__(self) -> None:
        self.stats: list[dict] = []

    def apply(self, global_state, aggregated, stats=None):
        self.stats.append(dict(stats or {}))
        return {**global_state, **aggregated}


def test_the_server_optimizer_gets_the_round_statistics() -> None:
    recorder = StatsRecorder()
    edges = {
        "fog_0": [edge("e1", shift=1, examples=1), edge("e2", shift=1, examples=3)],
        "fog_1": [edge("e3", shift=1, examples=4)],
    }

    run(tree(2), edges, server_optimizer=recorder)

    (stats,) = recorder.stats
    assert stats["train_edges"] == 3
    assert stats["edges_total"] == 3
    assert stats["train_examples"] == 8
    assert stats["train_steps"] == pytest.approx(1.0)  # the stub reports one step


class Exploding:
    """A trainer whose weights overflow: a diverged local optimisation."""

    def train(self, model, data=None, received=None, ctx=None):
        result = trainers.create("stub").train(model, data, received, ctx)
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
        return result


def test_an_edge_with_non_finite_weights_does_not_poison_the_model() -> None:
    edges = {
        "fog_0": [
            edge("e1", shift=2, examples=1),
            EdgeSpec("e2", model(A), trainer=Exploding()),
        ]
    }

    federation = run(tree(1, fog={"quorum": 0.5, "deadline": 10}), edges)

    state = federation.coordinator.state
    assert all(np.isfinite(v).all() for v in state.values())
    assert_global(federation, "trunk.0.weight", 2.0)
    (failed,) = names(federation, "edge.train_failed", "e2")
    assert "non-finite" in failed["tags"]["error"]


class Spy:
    """FedAvg that records what it is given and reports one dropped child."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict, object]] = []

    def aggregate(self, contributions, source, reference=None, rng=None):
        self.calls.append((dict(reference or {}), rng))
        return aggregators.create("fedavg").aggregate(contributions, source)

    def report(self):
        return [("diagnostic.selection", 1.0, {"dropped": ["e2"]})]


def test_aggregators_get_the_reference_and_a_stream_and_are_heard(
    monkeypatch,
) -> None:
    from onion_fl.roles import federation as module

    spy, original = Spy(), module._round_settings
    monkeypatch.setattr(
        module, "_round_settings", lambda raw: original(raw) | {"aggregator": spy}
    )
    malicious = EdgeSpec(
        "e2", model(A), trainer=trainers.create("stub"), tags={"malicious": True}
    )
    # A malicious edge whose update is discarded was not aggregated: not counted.
    diverged = EdgeSpec("e3", model(A), trainer=Exploding(), tags={"malicious": True})
    edges = {"fog_0": [edge("e1", shift=1), malicious, diverged]}

    federation = run(tree(1, fog={"quorum": 0.5, "deadline": 10}), edges)

    reference, rng = spy.calls[0]
    assert set(reference) >= set(INITIAL) and rng is not None
    (dropped,) = names(federation, "diagnostic.selection", "fog_0")
    assert dropped["tags"]["malicious_dropped"] == 1
    assert dropped["tags"]["malicious"] == 1


def test_a_malicious_edge_attacks_from_its_start_round() -> None:
    attack = attacks.create("scale", {"factor": 3.0, "start_round": 2})
    stub = trainers.create("stub", {"shift": 1.0})
    edges = {"fog_0": [EdgeSpec("e1", model(A), trainer=stub, attack=attack)]}

    federation = run(tree(1), edges, rounds=2)

    assert_global(federation, "trunk.0.weight", 1.0 + 3.0)  # honest, then ×3


def test_an_edge_with_local_dp_reports_its_epsilon_each_round() -> None:
    from onion_fl.learning.privacy import privacies

    dp = privacies.create("local_dp", {"clip": 10.0, "sigma": 1.0})
    spec = EdgeSpec("e1", model(A), trainer=trainers.create("stub"), privacy=dp)

    federation = run(tree(1), {"fog_0": [spec]}, rounds=2)

    values = [e["value"] for e in names(federation, "diagnostic.privacy_epsilon", "e1")]
    assert len(values) == 2 and values[0] < values[1]


def test_local_dp_clips_only_what_crosses_to_the_parent() -> None:
    from onion_fl.learning.privacy import privacies

    # The stub moves every weight by 1; a clip at the norm of the crossing keys
    # leaves them whole, unless the local head (sent down in round 1) counts too.
    crossing = [k for k in state_arrays(model(A)) if not k.startswith("head.")]
    clip = float(np.sqrt(sum(INITIAL[k].size for k in crossing)))
    dp = privacies.create("local_dp", {"clip": clip, "sigma": 1e-9})
    spec = edge("e1", shift=1, privacy=dp)

    federation = run(tree(1), {"fog_0": [spec]}, sharing="fedper")

    assert_global(federation, "trunk.0.weight", 1.0)


def test_the_statistics_count_the_holders_of_each_group() -> None:
    recorder = StatsRecorder()
    edges = {
        "fog_0": [edge("e1"), edge("e2")],
        "fog_1": [edge("e3", shape=B), EdgeSpec("e4", model(B), trainer=Broken())],
    }

    run(tree(2, fog={"quorum": 0.5, "deadline": 10}), edges, server_optimizer=recorder)

    (stats,) = recorder.stats
    assert stats["edges_total/adapter.a"] == 2 and stats["train_edges/adapter.a"] == 2
    assert stats["edges_total/adapter.b"] == 2 and stats["train_edges/adapter.b"] == 1
    assert stats["edges_total/trunk"] == 4 and stats["train_edges/trunk"] == 3


class OneVote(trainers.create("stub").__class__):
    """The stub trainer, asking for one vote per edge instead of one per example."""

    uniform_weights = True


def test_a_trainer_can_ask_for_one_vote_per_edge() -> None:
    def voter(node_id: str, shift: float, examples: int) -> EdgeSpec:
        trainer = OneVote(shift=shift, examples=examples)
        return EdgeSpec(node_id, model(A), trainer=trainer)

    edges = {
        "fog_0": [voter("e1", 1, 1), voter("e2", 4, 3)],
        "fog_1": [voter("e3", 4, 10)],
    }

    federation = run(tree(2), edges)

    assert_global(federation, "trunk.0.weight", 3.0)  # by examples: 53/14


class Locked(Exploding):
    """A valid plugin trainer that cannot be deep-copied (it holds a lock)."""

    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()


def test_a_trainer_that_cannot_be_copied_still_rolls_back_its_model() -> None:
    edges = {
        "fog_0": [
            edge("e1", shift=2, examples=1),
            EdgeSpec("e2", model(A), trainer=Locked()),
        ]
    }

    federation = run(tree(1, fog={"quorum": 0.5, "deadline": 10}), edges)

    assert_global(federation, "trunk.0.weight", 2.0)
    (failed,) = names(federation, "edge.train_failed", "e2")
    assert "non-finite" in failed["tags"]["error"]
