"""Tests for the dynamic continuum: host failures and edge failover (issue #190,
spec I1 §9).

Protocol tests only: edges use the stub trainer, and the experiment tests the
tiny CSV format fixture of ``test_experiment`` (docs/RULES.md).
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onion_fl.core.message import Message, Payload
from onion_fl.core.topology import parse_topology
from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import (
    _failover,
    _on_hosts,
    _on_machines,
    _paths,
    plan,
    run_scenario,
)
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.assignment import assignments, resolve
from onion_fl.infrastructure.failover import failovers
from onion_fl.infrastructure.model import InfrastructureError, parse_infrastructure
from onion_fl.infrastructure.network import downtime, host_events
from onion_fl.learning.trainers import trainers
from onion_fl.observability.events import event_fingerprint
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.runtime.sim import SimRuntime
from tests.test_experiment import TOPOLOGY, stub_score, workspace  # noqa: F401
from tests.test_experiment_infrastructure import with_infrastructure
from tests.test_infrastructure import lab
from tests.test_roles_protocol import INITIAL, A, FakeContext, model, tree

ZONE = {"name": "zone", "level": "fog"}

# --- model ---------------------------------------------------------------------------


def test_a_host_goes_down_and_comes_back_in_the_schedule() -> None:
    infra = parse_infrastructure(
        lab(
            network=lab()["network"]
            | {
                "schedule": [
                    {"host": "fog-1", "at": "10s", "down": True},
                    {"host": "fog-1", "at": "20s", "down": False},
                    {"host": "fog-2", "at": "5s", "down": True},
                ]
            }
        )
    )
    assert downtime(infra) == {"fog-1": [(10, 20)], "fog-2": [(5, math.inf)]}
    assert [(t, name, tags["host"]) for t, name, tags in host_events(infra)] == [
        (5, "host.down", "fog-2"),
        (10, "host.down", "fog-1"),
        (20, "host.up", "fog-1"),
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"host": "moon-1", "down": True}, "unknown host 'moon-1'"),
        ({"host": "fog-1", "set": {"latency": "1ms"}}, "takes down only"),
        ({"host": "fog-1", "between": ["fog", "dc"], "down": True}, "either"),
    ],
)
def test_a_bad_host_change_is_refused(change: dict, message: str) -> None:
    with pytest.raises(InfrastructureError, match=message):
        parse_infrastructure(lab(network=lab()["network"] | {"schedule": [change]}))


def test_a_link_change_keeps_its_identity() -> None:
    """``host`` is left out of a link change, so infrastructure_ids stay."""
    change = {"between": ["fog", "dc"], "at": 1.0, "down": True}
    infra = parse_infrastructure(lab(network=lab()["network"] | {"schedule": [change]}))
    (dumped,) = infra.model_dump(mode="json")["network"]["schedule"]
    assert "host" not in dumped


# --- runtime -------------------------------------------------------------------------


class Pinger:
    """Sets a timer at start; records when it fires."""

    def __init__(self, node_id: str) -> None:
        self.id, self.fired = node_id, []

    def on_start(self, ctx) -> None:
        ctx.set_timer(1.5, "tick")

    def on_timer(self, name, ctx) -> None:
        self.fired.append(ctx.now())

    def on_message(self, msg, ctx) -> None:
        pass


def test_a_node_waits_for_its_host_to_come_back() -> None:
    rt = SimRuntime()
    node = Pinger("n")
    rt.add_node(node)
    rt.add_host("h")
    rt.place("n", "h")
    rt.host_offline("h", 1.0, 4.0)
    rt.run()
    assert node.fired == [4.0]


# --- protocol ------------------------------------------------------------------------


class Recorder:
    """The stub trainer, remembering the head each training started from."""

    def __init__(self, shift: float) -> None:
        self.stub = trainers.create("stub", {"shift": shift, "examples": 10})
        self.heads: list[np.ndarray] = []

    def train(self, model, data=None, received=None, ctx=None):
        self.heads.append(np.asarray(received["head.t.weight"]).copy())
        return self.stub.train(model, data, received, ctx)


def federation_on_hosts(
    leaves: dict[str, list[str]],
    *,
    schedule: list[dict] = (),
    failover: Any = None,
    caps: dict[str, int] | None = None,
    rounds: int = 4,
) -> Any:
    """Fogs on their own sites, edges on the ``field``, 10 s of training a round.

    The field is closer to ``fog_1`` than to ``fog_2``.
    """
    caps = caps or {}
    fogs = sorted(leaves)
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", "deadline": "15s", "quorum": 0.5},
            "fog": {
                "defaults": {"deadline": "15s"},
                "nodes": [
                    {"id": f} | ({"max_children": caps[f]} if f in caps else {})
                    for f in fogs
                ],
            },
        }
    )
    edges = [e for f in fogs for e in leaves[f]]
    infra = parse_infrastructure(
        {
            "profiles": {"server": {}, "device": {"train_rate": 1.0}},
            "sites": {
                "dc": {"hosts": {"cloud": "server"}},
                "field": {"hosts": dict.fromkeys(edges, "device")},
            }
            | {f: {"hosts": {f: "server"}} for f in fogs},
            "network": {
                "links": [{"between": [f, "dc"], "latency": 0.01} for f in fogs]
                + [
                    {"between": ["field", f], "latency": 0.01 * (i + 1)}
                    for i, f in enumerate(fogs)
                ],
                "schedule": list(schedule),
            },
        }
    )
    resolved = resolve(
        infra,
        topology,
        root="cloud",
        aggregators={f: f for f in fogs},
        edges=assignments.create("explicit", {"hosts": {e: e for e in edges}}),
        leaves=leaves,
    )
    specs = {
        f: [
            EdgeSpec(id=e, model=model(A), trainer=Recorder(1.0 + 2 * i))
            for e in leaves[f]
        ]
        for i, f in enumerate(fogs)
    }
    specs, profiles = _on_hosts(specs, resolved)
    network, channels = _paths(resolved)
    federation = build_federation(
        topology,
        specs,
        initial_state=INITIAL,
        rounds=rounds,
        sharing=ZONE,
        seed=3,
        link_profiles=profiles,
        link_channels=channels,
    )
    _on_machines(federation.runtime, resolved, topology, network)
    if failover is not None:
        _failover(federation, resolved, topology, failover, network)
    federation.run()
    return federation


def events(federation: Any, name: str) -> list[dict]:
    return [e for e in federation.runtime.events if e["name"] == name]


def updates_at(federation: Any, fog: str, edge: str) -> list[int]:
    """Rounds of the updates of ``edge`` that reached ``fog``."""
    return [
        e["tags"]["round"]
        for e in events(federation, "message.delivered")
        if e["node"] == fog
        and e["tags"]["src"] == edge
        and e["tags"]["kind"] == "update"
    ]


DOWN_IN_ROUND_2 = [{"host": "fog_0", "at": 15, "down": True}]  # rounds of ~10 s
LEAVES = {"fog_0": ["e0", "e1"], "fog_1": ["e2", "e3"]}


def test_with_no_failover_a_down_fog_keeps_its_edges() -> None:
    """``none`` moves nothing: the run is the one without a failover."""
    plain = federation_on_hosts(LEAVES, schedule=DOWN_IN_ROUND_2)
    none = federation_on_hosts(
        LEAVES, schedule=DOWN_IN_ROUND_2, failover=failovers.create("none")
    )
    assert none.runtime.events == plain.runtime.events
    assert not events(none, "edge.reparented")
    assert [(e["t"], e["name"]) for e in events(none, "host.down")] == [
        (15, "host.down")
    ]
    assert none.coordinator.finished  # the other fog makes the quorum


def test_the_edges_of_a_fog_that_falls_train_the_next_round_under_another() -> None:
    federation = federation_on_hosts(
        LEAVES,
        schedule=DOWN_IN_ROUND_2,
        failover=failovers.create("nearest_alive", {"after": "1s"}),
    )
    moved = events(federation, "edge.reparented")
    assert [(e["node"], e["tags"]["previous"], e["tags"]["parent"]) for e in moved] == [
        ("e0", "fog_0", "fog_1"),
        ("e1", "fog_0", "fog_1"),
    ]
    assert all(e["tags"]["round"] == 2 for e in moved)
    # round 2's update went to the fallen fog and was lost; round 3 on, the new one
    assert updates_at(federation, "fog_0", "e0") == [1]
    assert updates_at(federation, "fog_1", "e0") == [3, 4]
    dropped = [
        e
        for e in events(federation, "message.dropped_offline")
        if e["node"] == "fog_0" and e["tags"]["src"] == "e0"
    ]
    assert [e["tags"]["round"] for e in dropped] == [2]
    # its first model from fog_1 carries fog_1's zone head, not fog_0's
    e0, e2 = (federation.edges[e].trainer.heads for e in ("e0", "e2"))
    assert not np.allclose(e0[1], e2[1])  # round 2: different zones
    np.testing.assert_array_equal(e0[2], e2[2])  # round 3: the same zone
    assert sorted(federation.aggregators["fog_1"].children) == ["e0", "e1", "e2", "e3"]
    assert federation.coordinator.finished
    assert (
        federation_on_hosts(
            LEAVES,
            schedule=DOWN_IN_ROUND_2,
            failover=failovers.create("nearest_alive", {"after": "1s"}),
        ).runtime.events
        == federation.runtime.events
    )  # the same seed, the same run


def test_no_edge_contributes_twice_to_a_round() -> None:
    federation = federation_on_hosts(
        LEAVES,
        schedule=DOWN_IN_ROUND_2,
        failover=failovers.create("nearest_alive", {"after": "1s"}),
    )
    taken = Counter(
        (e["tags"]["src"], e["tags"]["round"])
        for e in events(federation, "message.delivered")
        if e["tags"]["kind"] == "update" and e["tags"]["src"].startswith("e")
    )
    assert taken and max(taken.values()) == 1


def test_an_edge_refuses_a_round_it_took_under_its_old_parent() -> None:
    federation = build_federation(
        tree(2),
        {"fog_0": [EdgeSpec(id="e0", model=model(A), trainer=Recorder(1.0))]},
        initial_state=INITIAL,
        rounds=1,
    )
    edge, ctx = federation.edges["e0"], FakeContext("e0")
    payload = Payload(state=dict(INITIAL))
    for parent in ("fog_0", "fog_1"):  # round 3 from both
        edge.reparent(parent, ctx)
        edge.on_message(
            Message(
                kind="global_model", src=parent, dst="e0", round=3, payload=payload
            ),
            ctx,
        )
    trained = [m for m in ctx.sent if m.kind == "update"]
    assert [bool(m.payload.state) for m in trained] == [True, False]
    assert trained[1].payload.metrics == {"idle": 1.0}
    assert len(edge.trainer.heads) == 1


def test_a_full_fog_sends_the_edges_on_to_the_next() -> None:
    """fog_1 has room for one more: the second edge goes on to fog_2."""
    federation = federation_on_hosts(
        {"fog_0": ["e0", "e1"], "fog_1": ["e2"], "fog_2": ["e3"]},
        schedule=DOWN_IN_ROUND_2,
        failover=failovers.create("nearest_alive", {"after": "1s"}),
        caps={"fog_1": 2},
    )
    moved = {
        e["node"]: e["tags"]["parent"] for e in events(federation, "edge.reparented")
    }
    assert moved == {"e0": "fog_1", "e1": "fog_2"}
    assert federation.coordinator.finished


def test_the_edges_over_max_children_go_to_another_fog_at_the_start() -> None:
    federation = federation_on_hosts(
        {"fog_0": ["e0", "e1", "e2"], "fog_1": ["e3"]},
        failover=failovers.create("nearest_alive"),
        caps={"fog_0": 2},
        rounds=1,
    )
    (moved,) = events(federation, "edge.reparented")
    assert (moved["t"], moved["node"], moved["tags"]["parent"]) == (0, "e2", "fog_1")
    assert updates_at(federation, "fog_1", "e2") == [1]


def test_an_explicit_failover_follows_its_list() -> None:
    federation = federation_on_hosts(
        {"fog_0": ["e0"], "fog_1": ["e1"], "fog_2": ["e2"]},
        schedule=DOWN_IN_ROUND_2,
        failover=failovers.create(
            "explicit", {"after": 1, "parents": {"e*": ["fog_2", "fog_1"]}}
        ),
    )
    (moved,) = events(federation, "edge.reparented")
    assert (moved["node"], moved["tags"]["parent"]) == ("e0", "fog_2")


def test_a_fog_back_before_after_keeps_its_edges() -> None:
    federation = federation_on_hosts(
        LEAVES,
        schedule=[
            {"host": "fog_0", "at": 15, "down": True},
            {"host": "fog_0", "at": 15.5, "down": False},
        ],
        failover=failovers.create("nearest_alive", {"after": "1s"}),
    )
    assert not events(federation, "edge.reparented")
    assert [e["name"] for e in events(federation, "host.up")] == ["host.up"]


# --- experiment ----------------------------------------------------------------------


def test_failover_none_changes_nothing(workspace: Path) -> None:  # noqa: F811
    plain = parse_experiment(with_infrastructure(workspace))
    raw = with_infrastructure(workspace)
    raw["infrastructure"]["failover"] = "none"
    none = parse_experiment(raw)
    assert "failover" not in plain.dump()["infrastructure"]
    (a,), (b,) = scenarios(plain), scenarios(none)
    first = run_scenario(a, evaluate=stub_score) / "events.jsonl"
    second = run_scenario(b, evaluate=stub_score) / "events.jsonl"
    assert event_fingerprint(first) == event_fingerprint(second)


def with_cap(workspace: Path, failover: Any = None) -> dict:  # noqa: F811
    topology = TOPOLOGY | {
        "fog": {
            "nodes": [
                {"id": "fog_a", "home": "demo", "max_children": 1},
                {"id": "fog_b", "home": "demo"},
            ]
        }
    }
    raw = with_infrastructure(workspace, topology=topology)
    if failover is not None:
        raw["infrastructure"]["failover"] = failover
    return raw


def test_max_children_without_a_failover_is_refused_in_plan(
    workspace: Path,  # noqa: F811
) -> None:
    with pytest.raises(ConfigError, match="fog_a has .* edges but max_children 1"):
        plan(parse_experiment(with_cap(workspace)))


def test_a_run_sends_the_edges_over_max_children_on(
    workspace: Path,  # noqa: F811
) -> None:
    import json

    (scenario,) = scenarios(parse_experiment(with_cap(workspace, "nearest_alive")))
    path = run_scenario(scenario, evaluate=stub_score)
    assert json.loads((path / "run.json").read_text("utf-8"))["status"] == "finished"
    moved = [
        json.loads(line)
        for line in (path / "events.jsonl").read_text("utf-8").splitlines()
        if '"edge.reparented"' in line
    ]
    assert moved and {e["tags"]["parent"] for e in moved} == {"fog_b"}
