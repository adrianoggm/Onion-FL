"""Tests for fog autonomy when the link to the cloud drops (issue #289).

Protocol tests only: edges use the stub trainer (docs/RULES.md). The WAN
partition is a ``network.schedule`` outage of the link between a fog's site
and the cloud's, as the simulation models it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from onion_fl.core.message import Message, Payload
from onion_fl.core.topology import parse_topology
from onion_fl.experiment.config import ConfigError
from onion_fl.experiment.runner import _check_autonomy
from onion_fl.observability.run import _Summary
from onion_fl.observability.sinks import PrometheusSink
from onion_fl.roles import build_federation
from onion_fl.roles.policies import Autonomy
from tests.test_failover import LEAVES, events, federation_on_hosts
from tests.test_roles_protocol import INITIAL, KEY, edge, fog_by_hand, model_msg, tree

# fog_0's WAN is down from 15 s to 45 s: its round 2 update and the cloud's
# models of rounds 3 and 4 are lost; round 5's reaches it (rounds of ~10 s, the
# cloud closes at its 15 s deadline with fog_1 alone meanwhile).
OUTAGE = [
    {"between": ["fog_0", "dc"], "at": 15, "down": True},
    {"between": ["fog_0", "dc"], "at": 45, "down": False},
]
AUTONOMY = {"autonomy": {"after": "12s"}}  # longer than a round of the cloud


def run(fog: dict | None = None, root: dict | None = None, **kw: Any) -> Any:
    return federation_on_hosts(LEAVES, rounds=7, fog=fog, root=root, **kw)


def named(federation: Any, prefix: str) -> list[tuple[str, str, Any, dict]]:
    return [
        (e["node"], e["name"], e["value"], e["tags"])
        for e in federation.runtime.events
        if e["name"].startswith(prefix)
    ]


def closed(federation: Any, node: str) -> list[int]:
    return [
        e["tags"]["round"]
        for e in events(federation, "round.closed")
        if e["node"] == node
    ]


# --- off, and on without an outage ---------------------------------------------------


def test_without_autonomy_a_cut_fog_waits_for_the_cloud() -> None:
    federation = run(schedule=OUTAGE)

    assert not named(federation, "fog.")
    assert closed(federation, "fog_0") == [1, 2, 5, 6, 7]  # nothing while cut
    assert federation.coordinator.finished


def test_autonomy_with_a_parent_that_answers_changes_nothing() -> None:
    plain = run()
    autonomous = run(fog=AUTONOMY)

    assert autonomous.runtime.events == plain.runtime.events
    for key, value in plain.coordinator.state.items():
        np.testing.assert_array_equal(autonomous.coordinator.state[key], value)


# --- a WAN partition -----------------------------------------------------------------


def test_a_cut_fog_keeps_closing_rounds_and_the_cloud_reintegrates_them() -> None:
    federation = run(fog=AUTONOMY, schedule=OUTAGE)

    assert [(n, name, v) for n, name, v, _ in named(federation, "fog.autonomous")] == [
        ("fog_0", "fog.autonomous.started", 2.0),
        ("fog_0", "fog.autonomous.round", 1.0),
        ("fog_0", "fog.autonomous.round", 2.0),
        ("fog_0", "fog.autonomous.round", 3.0),
        ("fog_0", "fog.autonomous.ended", 3.0),
    ]
    # Its zone closed rounds 3 to 5 alone, then rejoined the cloud's.
    assert closed(federation, "fog_0") == [1, 2, 3, 4, 5, 6, 7]
    ended = named(federation, "fog.autonomous.ended")[0][3]
    assert ended == {"round": 5, "since": 2, "examples": 60.0, "reason": "parent"}
    (reconciled,) = named(federation, "fog.reconciled")
    assert reconciled[:3] == ("cloud", "fog.reconciled", 1.0)  # by samples
    assert reconciled[3] == {
        "src": "fog_0",
        "round": 5,
        "rounds": 3,
        "since": 2,
        "age": 3,
        "samples": 60.0,  # three rounds of two edges with 10 examples
        "action": "combined",
    }
    cloud_5 = [
        e["tags"]
        for e in events(federation, "round.closed")
        if e["node"] == "cloud" and e["tags"]["round"] == 5
    ]
    assert cloud_5[0]["responded"] == 2
    assert federation.coordinator.finished
    plain = run(schedule=OUTAGE).coordinator.state  # the zone's work is in
    assert any(
        not np.array_equal(value, plain[key])
        for key, value in federation.coordinator.state.items()
    )


def test_reconciling_by_staleness_takes_the_clouds_rule() -> None:
    polynomial = {"name": "next_round", "weighting": {"name": "polynomial", "a": 1.0}}
    fog = {"autonomy": {"after": "12s", "reconcile": "staleness"}}

    weighted = run(fog=fog, root={"staleness": polynomial}, schedule=OUTAGE)
    dropped = run(fog=fog, schedule=OUTAGE)  # the cloud drops late updates

    (kept,) = named(weighted, "fog.reconciled")
    assert kept[2] == pytest.approx(1 / (1 + 3))  # three rounds behind
    assert kept[3]["action"] == "combined"
    (lost,) = named(dropped, "fog.reconciled")
    assert (lost[2], lost[3]["action"]) == (0.0, "drop")
    cloud_5 = [
        e["tags"]["responded"]
        for e in events(dropped, "round.closed")
        if e["node"] == "cloud" and e["tags"]["round"] == 5
    ]
    assert cloud_5 == [1]


def test_autonomous_rounds_stop_at_max_rounds() -> None:
    federation = run(
        fog={"autonomy": {"after": "12s", "max_rounds": 1}}, schedule=OUTAGE
    )

    rounds = [v for _, name, v, _ in named(federation, "fog.autonomous.round")]
    assert rounds == [1.0]
    assert closed(federation, "fog_0") == [1, 2, 3, 6, 7]  # 5: the reconciliation
    assert named(federation, "fog.reconciled")[0][3]["rounds"] == 1


# --- the reconciliation, by hand ------------------------------------------------------


def update(src: str, round: int, value: float) -> Message:
    state = {KEY: np.full_like(INITIAL[KEY], value)}
    return Message(
        kind="update",
        src=src,
        dst="fog_0",
        round=round,
        payload=Payload(state=state, weights={KEY: 1.0}, metrics={}),
    )


def test_the_fog_sends_its_zones_change_applied_to_the_parents_model() -> None:
    fog, ctx = fog_by_hand(autonomy={"after": 5})
    fog.on_message(model_msg(1), ctx)
    fog.on_message(update("e1", 1, 1.0), ctx)
    fog.on_message(update("e2", 1, 3.0), ctx)  # round 1 closes at 2.0, sent up
    fog.on_timer("autonomy", ctx)  # the cloud sends nothing: round 2 on its own
    fog.on_message(update("e1", 2, 4.0), ctx)
    fog.on_message(update("e2", 2, 6.0), ctx)  # 5.0, kept: round 3 opens
    fog.on_message(model_msg(2, shift=10.0), ctx)  # back, while round 3 is open
    assert fog.autonomous is not None
    fog.on_message(update("e1", 3, 7.0), ctx)
    fog.on_message(update("e2", 3, 9.0), ctx)  # 8.0: the spell ends

    up = [m for m in ctx.sent if m.dst == "cloud" and m.kind == "update"]
    assert [m.round for m in up] == [1, 2]
    answer = up[-1]
    # The zone moved 6.0 from the 2.0 it started from, on top of round 2's model.
    np.testing.assert_allclose(answer.payload.state[KEY], INITIAL[KEY] + 10.0 + 6.0)
    assert answer.payload.weights[KEY] == 4.0  # two rounds of two edges
    assert answer.meta == {"autonomous": 2, "since": 1, "reconcile": "samples"}
    down = [m.round for m in ctx.sent if m.dst == "e1" and m.kind == "global_model"]
    assert down == [1, 2, 3]

    fog.on_message(model_msg(3), ctx)  # a round the zone already took: idle
    assert ctx.sent[-1].round == 3 and ctx.sent[-1].payload.metrics == {"idle": 1.0}
    fog.on_message(model_msg(4), ctx)  # a new one: the edges train it
    assert ctx.sent[-1].kind == "global_model" and ctx.sent[-1].round == 4
    names = [name for name, _ in ctx.events if name.startswith("fog.")]
    assert names == [
        "fog.autonomous.started",
        "fog.autonomous.round",
        "fog.autonomous.round",
        "fog.autonomous.ended",
        "fog.autonomous.ahead",
    ]


def test_a_stop_ends_the_spell_and_nothing_more_goes_up() -> None:
    fog, ctx = fog_by_hand(autonomy={"after": 5})
    fog.on_message(model_msg(1), ctx)
    fog.on_message(update("e1", 1, 1.0), ctx)
    fog.on_message(update("e2", 1, 3.0), ctx)
    fog.on_timer("autonomy", ctx)
    fog.on_message(
        Message(kind="control", src="cloud", dst="fog_0", meta={"stop": True}), ctx
    )
    fog.on_message(update("e1", 2, 4.0), ctx)
    fog.on_message(update("e2", 2, 6.0), ctx)

    up = [m.round for m in ctx.sent if m.dst == "cloud" and m.kind == "update"]
    assert up == [1]
    ended = [tags for name, tags in ctx.events if name == "fog.autonomous.ended"]
    assert [t["reason"] for t in ended] == ["stopped"]


# --- configuration -------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, message",
    [
        ({}, "after"),
        ({"after": 0}, "must be positive"),
        ({"after": "5s", "reconcile": "mean"}, "reconcile"),
        ({"after": "5s", "max_rounds": 0}, "max_rounds"),
        ({"after": "5s", "every": 1}, "every"),
    ],
)
def test_a_bad_autonomy_is_refused(spec: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Autonomy.parse(spec)


def test_autonomy_is_refused_at_the_root() -> None:
    topology = tree(1, root={"autonomy": {"after": "5s"}})
    with pytest.raises(ValueError, match="no parent"):
        build_federation(
            topology, {"fog_0": [edge("e1")]}, initial_state=INITIAL, rounds=1
        )
    with pytest.raises(ConfigError, match="no parent"):
        _check_autonomy(topology)


def test_the_config_names_the_fogs_with_autonomy() -> None:
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"nodes": [{"id": "a", "autonomy": {"after": "5s"}}, {"id": "b"}]},
        }
    )
    assert _check_autonomy(topology) == ["a"]
    with pytest.raises(ConfigError, match="topology: a: duration"):
        _check_autonomy(
            parse_topology(
                {
                    "levels": ["global", "fog", "edge"],
                    "root": {"id": "cloud"},
                    "fog": {"nodes": [{"id": "a", "autonomy": {"after": "never"}}]},
                }
            )
        )


# --- observability -------------------------------------------------------------------


def test_the_summary_and_prometheus_count_autonomous_rounds() -> None:
    federation = run(fog=AUTONOMY, schedule=OUTAGE)
    summary, sink = _Summary(), PrometheusSink()
    for raw in federation.runtime.events:
        level, role = (
            ("fog", "aggregator") if raw["node"].startswith("fog") else ("", "")
        )
        event = {
            "t_virtual": raw["t"],
            "run_id": "r",
            "topology_id": "t",
            "level": level,
            "node": raw["node"],
            "role": role,
            "kind": "lifecycle",
            "name": raw["name"],
            "value": raw.get("value"),
            "tags": raw.get("tags") or {},
        }
        summary.add(event)
        sink.write(event)

    assert summary.as_dict(0.0)["autonomous_rounds"] == {"fog_0": 3}
    labels = {
        "run_id": "r",
        "topology_id": "t",
        "level": "fog",
        "node": "fog_0",
        "dataset": "",
    }
    rounds = sink.registry.get_sample_value("onionfl_autonomous_rounds_total", labels)
    assert rounds == 3
    assert sink.registry.get_sample_value("onionfl_autonomous", labels) == 0  # ended
    assert "autonomous_rounds" not in _Summary().as_dict(0.0)
