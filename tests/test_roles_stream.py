"""Edges fed by streams, idle rounds and paced rounds (continuum C3, issue #158).

Every edge uses the stub trainer (or a wrapper that records what it was
given), so nothing learns. The rows are a hand-written format fixture: the
tests check when rows arrive, are predicted, scored and trained on, not how
well (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np

from onion_fl.core.topology import parse_topology
from onion_fl.data.contract import SubjectData
from onion_fl.data.stream import LabelsConfig, StreamConfig, edge_stream
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation

A = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
INITIAL = state_arrays(ModularMLP(CONFIG, [A], seed=0))


def tree(n_fogs: int):
    fogs = [{"id": f"fog_{i}"} for i in range(n_fogs)]
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"nodes": fogs},
        }
    )


SPEED = 60.0
EVERY = 300.0  # data seconds between rounds: five one-minute rows


def rows(subject: str, n: int = 20) -> SubjectData:
    t = np.arange(n) * 60.0
    return SubjectData(
        X=np.stack([t / 600, np.cos(t)], axis=1).astype(np.float32),
        y=(np.arange(n) // 4) % 2,
        dataset="a",
        subject=subject,
        task="t",
        n_classes=2,
        feature_names=["f0", "f1"],
        t=t,
    )


def stream_of(subject: str, fraction: float = 1.0, delay: float = 0.0):
    config = StreamConfig(bootstrap=120, round_every=EVERY, batch_size=1, speed=SPEED)
    labels = LabelsConfig(fraction=fraction, delay=delay)
    return edge_stream(rows(subject), config, labels, seed=0)


class Recording:
    """The stub trainer, keeping the continuum time and rows of every call."""

    def __init__(self) -> None:
        self.stub = trainers.create("stub", {"shift": 1.0})
        self.calls: list[tuple[float, np.ndarray]] = []

    def train(self, model, data=None, received=None, ctx=None):
        self.calls.append((ctx.now() * SPEED, np.asarray(data.t).copy()))
        return self.stub.train(model, data, received, ctx)


def streaming(rounds: int = 6, **streams) -> tuple:
    trainers_by_edge = {name: Recording() for name in streams}
    specs = {
        name: EdgeSpec(
            name,
            ModularMLP(CONFIG, [A], seed=0),
            trainer=trainers_by_edge[name],
            stream=stream,
        )
        for name, stream in streams.items()
    }
    names = sorted(specs)
    fogs = min(2, len(names))  # a fog without edges would never register
    federation = build_federation(
        tree(fogs),
        {f"fog_{i}": [specs[n] for n in names[i::fogs]] for i in range(fogs)},
        initial_state=INITIAL,
        rounds=rounds,
        round_every=EVERY / SPEED,
    )
    federation.run()
    return federation, trainers_by_edge


def events(federation, name: str, node: str | None = None) -> list[dict]:
    return [
        e
        for e in federation.runtime.events
        if e["name"] == name and (node is None or e["node"] == node)
    ]


def test_without_labels_no_edge_ever_trains_and_every_round_is_idle() -> None:
    federation, recorded = streaming(
        a1=stream_of("a-1", fraction=0.0), a2=stream_of("a-2", fraction=0.0)
    )

    assert all(not r.calls for r in recorded.values())
    assert len(events(federation, "round.idle", "cloud")) == 6
    assert not events(federation, "round.quorum_failed")
    for key, value in INITIAL.items():
        np.testing.assert_array_equal(federation.coordinator.state[key], value)


def test_no_row_trains_before_its_label_arrives() -> None:
    streams = {"a1": stream_of("a-1", delay=600), "a2": stream_of("a-2", delay=600)}
    federation, recorded = streaming(**streams)

    for name, recording in recorded.items():
        label_at = dict(zip(streams[name].data.t, streams[name].label_at, strict=True))
        assert recording.calls, name
        for now, t in recording.calls:
            assert all(label_at[x] <= now for x in t), (name, now)


def test_idle_children_are_left_out_of_the_quorum() -> None:
    federation, recorded = streaming(
        a1=stream_of("a-1"), a2=stream_of("a-2", fraction=0.0)
    )

    assert recorded["a1"].calls and not recorded["a2"].calls
    assert not events(federation, "round.quorum_failed")
    # only a1 trains: its stub moves every weight by 1 per round with data
    trained = len(recorded["a1"].calls)
    np.testing.assert_allclose(
        federation.coordinator.state["trunk.0.weight"],
        INITIAL["trunk.0.weight"] + trained,
        rtol=1e-6,
    )


def test_the_volume_of_each_round_is_in_the_events() -> None:
    federation, _ = streaming(a1=stream_of("a-1", fraction=0.5, delay=60))

    arrived = events(federation, "data.arrived", "a1")
    assert [e["tags"]["round"] for e in arrived] == [1, 2, 3, 4, 5, 6]
    assert sum(e["value"] for e in arrived) == 20  # every row, history included
    assert arrived[0]["tags"]["history"] == 2
    for e in arrived:
        assert e["tags"]["labelled"] + e["tags"]["unlabelled"] == e["value"]
    late = sum(e["value"] for e in events(federation, "data.labelled", "a1"))
    assert 0 < late <= 10  # half the rows are labelled, each a minute late


def test_arrivals_are_scored_with_the_model_served_when_they_came() -> None:
    federation, _ = streaming(a1=stream_of("a-1"))

    scored = [
        e
        for e in events(federation, "eval.accuracy", "a1")
        if e["tags"]["model"] == "prequential"
    ]
    # round 1 serves nothing yet; later rounds score the rows that came since
    assert [e["tags"]["round"] for e in scored] == [2, 3, 4, 5]
    # every row after t0 but the one that came with round 1, before any model
    assert sum(e["tags"]["samples"] for e in scored) == 17
    at_fog = [
        e
        for e in events(federation, "eval.accuracy", "fog_0")
        if e["tags"]["model"] == "prequential"
    ]
    assert at_fog and all(e["tags"]["source"] == "children" for e in at_fog)


def test_rounds_are_spaced_by_round_every() -> None:
    federation, _ = streaming(a1=stream_of("a-1"))

    starts = [e["t"] for e in events(federation, "round.started", "cloud")]
    assert len(starts) == 6
    np.testing.assert_allclose(np.diff(starts), EVERY / SPEED)


def test_a_stream_run_is_deterministic() -> None:
    def trace():
        federation, _ = streaming(
            a1=stream_of("a-1", fraction=0.5, delay=60), a2=stream_of("a-2")
        )
        return [
            (e["name"], e["node"], e["tags"].get("round"), e["value"])
            for e in federation.runtime.events
        ], federation.coordinator.state

    (first, state), (second, again) = trace(), trace()
    assert first == second
    for key, value in state.items():
        np.testing.assert_array_equal(again[key], value)
