"""Edges fed by streams, idle rounds and paced rounds (continuum C3, issue #158).

Every edge uses the stub trainer (or a wrapper that records what it was
given), so nothing learns. The rows are a hand-written format fixture: the
tests check when rows arrive, are predicted, scored and trained on, not how
well (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.core.message import Message, Payload
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


def tree(n_fogs: int, fog: dict | None = None):
    fogs = [{"id": f"fog_{i}"} for i in range(n_fogs)]
    return parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"defaults": fog or {}, "nodes": fogs},
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


def test_a_stream_evaluator_scores_what_arrived_since_it_was_last_asked() -> None:
    learner = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=Recording(),
        stream=stream_of("a-1"),
    )
    validator = EdgeSpec(
        "val-a-9", ModularMLP(CONFIG, [A], seed=0), train=False, stream=stream_of("a-9")
    )
    federation = build_federation(
        tree(1, fog={"eval": {"every": 1}}),
        {"fog_0": [learner, validator]},
        initial_state=INITIAL,
        rounds=6,
        round_every=EVERY / SPEED,
    )
    federation.run()

    zone = [
        e
        for e in events(federation, "eval.accuracy", "fog_0")
        if e["tags"]["model"] == "zone"
    ]
    # disjoint windows: every row after t0 is scored once, history never
    assert sum(e["tags"]["samples"] for e in zone) == 18


def test_a_round_that_runs_late_skips_and_repeats_no_row() -> None:
    from onion_fl.runtime.devices import compute_models

    # 10 declared samples at 1 a second: each round trains for 10 virtual
    # seconds, twice round_every, so rounds open late and unevenly.
    slow = compute_models.create("samples_per_second", {"samples_per_second": 1.0})
    stream = stream_of("a-1", delay=60)
    recording = Recording()
    spec = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=recording,
        stream=stream,
        compute=slow,
    )
    federation = build_federation(
        tree(1),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=8,
        round_every=EVERY / SPEED,
    )
    federation.run()

    trained = np.concatenate([t for _, t in recording.calls])
    last = recording.calls[-1][0]
    due = stream.data.t[
        np.isfinite(stream.trainable_at) & (stream.trainable_at <= last)
    ]
    assert sorted(trained.tolist()) == sorted(due.tolist())  # each once, none skipped


class Broken:
    def train(self, *args, **kwargs):
        raise RuntimeError("out of memory")


def test_a_failed_round_keeps_the_scores_of_its_idle_children() -> None:
    idle = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=Recording(),
        stream=stream_of("a-1", fraction=0.0),
    )
    broken = EdgeSpec(
        "a2", ModularMLP(CONFIG, [A], seed=0), trainer=Broken(), stream=stream_of("a-2")
    )
    federation = build_federation(
        tree(1, fog={"deadline": 1}),
        {"fog_0": [idle, broken]},
        initial_state=INITIAL,
        rounds=4,
        round_every=EVERY / SPEED,
    )
    federation.run()

    assert events(federation, "round.quorum_failed", "fog_0")
    kept = [
        e
        for e in events(federation, "eval.accuracy", "fog_0")
        if e["tags"]["model"] == "prequential" and e["tags"]["source"] == "children"
    ]
    assert [e["tags"]["round"] for e in kept] == [2, 3, 4]


class Flaky(Recording):
    """Fails its first call, by raising or by leaving non-finite weights."""

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how, self.failed = how, None

    def train(self, model, data=None, received=None, ctx=None):
        if self.failed is not None:
            return super().train(model, data, received, ctx)
        self.failed = np.asarray(data.t).copy()
        if self.how == "raise":
            raise RuntimeError("out of memory")
        result = self.stub.train(model, data, received, ctx)
        import torch

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float("nan"))
        return result


@pytest.mark.parametrize("how", ["raise", "nan"])
def test_the_rows_of_a_failed_training_are_trained_next_time(how: str) -> None:
    flaky = Flaky(how)
    spec = EdgeSpec(
        "a1", ModularMLP(CONFIG, [A], seed=0), trainer=flaky, stream=stream_of("a-1")
    )
    federation = build_federation(
        tree(1, fog={"deadline": 1}),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=6,
        round_every=EVERY / SPEED,
    )
    federation.run()

    assert len(flaky.failed) and events(federation, "edge.train_failed", "a1")
    trained = np.concatenate([t for _, t in flaky.calls])  # the calls that worked
    assert set(flaky.failed) <= set(flaky.calls[0][1])  # retried at once
    assert len(trained) == len(set(trained))  # and still once each


class ChangesThenRaises(Recording):
    """Moves its local head, then fails, on its first call; records the head
    it starts each call from."""

    def __init__(self) -> None:
        super().__init__()
        self.heads: list[dict] = []

    def train(self, model, data=None, received=None, ctx=None):
        heads = {
            k: v.copy() for k, v in state_arrays(model).items() if k.startswith("head.")
        }
        self.heads.append(heads)
        if len(self.heads) == 1:
            import torch

            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.startswith("head."):
                        parameter.add_(100.0)
            raise RuntimeError("out of memory")
        return super().train(model, data, received, ctx)


def test_a_training_that_raises_leaves_no_trace_in_the_local_model() -> None:
    trainer = ChangesThenRaises()
    spec = EdgeSpec(
        "a1", ModularMLP(CONFIG, [A], seed=0), trainer=trainer, stream=stream_of("a-1")
    )
    federation = build_federation(
        tree(1, fog={"deadline": 1}),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=3,
        sharing="fedper",  # the head stays local: no broadcast overwrites it
        round_every=EVERY / SPEED,
    )
    federation.run()

    first, retry = trainer.heads[0], trainer.heads[1]
    for key, value in first.items():
        np.testing.assert_array_equal(retry[key], value, err_msg=key)


# --- replay memory (continuum C4) ---------------------------------------------------


def replaying(
    trainer,
    name: str = "fifo",
    ratio: float = 0.5,
    rounds: int = 8,
    before=lambda federation: None,
    **kw,
):
    from onion_fl.continuum.memory import memories

    replay = memories.create(name, {} if name == "none" else {"capacity": 1000})
    replay.rng = np.random.default_rng(7)
    spec = EdgeSpec(
        "a1",
        ModularMLP(CONFIG, [A], seed=0),
        trainer=trainer,
        stream=kw.pop("stream", stream_of("a-1")),
        replay=replay,
        replay_ratio=ratio,
    )
    federation = build_federation(
        tree(1, fog={"deadline": 1}),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=rounds,
        round_every=EVERY / SPEED,
    )
    before(federation)
    federation.run()
    return federation, federation.edges["a1"]


@pytest.mark.parametrize("how", ["raise", "nan"])
def test_a_row_enters_the_memory_only_once_its_training_succeeded(how: str) -> None:
    flaky = Flaky(how)
    federation, edge = replaying(flaky)

    consumed = np.flatnonzero(edge._consumed_by >= 0)
    assert sorted(edge.replay.rows().tolist()) == consumed.tolist()
    assert set(flaky.failed) <= set(edge.stream.data.t[edge.replay.rows()])


def test_each_training_replays_its_share_of_the_memory() -> None:
    recording = Recording()
    replaying(recording, ratio=0.5)

    seen: set[float] = set()
    assert len(recording.calls) > 2
    for _, t in recording.calls:
        recent = [x for x in t if x not in seen]
        replayed = len(t) - len(recent)
        # replay_ratio 0.5: as many replayed rows as recent ones, if kept
        assert recent and replayed == min(len(recent), len(seen))
        seen |= set(t.tolist())


class Weights:
    """A fog's aggregator, keeping the weight each child sent."""

    def __init__(self, inner) -> None:
        self.inner, self.sent = inner, []

    def aggregate(self, contributions, *args, **kw):
        self.sent += [max(c.weights.values()) for c in contributions]
        return self.inner.aggregate(contributions, *args, **kw)


def test_replayed_rows_add_nothing_to_the_aggregation_weight() -> None:
    recording, weights = Recording(), []

    def wrap(federation) -> None:
        fog = federation.aggregators["fog_0"]
        fog.aggregator = Weights(fog.aggregator)
        weights.append(fog.aggregator)

    federation, _ = replaying(recording, ratio=0.5, before=wrap)

    seen: set[float] = set()
    recent = []
    for _, t in recording.calls:
        recent.append(sum(1 for x in t if x not in seen))
        seen |= set(t.tolist())
    trained = [e["tags"]["examples"] for e in events(federation, "edge.trained", "a1")]
    assert any(len(t) > n for (_, t), n in zip(recording.calls, recent, strict=True))
    assert weights[0].sent == recent and trained == recent  # new rows only


def test_the_memory_is_reported_after_each_training_that_worked() -> None:
    federation, edge = replaying(Flaky("raise"))

    reports = events(federation, "diagnostic.memory", "a1")
    trained = [e["tags"]["round"] for e in events(federation, "edge.trained", "a1")]
    assert [e["tags"]["round"] for e in reports] == trained  # not the failed one
    assert reports[-1]["value"] == len(edge.replay.rows())  # the memory as it ends
    assert sum(reports[-1]["tags"]["classes"]) == reports[-1]["value"]
    assert reports[-1]["tags"]["capacity"] == 1000 and reports[-1]["tags"]["age"] > 0


def test_the_age_of_the_memory_counts_from_when_its_rows_were_observed() -> None:
    recording = Recording()
    federation, _ = replaying(recording)

    first = events(federation, "diagnostic.memory", "a1")[0]
    now, t = recording.calls[0]  # the first training: nothing replayed yet
    # t₀ is 120 s, so the history rows were observed 120 and 60 s before it
    assert first["tags"]["age"] == pytest.approx(float(np.mean(now - (t - 120.0))))


def test_a_replay_run_is_deterministic() -> None:
    def trace():
        federation, _ = replaying(Recording(), name="reservoir")
        return [
            (e["name"], e["node"], e["tags"].get("round"), e["value"])
            for e in federation.runtime.events
        ], federation.coordinator.state

    (first, state), (second, again) = trace(), trace()
    assert first == second
    for key, value in state.items():
        np.testing.assert_array_equal(again[key], value)


def test_without_replay_the_edge_trains_as_before() -> None:
    plain = Recording()
    federation = build_federation(
        tree(1),
        {
            "fog_0": [
                EdgeSpec(
                    "a1",
                    ModularMLP(CONFIG, [A], seed=0),
                    trainer=plain,
                    stream=stream_of("a-1"),
                )
            ]
        },
        initial_state=INITIAL,
        rounds=8,
        round_every=EVERY / SPEED,
    )
    federation.run()
    nothing = Recording()
    _, edge = replaying(nothing, name="none")

    assert [t.tolist() for _, t in nothing.calls] == [
        t.tolist() for _, t in plain.calls
    ]
    assert edge.replay.rows().tolist() == []


def test_the_snapshot_saves_the_replay_memory() -> None:
    from onion_fl.roles import snapshot_federation

    federation, edge = replaying(Recording(), name="reservoir")
    saved = snapshot_federation(federation).nodes["a1"]

    np.testing.assert_array_equal(saved.arrays["replay/rows"], edge.replay.rows())
    assert saved.meta["replay"]["seen"] == edge.replay.seen
    # what was consumed goes with it: a row is in memory or still unconsumed
    np.testing.assert_array_equal(saved.arrays["stream/consumed_by"], edge._consumed_by)


@pytest.mark.parametrize("edge_state", [True, False])
def test_the_memory_and_what_was_consumed_go_with_the_edge_state(
    edge_state: bool,
) -> None:
    from types import SimpleNamespace

    from onion_fl.experiment.runner import _restore_parts
    from onion_fl.roles import snapshot_federation

    federation, _ = replaying(Recording(), name="reservoir")
    restore = SimpleNamespace(model=True, server_state=True, edge_state=edge_state)

    kept = _restore_parts(snapshot_federation(federation), restore, {"a1"})

    saved = kept.nodes["a1"]
    parts = {k.split("/")[0] for k in saved.arrays}
    assert (
        {"replay", "stream"} <= parts
        if edge_state
        else not parts & {"replay", "stream"}
    )
    assert ("replay" in saved.meta) == edge_state


@pytest.mark.parametrize("ratio", [1.0, 1.5, -0.1])
def test_a_replay_ratio_outside_zero_to_one_is_refused(ratio: float) -> None:
    with pytest.raises(ValueError, match="replay_ratio"):
        replaying(Recording(), ratio=ratio, rounds=1)


def edge_with(replay, stream=None) -> EdgeSpec:
    model = ModularMLP(CONFIG, [A], seed=0)
    return EdgeSpec("a1", model, trainer=Recording(), stream=stream, replay=replay)


def test_a_replay_memory_needs_a_stream() -> None:
    from onion_fl.continuum.memory import memories

    memory = memories.create("fifo", {"capacity": 10})
    with pytest.raises(ValueError, match="stream"):
        build_federation(
            tree(1), {"fog_0": [edge_with(memory)]}, initial_state=INITIAL, rounds=1
        )


def test_a_memory_plugin_must_offer_what_the_edge_uses() -> None:
    class Partial:  # no capacity, rows, sample or state
        def add(self, rows, y) -> None:
            pass

    with pytest.raises(TypeError, match="replay memory"):
        build_federation(
            tree(1),
            {"fog_0": [edge_with(Partial(), stream_of("a-1"))]},
            initial_state=INITIAL,
            rounds=1,
        )


def test_an_edge_replays_the_configs_default_share() -> None:
    from onion_fl.experiment.config import ContinualConfig

    assert EdgeSpec("a1", None).replay_ratio == ContinualConfig().replay_ratio


# --- statuses and the local trigger (continuum C6) --------------------------------


class Hand:
    """A context driven by hand: what is sent and emitted is kept, and the time
    and the timers are set by the test."""

    def __init__(self) -> None:
        self.t = 0.0
        self.rng = np.random.default_rng(0)
        self.sent: list[Message] = []
        self.events: list[tuple[str, float, dict]] = []
        self.timers: dict[str, float] = {}

    def send(self, msg) -> None:
        self.sent.append(msg)

    def set_timer(self, delay, name) -> None:
        self.timers[name] = self.t + delay

    def cancel_timer(self, name) -> None:
        self.timers.pop(name, None)

    def now(self) -> float:
        return self.t

    def emit(self, name, value=None, **tags) -> None:
        self.events.append((name, value, tags))

    def compute(self, samples) -> None:
        pass

    def named(self, name: str) -> list[tuple[float, dict]]:
        return [(v, tags) for n, v, tags in self.events if n == name]


def edge_by_hand(stream, edge_trigger=None, drift=None, status_every=60.0):
    """A streaming edge of a continuous federation, started by hand."""
    from onion_fl.continuum.pace import Continuum

    pace = Continuum(
        trigger=None,
        edge_trigger=edge_trigger,
        status_every=status_every / SPEED,
        speed=SPEED,
        until=float("inf"),
        drift=drift or {},
    )
    recording = Recording()
    spec = EdgeSpec(
        "a1", ModularMLP(CONFIG, [A], seed=0), trainer=recording, stream=stream
    )
    federation = build_federation(
        tree(1),
        {"fog_0": [spec]},
        initial_state=INITIAL,
        rounds=1,
        continuum=pace,
    )
    edge, ctx = federation.edges["a1"], Hand()
    edge.on_start(ctx)
    return edge, ctx, recording


def tick(edge, ctx, until: float) -> None:
    """Fire the edge's status timer up to virtual time ``until``."""
    while ctx.timers.get("status", float("inf")) <= until:
        ctx.t = ctx.timers.pop("status")
        edge.on_timer("status", ctx)


def serve(edge, ctx, round: int) -> None:
    """The fog's model for ``round`` reaches the edge now."""
    msg = Message(
        kind="global_model",
        src="fog_0",
        dst="a1",
        round=round,
        payload=Payload(state=INITIAL),
        meta={"bootstrap": round == 1},
    )
    edge.on_message(msg, ctx)


def test_an_edge_reports_what_became_trainable_without_data() -> None:
    edge, ctx, _ = edge_by_hand(stream_of("a-1"))

    tick(edge, ctx, until=25 * 60 / SPEED)  # 25 minutes of data

    statuses = [m for m in ctx.sent if m.kind == "status"]
    assert len(statuses) == 25 and all(m.dst == "fog_0" for m in statuses)
    assert all(not m.payload.state for m in statuses)  # no data, only counts
    assert all(set(m.payload.metrics) == {"at", "round", "fresh"} for m in statuses)
    assert sum(m.payload.metrics["fresh"] for m in statuses) == 20  # every row


def labels_switch(subject: str, n: int = 40):
    """A stream whose label switches from 0 to 1 halfway: a prior shift."""
    t = np.arange(n) * 60.0
    data = SubjectData(
        X=np.stack([t / 600, np.cos(t)], axis=1).astype(np.float32),
        y=(np.arange(n) >= n // 2).astype(int),
        dataset="a",
        subject=subject,
        task="t",
        n_classes=2,
        feature_names=["f0", "f1"],
        t=t,
    )
    config = StreamConfig(bootstrap=600, batch_size=1, speed=SPEED)
    return edge_stream(data, config, LabelsConfig(fraction=1.0), seed=0)


def test_an_edge_detects_a_prior_shift_after_it_happens() -> None:
    edge, ctx, _ = edge_by_hand(labels_switch("a-1"), drift={"prior": "page_hinkley"})

    tick(edge, ctx, until=40 * 60 / SPEED)

    found = ctx.named("drift.detected")
    assert found and all(tags["kind"] == "prior" for _, tags in found)
    # never before the switch, 600 s of data after t0, when its first label comes
    assert all(tags["at"] >= 600 for _, tags in found)
    statuses = [m for m in ctx.sent if m.kind == "status"]
    detected = [m.payload.metrics["drift.prior.detected"] for m in statuses]
    assert sum(detected) == len(found)  # and its status says so


def bypassed(edge, recording) -> int:
    """Trainings the local trigger does not decide: v0, and after the stream."""
    return 1 + sum(1 for now, _ in recording.calls[1:] if now >= edge.stream.drain)


def test_an_edge_trains_only_when_its_local_trigger_fires() -> None:
    from onion_fl.continuum.triggers import triggers

    five = triggers.create("volume", {"samples": 5})
    edge, ctx, recording = edge_by_hand(stream_of("a-1"), edge_trigger=five)

    for round in range(1, 10):  # a round every two minutes of data
        ctx.t = round * 120 / SPEED
        serve(edge, ctx, round)

    updates = [m for m in ctx.sent if m.kind == "update"]
    idle = [m for m in updates if "idle" in m.payload.metrics]
    fired = ctx.named("trigger.fired")
    assert idle and len(recording.calls) == len(updates) - len(idle)
    # v0 and what is left once the stream ended train without the trigger
    assert len(fired) == len(recording.calls) - bypassed(edge, recording)
    assert all(value >= 5 for value, _ in fired)
    trained = np.concatenate([t for _, t in recording.calls])
    assert len(trained) == len(set(trained))  # nothing repeated


# --- rounds opened by triggers (continuum C6) -------------------------------------


def continuous(
    trigger,
    edge_trigger=None,
    status_every: float = 60.0,
    rounds: int = 1000,
    compute=None,
    before=lambda federation: None,
    runtime=None,
    **streams,
):
    """A continuous federation of stub edges, run to its end."""
    from onion_fl.continuum.pace import Continuum
    from onion_fl.continuum.triggers import drift_detectors, triggers

    def built(spec):
        if spec is None:
            return None
        name, params = (spec, {}) if isinstance(spec, str) else (spec["name"], spec)
        return triggers.create(name, {k: v for k, v in params.items() if k != "name"})

    federated, local = built(trigger), built(edge_trigger)
    pace = Continuum(
        trigger=federated,
        edge_trigger=local,
        status_every=status_every / SPEED,
        speed=SPEED,
        until=max(s.drain for s in streams.values()) / SPEED,
        drift=drift_detectors(federated, local),
    )
    trainers_by_edge = {name: Recording() for name in streams}
    specs = [
        EdgeSpec(
            name,
            ModularMLP(CONFIG, [A], seed=0),
            trainer=trainers_by_edge[name],
            stream=s,
            compute=compute,
        )
        for name, s in sorted(streams.items())
    ]
    fogs = min(2, len(specs))  # a fog without edges would never register
    federation = build_federation(
        tree(fogs),
        {f"fog_{i}": specs[i::fogs] for i in range(fogs)},
        initial_state=INITIAL,
        rounds=rounds,
        continuum=pace,
        runtime=runtime,
    )
    before(federation)
    federation.run()
    return federation, trainers_by_edge


def opened(federation) -> list[tuple[str, float]]:
    return [
        (e["tags"]["trigger"], e["tags"]["at"])
        for e in events(federation, "trigger.fired", "cloud")
    ]


def test_a_schedule_opens_a_round_every_period_of_data_time() -> None:
    every = {"name": "schedule", "every": 300}
    federation, _ = continuous(every, status_every=60, a1=stream_of("a-1"))

    names = [name for name, _ in opened(federation)]
    times = [at for _, at in opened(federation)]
    assert names[0] == "start" and names[-1] == "horizon"
    assert set(names[1:-1]) == {"schedule"}
    gaps = [b - a for a, b in zip(times[:-2], times[1:-1], strict=True)]
    assert gaps and all(g == pytest.approx(300.0) for g in gaps)
    started = events(federation, "round.started", "cloud")
    assert [e["tags"]["at"] for e in started] == times  # the cursor on each round


def test_a_schedule_reproduces_the_paced_rounds() -> None:
    every = {"name": "schedule", "every": EVERY}
    paced, _ = streaming(rounds=6, a1=stream_of("a-1"), a2=stream_of("a-2"))
    federation, _ = continuous(
        every, status_every=EVERY, a1=stream_of("a-1"), a2=stream_of("a-2")
    )

    for key, value in paced.coordinator.state.items():
        np.testing.assert_array_equal(federation.coordinator.state[key], value)


def test_a_volume_trigger_waits_for_its_rows() -> None:
    six = {"name": "volume", "samples": 6}
    federation, _ = continuous(six, a1=stream_of("a-1"), a2=stream_of("a-2"))

    fired = events(federation, "trigger.fired", "cloud")
    volume = [e for e in fired if e["tags"]["trigger"] == "volume"]
    assert volume and all(e["value"] >= 6 for e in volume)


def test_a_drift_trigger_opens_a_round_when_the_prior_shifts() -> None:
    prior = {"name": "drift", "kind": "prior"}
    federation, _ = continuous(prior, a1=labels_switch("a-1"), a2=labels_switch("a-2"))

    assert "drift/prior" in [name for name, _ in opened(federation)]
    zones = events(federation, "drift.detected", "fog_0")
    zones += events(federation, "drift.detected", "fog_1")
    assert zones  # the fogs detect it on their pooled statistic too


def test_statuses_carry_counts_and_statistics_only() -> None:
    received = []

    def spy(federation) -> None:
        cloud = federation.coordinator
        handle = cloud.on_message

        def on_message(msg, ctx) -> None:
            if msg.kind == "status":
                received.append(msg.payload)
            handle(msg, ctx)

        cloud.on_message = on_message

    continuous(
        {"name": "drift", "kind": "prior"},
        before=spy,
        a1=labels_switch("a-1"),
        a2=labels_switch("a-2"),
    )

    keys = {"at", "round", "fresh"}
    keys |= {"drift.prior.sum", "drift.prior.n", "drift.prior.detected"}
    assert received
    assert all(not p.state and set(p.metrics) <= keys for p in received)


def test_a_drift_trigger_without_labels_never_fires() -> None:
    config = StreamConfig(bootstrap=120, batch_size=1, speed=SPEED)
    unlabelled = edge_stream(rows("a-1"), config, LabelsConfig(fraction=0.0), seed=0)
    prior = {"name": "drift", "kind": "prior"}
    federation, _ = continuous(prior, a1=unlabelled)

    assert not events(federation, "drift.detected")
    assert [name for name, _ in opened(federation)] == ["start", "horizon"]


def test_a_trigger_that_never_fires_still_ends_at_the_last_label() -> None:
    never = {"name": "volume", "samples": 10_000}
    federation, _ = continuous(never, a1=stream_of("a-1"))

    assert [name for name, _ in opened(federation)] == ["start", "horizon"]
    (finished,) = events(federation, "run.finished")
    later = finished["t"] + 60 / SPEED  # one status period, for the stop to arrive
    sent = events(federation, "message.sent")
    assert not [e for e in sent if e["tags"]["kind"] == "status" and e["t"] > later]


def test_a_trigger_during_an_open_round_waits_for_it() -> None:
    from onion_fl.runtime.devices import compute_models

    slow = compute_models.create("samples_per_second", {"samples_per_second": 5})
    every = {"name": "schedule", "every": 60}
    federation, _ = continuous(
        every, status_every=60, compute=slow, a1=stream_of("a-1")
    )

    ends = ("round.closed", "round.idle", "round.quorum_failed")
    closed = {
        e["tags"]["round"]: e["t"]
        for e in federation.runtime.events
        if e["node"] == "cloud" and e["name"] in ends
    }
    started = events(federation, "round.started", "cloud")
    rounds = [e["tags"]["round"] for e in started]
    assert rounds == list(range(1, len(rounds) + 1)) and len(rounds) > 2
    for e in started[1:]:  # each opens once the previous one has closed
        assert e["t"] >= closed[e["tags"]["round"] - 1]


def test_an_edge_in_a_federation_trains_only_when_its_trigger_fires() -> None:
    federation, trainers_by_edge = continuous(
        {"name": "schedule", "every": 120},
        edge_trigger={"name": "volume", "samples": 5},
        a1=stream_of("a-1"),
    )

    calls = trainers_by_edge["a1"].calls
    fired = events(federation, "trigger.fired", "a1")
    edge = federation.edges["a1"]
    assert calls and len(fired) == len(calls) - bypassed(edge, trainers_by_edge["a1"])
    assert all(e["value"] >= 5 for e in fired)
    updates = [
        e
        for e in events(federation, "message.sent", "a1")
        if e["tags"]["kind"] == "update"
    ]
    assert len(updates) > len(calls)  # some rounds answered idle, rows kept
    trained = np.concatenate([t for _, t in calls])
    assert len(trained) == len(set(trained))  # nothing repeated


# --- fixes from the review of the branch ------------------------------------------


def test_a_lost_stop_does_not_keep_the_run_ticking(monkeypatch) -> None:
    from onion_fl.runtime.sim import SimRuntime

    lost = lambda node, ctx: None  # noqa: E731 - every stop is lost on its link
    monkeypatch.setattr("onion_fl.roles.nodes._stop_children", lost)
    every = {"name": "schedule", "every": 300}

    federation, _ = continuous(
        every,
        runtime=SimRuntime(max_events=50_000),
        a1=stream_of("a-1"),
        a2=stream_of("a-2"),
    )

    assert events(federation, "run.finished")  # and run() came back


def fog_by_hand():
    """A fog of a continuous federation with two registered edges, by hand."""
    from onion_fl.continuum.pace import Continuum

    pace = Continuum(None, None, 1.0, SPEED, float("inf"), {})
    specs = [
        EdgeSpec(
            n, ModularMLP(CONFIG, [A], seed=0), trainer=Recording(), stream=stream_of(n)
        )
        for n in ("e1", "e2")
    ]
    federation = build_federation(
        tree(1), {"fog_0": specs}, initial_state=INITIAL, rounds=5, continuum=pace
    )
    fog, ctx = federation.aggregators["fog_0"], Hand()
    fog.on_start(ctx)
    for child in ("e1", "e2"):
        meta = {"role": "edge", "edges": 1}
        fog.on_message(Message(kind="hello", src=child, dst="fog_0", meta=meta), ctx)
    return fog, ctx


def status(src: str, round: int, fresh: float) -> Message:
    metrics = {"at": 0.0, "round": float(round), "fresh": fresh}
    return Message(
        kind="status", src=src, dst="fog_0", payload=Payload(metrics=metrics)
    )


def test_a_fog_counts_only_what_its_children_saw_after_the_round() -> None:
    fog, ctx = fog_by_hand()
    fog.on_message(status("e1", 0, 4.0), ctx)  # heard before round 1 opened here
    model = Payload(state=INITIAL)
    meta = {"bootstrap": True}
    opened = Message(
        kind="global_model", src="cloud", dst="fog_0", round=1, payload=model, meta=meta
    )
    fog.on_message(opened, ctx)
    fog.on_message(status("e2", 0, 5.0), ctx)  # sent before e2 got round 1
    fog.on_message(status("e1", 1, 3.0), ctx)  # after: rows round 1 did not take

    ctx.t = 1.0
    fog.on_timer("status", ctx)

    (up,) = [m for m in ctx.sent if m.kind == "status"]
    assert up.payload.metrics["fresh"] == 3.0 and up.payload.metrics["round"] == 1.0


def test_an_edge_tags_its_status_with_the_round_it_last_got() -> None:
    edge, ctx, _ = edge_by_hand(stream_of("a-1"))
    tick(edge, ctx, until=2.0)
    ctx.t = 2.5
    serve(edge, ctx, 1)
    tick(edge, ctx, until=4.0)

    rounds = [m.payload.metrics["round"] for m in ctx.sent if m.kind == "status"]
    assert rounds == [0.0, 0.0, 1.0, 1.0]


def test_the_bootstrap_round_trains_whatever_the_local_trigger() -> None:
    from onion_fl.continuum.triggers import triggers

    never = triggers.create("volume", {"samples": 10_000})
    edge, ctx, recording = edge_by_hand(stream_of("a-1"), edge_trigger=never)

    serve(edge, ctx, 1)  # the first model: v0 trains on the history

    assert len(recording.calls) == 1 and not ctx.named("trigger.fired")


def test_an_edge_whose_stream_has_ended_trains_what_is_left() -> None:
    from onion_fl.continuum.triggers import triggers

    never = triggers.create("volume", {"samples": 10_000})
    edge, ctx, recording = edge_by_hand(stream_of("a-1"), edge_trigger=never)
    serve(edge, ctx, 1)
    ctx.t = 600 / SPEED
    serve(edge, ctx, 2)  # mid-stream: the trigger holds the rows back
    ctx.t = edge.stream.drain / SPEED
    serve(edge, ctx, 3)  # the last label is in: nothing more will come

    assert len(recording.calls) == 2
    assert (edge._consumed_by >= 0).all()  # every row trained once


@pytest.mark.parametrize("rounds, reason", [(3, "rounds"), (1000, "horizon")])
def test_a_run_says_whether_its_cap_or_its_last_label_ended_it(
    rounds: int, reason: str
) -> None:
    every = {"name": "schedule", "every": 60}
    federation, _ = continuous(every, rounds=rounds, a1=stream_of("a-1"))

    (finished,) = events(federation, "run.finished")
    assert finished["tags"]["reason"] == reason
