"""Edges fed by streams, idle rounds and paced rounds (continuum C3, issue #158).

Every edge uses the stub trainer (or a wrapper that records what it was
given), so nothing learns. The rows are a hand-written format fixture: the
tests check when rows arrive, are predicted, scored and trained on, not how
well (docs/RULES.md).
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_the_memory_is_reported_each_round() -> None:
    federation, edge = replaying(Recording())

    reports = events(federation, "diagnostic.memory", "a1")
    assert [e["tags"]["round"] for e in reports] == list(range(1, 9))
    assert reports[-1]["value"] == len(edge.replay.rows())
    assert sum(reports[-1]["tags"]["classes"]) == reports[-1]["value"]
    assert reports[-1]["tags"]["capacity"] == 1000 and reports[-1]["tags"]["age"] > 0


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
