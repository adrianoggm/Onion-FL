"""Tests for the virtual-clock simulation runtime (issue #81).

The nodes here are protocol test doubles: they pass messages around and
learn nothing, as allowed for protocol tests by docs/RULES.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from onion_fl.core.codec import get_codec
from onion_fl.core.context import node_rng
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.runtime.devices import availability_models, compute_models
from onion_fl.runtime.network import LinkProfile
from onion_fl.runtime.sim import SimRuntime, SimulationError

FIXED = LinkProfile(latency_s=0.1)


class Pinger(Node):
    """Sends one message per entry of ``rounds`` at start and records replies."""

    def __init__(self, node_id: str, peer: str, rounds=(0,)) -> None:
        super().__init__(node_id)
        self.peer = peer
        self.rounds = rounds
        self.replies: list[tuple[float, Message]] = []

    def on_start(self, ctx) -> None:
        for r in self.rounds:
            ctx.send(Message(kind="update", src=self.id, dst=self.peer, round=r))

    def on_message(self, msg, ctx) -> None:
        self.replies.append((ctx.now(), msg))


class Echo(Node):
    """Replies to every message, optionally declaring compute work first."""

    def __init__(self, node_id: str, work: float = 0.0) -> None:
        super().__init__(node_id)
        self.work = work
        self.seen: list[tuple[float, Message]] = []

    def on_message(self, msg, ctx) -> None:
        self.seen.append((ctx.now(), msg))
        if self.work:
            ctx.compute(self.work)
        ctx.send(
            Message(kind="global_model", src=self.id, dst=msg.src, round=msg.round)
        )


def pair(profile=FIXED, *, echo=None, pinger=None, seed=0, **echo_kwargs):
    rt = SimRuntime(seed=seed)
    pinger = pinger or Pinger("edge_1", "fog_0")
    echo = echo or Echo("fog_0")
    rt.add_node(pinger)
    rt.add_node(echo, **echo_kwargs)
    rt.add_link("edge_1", "fog_0", profile)
    return rt, pinger, echo


def names(rt: SimRuntime) -> list[str]:
    return [e["name"] for e in rt.events]


# --- clock and delivery ------------------------------------------------------


def test_ping_pong_advances_the_virtual_clock() -> None:
    rt, pinger, echo = pair()
    rt.run()

    assert echo.seen[0][0] == pytest.approx(0.1)
    assert pinger.replies[0][0] == pytest.approx(0.2)
    assert rt.now == pytest.approx(0.2)


def test_on_start_runs_at_time_zero_in_insertion_order() -> None:
    order = []

    class Recorder(Node):
        def on_start(self, ctx) -> None:
            order.append((self.id, ctx.now()))

    rt = SimRuntime()
    for node_id in ("cloud", "fog_0", "edge_1"):
        rt.add_node(Recorder(node_id))
    rt.run()

    assert order == [("cloud", 0.0), ("fog_0", 0.0), ("edge_1", 0.0)]


def test_sent_messages_record_their_encoded_size() -> None:
    rt, pinger, _ = pair()
    rt.run()

    sent = next(
        e for e in rt.events if e["name"] == "message.sent" and e["node"] == "edge_1"
    )
    expected = Message(kind="update", src="edge_1", dst="fog_0", round=0)
    assert sent["value"] == get_codec("json").size(expected)
    assert sent["tags"] == {
        "dst": "fog_0",
        "kind": "update",
        "round": 0,
        "codec": "json",
        "msg_id": "edge_1>fog_0#1",
    }


def test_link_codec_changes_the_bytes_on_the_wire() -> None:
    class BigSender(Node):
        def on_start(self, ctx) -> None:
            state = {"w": np.arange(512, dtype=np.float32) / 7}
            ctx.send(
                Message(
                    kind="update",
                    src=self.id,
                    dst="fog_0",
                    payload=Payload(state=state),
                )
            )

    sizes = {}
    for codec in ("json", "npz"):
        rt = SimRuntime()
        rt.add_node(BigSender("edge_1"))
        rt.add_node(Echo("fog_0"))
        rt.add_link("edge_1", "fog_0", FIXED, codec=codec)
        rt.run()
        sizes[codec] = next(
            e["value"] for e in rt.events if e["name"] == "message.sent"
        )

    assert sizes["npz"] < sizes["json"]


def test_receivers_get_a_decoded_copy() -> None:
    sent = Message(kind="update", src="edge_1", dst="fog_0", round=0)
    rt, _, echo = pair()
    rt.run()

    received = echo.seen[0][1]
    assert received == sent
    assert received is not sent


def test_bandwidth_delays_large_messages() -> None:
    class BigSender(Node):
        def on_start(self, ctx) -> None:
            state = {"w": np.zeros(2_000, dtype=np.float32)}
            ctx.send(
                Message(
                    kind="update",
                    src=self.id,
                    dst="fog_0",
                    payload=Payload(state=state),
                )
            )

    rt = SimRuntime()
    echo = Echo("fog_0")
    rt.add_node(BigSender("edge_1"))
    rt.add_node(echo)
    rt.add_link(
        "edge_1",
        "fog_0",
        LinkProfile(latency_s=0.1, bandwidth_up_bps=8_000),
        codec="npz",
    )
    rt.run()

    size = next(e["value"] for e in rt.events if e["name"] == "message.sent")
    assert echo.seen[0][0] == pytest.approx(size * 8 / 8_000 + 0.1)


# --- timers ------------------------------------------------------------------


class Timed(Node):
    def __init__(self, node_id: str, script) -> None:
        super().__init__(node_id)
        self.script = script
        self.fired: list[tuple[str, float]] = []

    def on_start(self, ctx) -> None:
        self.script(ctx)

    def on_timer(self, name, ctx) -> None:
        self.fired.append((name, ctx.now()))


def run_timed(script) -> Timed:
    rt = SimRuntime()
    node = Timed("fog_0", script)
    rt.add_node(node)
    rt.run()
    return node


def test_timer_fires_after_its_delay() -> None:
    node = run_timed(lambda ctx: ctx.set_timer(2.5, "deadline"))

    assert node.fired == [("deadline", 2.5)]


def test_rearming_a_timer_replaces_it() -> None:
    def script(ctx) -> None:
        ctx.set_timer(1.0, "deadline")
        ctx.set_timer(3.0, "deadline")

    assert run_timed(script).fired == [("deadline", 3.0)]


def test_cancelled_timer_does_not_fire() -> None:
    def script(ctx) -> None:
        ctx.set_timer(1.0, "deadline")
        ctx.set_timer(2.0, "other")
        ctx.cancel_timer("deadline")

    assert run_timed(script).fired == [("other", 2.0)]


def test_negative_timer_delay_is_a_node_error() -> None:
    rt = SimRuntime()
    rt.add_node(Timed("fog_0", lambda ctx: ctx.set_timer(-1.0, "x")))
    rt.run()

    assert names(rt) == ["node.error"]


# --- compute -----------------------------------------------------------------


def test_declared_compute_delays_the_reply() -> None:
    compute = compute_models.create("samples_per_second", {"samples_per_second": 500})
    rt, pinger, _ = pair(echo=Echo("fog_0", work=1000), compute=compute)
    rt.run()

    assert pinger.replies[0][0] == pytest.approx(0.1 + 2.0 + 0.1)


def test_messages_wait_while_the_node_is_busy() -> None:
    compute = compute_models.create("samples_per_second", {"samples_per_second": 500})
    pinger = Pinger("edge_1", "fog_0", rounds=(0, 1))
    rt, _, echo = pair(echo=Echo("fog_0", work=1000), pinger=pinger, compute=compute)
    rt.run()

    (t0, _), (t1, _) = echo.seen
    assert t0 == pytest.approx(0.1)
    assert t1 == pytest.approx(2.1)  # arrived at 0.1, handled when the first job ended
    assert [t for t, _ in pinger.replies] == pytest.approx([2.2, 4.2])


def test_measured_compute_uses_the_wall_time() -> None:
    compute = compute_models.create("measured", {"factor": 1e6})
    rt, pinger, _ = pair(compute=compute)
    rt.run()

    assert pinger.replies[0][0] > 0.2


# --- availability and losses -------------------------------------------------


def test_a_crashed_node_drops_what_it_receives() -> None:
    crash = availability_models.create("crash_at", {"t": 0.05})
    rt, pinger, echo = pair(availability=crash)
    rt.run()

    assert echo.seen == [] and pinger.replies == []
    assert "message.dropped_offline" in names(rt)


def test_bernoulli_offline_rounds_drop_only_round_messages() -> None:
    offline = availability_models.create("bernoulli", {"p": 1.0})
    pinger = Pinger("edge_1", "fog_0", rounds=(1, None))
    rt, _, echo = pair(pinger=pinger, availability=offline)
    rt.run()

    assert [msg.round for _, msg in echo.seen] == [None]


def test_an_offline_node_skips_its_timers() -> None:
    rt = SimRuntime()
    node = Timed("fog_0", lambda ctx: ctx.set_timer(1.5, "deadline"))
    rt.add_node(
        node,
        availability=availability_models.create("schedule", {"offline": [[1.0, 2.0]]}),
    )
    rt.run()

    assert node.fired == []
    assert names(rt) == ["timer.dropped_offline"]


def test_lossy_links_drop_and_record() -> None:
    rt, pinger, echo = pair(LinkProfile(latency_s=0.1, loss=1.0))
    rt.run()

    assert echo.seen == []
    assert names(rt).count("link.dropped") == 1


# --- errors ------------------------------------------------------------------


class Broken(Node):
    def on_message(self, msg, ctx) -> None:
        raise RuntimeError("boom")


def test_node_errors_do_not_stop_the_simulation() -> None:
    rt = SimRuntime()
    pinger = Pinger("edge_1", "fog_0")
    other = Pinger("edge_2", "fog_1")
    echo = Echo("fog_1")
    for node in (pinger, Broken("fog_0"), other, echo):
        rt.add_node(node)
    rt.add_link("edge_1", "fog_0", FIXED)
    rt.add_link("edge_2", "fog_1", FIXED)
    rt.run()

    errors = [e for e in rt.events if e["name"] == "node.error"]
    assert [e["node"] for e in errors] == ["fog_0"]
    assert "boom" in errors[0]["tags"]["error"]
    assert len(other.replies) == 1


@pytest.mark.parametrize(
    "msg",
    [
        Message(kind="update", src="edge_1", dst="nowhere"),
        Message(kind="update", src="someone_else", dst="fog_0"),
    ],
    ids=["no-link", "foreign-src"],
)
def test_invalid_sends_are_node_errors(msg: Message) -> None:
    class Sender(Node):
        def on_start(self, ctx) -> None:
            ctx.send(msg)

    rt = SimRuntime()
    rt.add_node(Sender("edge_1"))
    rt.add_node(Echo("fog_0"))
    rt.add_link("edge_1", "fog_0", FIXED)
    rt.run()

    assert names(rt) == ["node.error"]


def test_links_need_known_nodes_and_are_unique() -> None:
    rt = SimRuntime()
    rt.add_node(Echo("fog_0"))
    with pytest.raises(SimulationError, match="unknown node"):
        rt.add_link("edge_1", "fog_0", FIXED)
    rt.add_node(Echo("edge_1"))
    rt.add_link("edge_1", "fog_0", FIXED)
    with pytest.raises(SimulationError, match="already"):
        rt.add_link("fog_0", "edge_1", FIXED)
    with pytest.raises(SimulationError, match="already"):
        rt.add_node(Echo("fog_0"))


# --- determinism and run control ----------------------------------------------


def lossy_scenario(seed: int) -> list[dict]:
    profile = LinkProfile(
        latency_s=0.05, jitter_s=0.04, distribution="lognormal", loss=0.3
    )
    rt = SimRuntime(seed=seed)
    rt.add_node(Echo("fog_0"))
    for i in range(5):
        rt.add_node(Pinger(f"edge_{i}", "fog_0", rounds=range(4)))
        rt.add_link(f"edge_{i}", "fog_0", profile)
    rt.run()
    return rt.events


def test_same_seed_gives_identical_event_logs() -> None:
    assert lossy_scenario(11) == lossy_scenario(11)


def test_a_different_seed_changes_the_event_log() -> None:
    assert lossy_scenario(11) != lossy_scenario(12)


def test_run_until_stops_and_can_resume() -> None:
    rt, pinger, echo = pair()

    rt.run(until=0.15)
    assert len(echo.seen) == 1 and pinger.replies == []
    assert rt.now == pytest.approx(0.15)

    rt.run()
    assert len(pinger.replies) == 1


def test_max_events_stops_an_endless_run() -> None:
    class Forever(Node):
        def on_start(self, ctx) -> None:
            ctx.set_timer(1.0, "tick")

        def on_timer(self, name, ctx) -> None:
            ctx.set_timer(1.0, "tick")

    rt = SimRuntime(max_events=50)
    rt.add_node(Forever("cloud"))

    with pytest.raises(SimulationError, match="max_events"):
        rt.run()


# --- remaining paths ----------------------------------------------------------


def test_nodes_can_emit_their_own_events() -> None:
    class Talker(Node):
        def on_start(self, ctx) -> None:
            ctx.emit("edge.train_loss", 0.42, dataset="swell")

    rt = SimRuntime()
    rt.add_node(Talker("edge_1"))
    rt.run()

    assert rt.events == [
        {
            "t": 0.0,
            "node": "edge_1",
            "name": "edge.train_loss",
            "value": 0.42,
            "tags": {"dataset": "swell"},
        }
    ]


def test_negative_compute_is_a_node_error() -> None:
    rt = SimRuntime()
    rt.add_node(Timed("edge_1", lambda ctx: ctx.compute(-5)))
    rt.run()

    assert names(rt) == ["node.error"]


def test_timers_wait_while_the_node_is_busy() -> None:
    class BusyThenTimer(Timed):
        def on_start(self, ctx) -> None:
            ctx.set_timer(1.0, "deadline")
            ctx.compute(1500)  # 3 s of work at 500 samples/s

    rt = SimRuntime()
    node = BusyThenTimer("fog_0", script=None)
    rt.add_node(
        node,
        compute=compute_models.create(
            "samples_per_second", {"samples_per_second": 500}
        ),
    )
    rt.run()

    assert node.fired == [("deadline", 3.0)]


def test_a_node_that_crashes_while_computing_drops_its_pending_messages() -> None:
    compute = compute_models.create("samples_per_second", {"samples_per_second": 500})
    crash = availability_models.create("crash_at", {"t": 1.0})
    rt, pinger, _ = pair(
        echo=Echo("fog_0", work=1000), compute=compute, availability=crash
    )
    rt.run()

    assert pinger.replies == []
    dropped = [e for e in rt.events if e["name"] == "message.dropped_offline"]
    assert dropped[-1]["tags"] == {"pending": 1}


def test_messages_that_cannot_be_encoded_are_node_errors() -> None:
    class BadMeta(Node):
        def on_start(self, ctx) -> None:
            ctx.send(
                Message(
                    kind="control", src=self.id, dst="fog_0", meta={"when": object()}
                )
            )

    rt = SimRuntime()
    rt.add_node(BadMeta("edge_1"))
    rt.add_node(Echo("fog_0"))
    rt.add_link("edge_1", "fog_0", FIXED)
    rt.run()

    (error,) = rt.events
    assert error["name"] == "node.error"
    assert error["tags"]["handler"] == "send"


class Drawer(Node):
    """Draws from its rng in two separate handlers."""

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.draws: list[float] = []

    def on_start(self, ctx) -> None:
        self.draws.append(float(ctx.rng.random()))
        ctx.set_timer(1.0, "again")

    def on_timer(self, name, ctx) -> None:
        self.draws.append(float(ctx.rng.random()))


def test_a_nodes_rng_is_one_stream_across_handlers() -> None:
    sim = SimRuntime(seed=3)
    node = Drawer("n")
    sim.add_node(node)

    sim.run()

    assert node.draws[0] != node.draws[1]
    stream = node_rng(3, "n")
    assert node.draws == [float(stream.random()), float(stream.random())]


class Emitter(Node):
    def on_start(self, ctx) -> None:
        ctx.emit("hello.world", 1.0, colour="blue")
        ctx.set_timer(1.0, "again")

    def on_timer(self, name, ctx) -> None:
        ctx.emit("hello.again")


def test_listeners_receive_every_recorded_event() -> None:
    seen: list[dict] = []
    sim = SimRuntime(seed=0, listeners=[seen.append])
    sim.add_node(Emitter("n"))

    sim.run()

    assert seen == sim.events
    assert [e["name"] for e in seen] == ["hello.world", "hello.again"]


def test_each_message_gets_an_id_seen_at_both_ends() -> None:
    rt, _, _ = pair(pinger=Pinger("edge_1", "fog_0", rounds=(0, 1)))

    rt.run()

    sent = [e["tags"]["msg_id"] for e in rt.events if e["name"] == "message.sent"]
    delivered = [e["tags"]["msg_id"] for e in rt.events if e["name"] == "message.delivered"]
    assert len(set(sent)) == len(sent) == 4
    assert sorted(delivered) == sorted(sent)
