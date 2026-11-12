"""Tests for the real runtime, the transports and the multi-process launcher (issue #99).

The edges use the stub trainer (docs/RULES.md). MQTT is mocked with a fake
paho client; the end-to-end test talks to a real broker and is skipped when
none is reachable (``ONIONFL_MQTT=host:port``, default ``localhost:1883``).
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from onion_fl.core.message import Message
from onion_fl.core.node import Node
from onion_fl.core.topology import parse_topology
from onion_fl.learning.model import (
    DataShape,
    ModularMLP,
    ModularMLPConfig,
    state_arrays,
)
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.runtime.real import RealRuntime
from onion_fl.transports import transports
from onion_fl.transports.memory import MemoryTransport

SHAPE = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
CONFIG = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)


def model() -> ModularMLP:
    return ModularMLP(CONFIG, [SHAPE], seed=0)


# --- memory transport ---------------------------------------------------------------------------


def test_the_memory_transport_delivers_to_subscribers_of_an_address() -> None:
    got: list[bytes] = []
    a, b = MemoryTransport("bus-1"), MemoryTransport("bus-1")
    b.on_receive("x/inbox", got.append)
    a.start()
    b.start()

    a.send("x/inbox", b"hello")
    a.send("y/inbox", b"nobody")

    assert got == [b"hello"]
    assert a.stats == {"sent": 2, "bytes_sent": 11, "received": 0, "bytes_received": 0}
    assert b.stats["received"] == 1


def test_memory_buses_are_separate() -> None:
    got: list[bytes] = []
    MemoryTransport("bus-a").on_receive("x", got.append)

    MemoryTransport("bus-b").send("x", b"lost")

    assert got == []


# --- MQTT transport (fake paho client) ------------------------------------------------------------


class FakeClient:
    instances: list[FakeClient] = []
    acknowledges = True  # whether the broker answers subscriptions

    def __init__(self, *args, **kwargs) -> None:
        self.kwargs = kwargs
        self.subscribed: list[tuple[str, int]] = []
        self.published: list[tuple[str, bytes, int]] = []
        self.reconnect = None
        self.connected_to = None
        self.pending: list[int] = []
        FakeClient.instances.append(self)

    def reconnect_delay_set(self, min_delay, max_delay):
        self.reconnect = (min_delay, max_delay)

    def connect_async(self, host, port, keepalive):
        self.connected_to = (host, port, keepalive)

    def loop_start(self):
        self.on_connect(self, None, None, 0, None)
        if self.acknowledges:
            self.acknowledge()

    def acknowledge(self) -> None:
        """The broker answers every pending subscription."""
        while self.pending:
            self.on_subscribe(self, None, self.pending.pop(0), [0], None)

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))
        self.pending.append(len(self.subscribed))
        return 0, len(self.subscribed)

    def publish(self, topic, payload, qos):
        self.published.append((topic, payload, qos))

    def deliver(self, topic: str, payload: bytes) -> None:
        message = type("M", (), {"topic": topic, "payload": payload})()
        self.on_message(self, None, message)


@pytest.fixture
def fake_paho(monkeypatch):
    import paho.mqtt.client as mqtt

    FakeClient.instances = []
    monkeypatch.setattr(FakeClient, "acknowledges", True)
    monkeypatch.setattr(mqtt, "Client", FakeClient)
    return FakeClient


def test_mqtt_publishes_and_subscribes_with_the_link_qos(fake_paho) -> None:
    got: list[bytes] = []
    transport = transports.create("mqtt", {"broker": "broker.local:1884", "qos": 2})
    transport.on_receive("onionfl/r/fog/inbox", got.append)
    transport.start()
    listener, publisher = fake_paho.instances  # one connection each way (#192)

    transport.send("onionfl/r/cloud/inbox", b"payload")
    listener.deliver("onionfl/r/fog/inbox", b"reply")

    for client in (listener, publisher):
        assert client.connected_to[:2] == ("broker.local", 1884)
    assert listener.subscribed == [("onionfl/r/fog/inbox", 2)]
    assert listener.published == [] and publisher.subscribed == []
    assert publisher.published == [("onionfl/r/cloud/inbox", b"payload", 2)]
    assert got == [b"reply"]
    assert transport.stats["bytes_sent"] == 7 and transport.stats["received"] == 1


def test_mqtt_resubscribes_after_a_reconnect(fake_paho) -> None:
    transport = transports.create("mqtt", {"reconnect_min": 2, "reconnect_max": 60})
    transport.on_receive("a/inbox", lambda data: None)
    transport.start()
    client = fake_paho.instances[0]

    client.on_connect(client, None, None, 0, None)  # the broker came back

    assert client.reconnect == (2, 60)
    assert client.subscribed == [("a/inbox", 1), ("a/inbox", 1)]


def test_mqtt_starts_only_once_every_inbox_is_subscribed(fake_paho) -> None:
    """Otherwise a node could send to one it hosts before that one listens."""
    fake_paho.acknowledges = False
    transport = transports.create("mqtt", {"connect_timeout": 0.05})
    transport.on_receive("a/inbox", lambda data: None)
    with pytest.raises(ConnectionError):
        transport.start()

    fake_paho.instances[0].acknowledge()
    assert transport._listening.is_set()


def test_mqtt_turns_nagle_off_on_both_connections(fake_paho) -> None:
    """Every socket paho opens, reconnections included, gets TCP_NODELAY (#198)."""
    transports.create("mqtt", {}).start()

    assert len(fake_paho.instances) == 2
    for client in fake_paho.instances:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            client.on_socket_open(client, None, sock)
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)


def test_mqtt_keeps_nagle_when_the_link_asks(fake_paho) -> None:
    transports.create("mqtt", {"nodelay": False}).start()

    assert all(not hasattr(c, "on_socket_open") for c in fake_paho.instances)


def test_the_launcher_can_set_nagle_on_every_mqtt_link() -> None:
    """As it sets the broker: the emulated backend decides both for its lab."""
    rt = RealRuntime("run-nd", broker="10.0.0.3:1883", nodelay=False)
    plain = RealRuntime("run-nd")

    forced = rt._transport({"name": "mqtt", "qos": 1})
    assert forced.params.nodelay is False and forced.params.broker == "10.0.0.3:1883"
    assert plain._transport("mqtt").params.nodelay is True
    assert rt._transport("memory") is not forced


@pytest.mark.parametrize("params", [{"qos": 3}, {"broker": "no-port:x"}])
def test_mqtt_params_are_validated(params) -> None:
    (field,) = params
    with pytest.raises(ValueError, match=rf"\n{field}\n"):
        transports.create("mqtt", params)


# --- the real runtime in one process -----------------------------------------------------------------


class Echo(Node):
    stopped = True  # passive: it never holds the run open

    def on_message(self, msg, ctx):
        ctx.send(Message(kind="control", src=self.id, dst=msg.src, meta={"echo": True}))


class Ping(Node):
    """Pings at start and stops on its timer, after the echo is back."""

    stopped = False

    def on_start(self, ctx):
        ctx.send(Message(kind="control", src=self.id, dst="echo"))
        ctx.set_timer(0.2, "tick")

    def on_timer(self, name, ctx):
        ctx.emit("tick.fired", ctx.now())
        self.stopped = True


def test_messages_carry_their_id_and_measure_latency() -> None:
    rt = RealRuntime("run-1", seed=0, heartbeat_s=None)
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "latency"})

    rt.run(timeout=5)

    sent = [e for e in rt.events if e["name"] == "message.sent"]
    delivered = [e for e in rt.events if e["name"] == "message.delivered"]
    assert {e["tags"]["msg_id"] for e in sent} == {
        e["tags"]["msg_id"] for e in delivered
    }
    assert all(e["tags"]["latency"] >= 0 for e in delivered)
    assert any(e["name"] == "tick.fired" for e in rt.events)


def _ping_echo(name: str) -> RealRuntime:
    rt = RealRuntime(name, seed=0, heartbeat_s=None)
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": name})
    rt.run(timeout=5)
    return rt


def test_tracing_is_off_unless_the_environment_asks(monkeypatch) -> None:
    monkeypatch.delenv("ONIONFL_TRACE", raising=False)
    off = _ping_echo("trace-off")
    monkeypatch.setenv("ONIONFL_TRACE", "0")
    zero = _ping_echo("trace-zero")

    for rt in (off, zero):
        assert not [e for e in rt.events if e["name"].startswith("trace.")]


def test_tracing_times_each_message_through_the_process(monkeypatch) -> None:
    """``ONIONFL_TRACE=1`` adds one stamped event per message sent and handled (#198)."""
    monkeypatch.delenv("ONIONFL_TRACE", raising=False)
    plain = _ping_echo("trace-plain")
    monkeypatch.setenv("ONIONFL_TRACE", "1")
    traced = _ping_echo("trace-on")

    def stamps(name: str) -> dict[str, list[float]]:
        return {
            e["tags"]["msg_id"]: e["tags"]["wall"]
            for e in traced.events
            if e["name"] == name
        }

    sent = {e["tags"]["msg_id"] for e in traced.events if e["name"] == "message.sent"}
    on_send, on_delivery = stamps("trace.sent"), stamps("trace.delivered")
    assert set(on_send) == set(on_delivery) == sent and len(sent) == 2
    for wall in on_send.values():  # stamped, encoded, recorded, handed over
        assert len(wall) == 4 and wall == sorted(wall)
    for wall in on_delivery.values():  # received, popped, decoded, recorded, handled
        assert len(wall) == 5 and wall == sorted(wall)
    # Everything else is what the run records without tracing.
    names = [e["name"] for e in traced.events if not e["name"].startswith("trace.")]
    assert names == [e["name"] for e in plain.events]


def test_nodes_not_hosted_here_only_receive_messages() -> None:
    other = RealRuntime("run-2", hosted={"echo"}, heartbeat_s=None)
    here = RealRuntime("run-2", hosted={"ping"}, heartbeat_s=None)
    for rt in (other, here):
        rt.add_node(Ping("ping"))
        rt.add_node(Echo("echo"))
        rt.add_link("ping", "echo", transport={"name": "memory", "bus": "hosted"})
    other.start()

    here.run(timeout=5)
    other.stop()

    assert {e["node"] for e in here.events} == {"ping"}
    assert any(e["name"] == "message.delivered" for e in here.events)


def test_heartbeats_report_round_queue_and_cpu() -> None:
    rt = RealRuntime("run-3", heartbeat_s=0.01)
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "beats"})

    rt.run(timeout=0.3, stop=lambda: False)

    beats = [e for e in rt.events if e["name"] == "node.heartbeat"]
    assert beats and {"queue", "cpu_s"} <= set(beats[0]["tags"])


def test_every_node_sends_a_last_heartbeat_when_the_run_ends() -> None:
    rt = RealRuntime("run-4", heartbeat_s=10.0)  # longer than the run
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "last"})

    rt.run(timeout=5)

    beats = [e for e in rt.events if e["name"] == "node.heartbeat"]
    assert {e["node"] for e in beats} == {"ping", "echo"}
    assert all(e["tags"]["final"] for e in beats)


def test_a_whole_federation_runs_on_the_real_runtime() -> None:
    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {
                    "link_up": {"transport": {"name": "memory", "bus": "fed"}}
                },
                "nodes": [{"id": "fog_0"}],
            },
            "edge": {"link_up": {"transport": {"name": "memory", "bus": "fed"}}},
        }
    )
    edges = {
        "fog_0": [
            EdgeSpec(f"e{i}", model(), trainer=trainers.create("stub"))
            for i in range(3)
        ]
    }
    runtime = RealRuntime("run-fed", seed=0, heartbeat_s=None)
    federation = build_federation(
        topology, edges, initial_state=state_arrays(model()), rounds=2, runtime=runtime
    )

    runtime.run(timeout=10)

    assert federation.coordinator.finished
    assert all(
        node.stopped
        for node in [*federation.aggregators.values(), *federation.edges.values()]
    )
    expected = state_arrays(model())["trunk.0.weight"] + 2.0
    np.testing.assert_allclose(
        federation.coordinator.state["trunk.0.weight"], expected, rtol=1e-6
    )


# --- process groups ---------------------------------------------------------------------------------


def test_one_process_per_aggregator_and_one_for_the_root() -> None:
    from onion_fl.experiment.real import process_groups

    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"nodes": [{"id": "fog_a"}, {"id": "fog_b"}]},
        }
    )
    edges = {"fog_a": ["e1", "e2"], "fog_b": ["e3"], "cloud": ["test-1"]}

    groups = process_groups(topology, edges)

    assert groups == {
        "cloud": ["cloud", "test-1"],
        "fog_a": ["fog_a", "e1", "e2"],
        "fog_b": ["fog_b", "e3"],
    }


# --- end to end with a real broker ---------------------------------------------------------------------

BROKER = os.environ.get("ONIONFL_MQTT", "localhost:1883")


def _broker_up() -> bool:
    host, _, port = BROKER.partition(":")
    try:
        with socket.create_connection((host, int(port or 1883)), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")
def test_a_real_run_over_mqtt_with_one_process_per_group(tmp_path: Path) -> None:
    import textwrap

    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import run_experiment

    rows = ["pp,cond,f1"] + [
        f"{s},{'NT'[i % 2]},{s + i}" for s in range(1, 9) for i in range(4)
    ]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {(tmp_path / "raw").as_posix()}
            source: {{reader: csv, path: t.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    mqtt = {"name": "mqtt", "broker": BROKER}
    config = parse_experiment(
        {
            "name": "real_demo",
            "topology": {
                "name": "two_fogs",
                "levels": ["global", "fog", "edge"],
                "root": {"id": "cloud"},
                "fog": {
                    "defaults": {"link_up": {"transport": mqtt}},
                    "nodes": [{"id": "fog_a"}, {"id": "fog_b"}],
                },
                "edge": {"link_up": {"transport": mqtt}},
            },
            "data": {"datasets": {"demo": {}}, "roles": {"test": 0.25}},
            "learning": {
                "model": {
                    "name": "modular_mlp",
                    "adapter_width": 2,
                    "trunk_hidden": [2],
                },
                "trainer": "stub",
            },
            "rounds": 2,
            "evaluation": {"global": {"every": None}},
            "runtime": {"mode": "real", "timeout": 60, "heartbeat": 0.5},
            "paths": {
                "datasets": str(tmp_path / "datasets"),
                "runs": str(tmp_path / "runs"),
                "cache": str(tmp_path / "cache"),
            },
        }
    )

    (path,) = run_experiment(config)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    events = [
        json.loads(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {e["node"] for e in events} >= {"cloud", "fog_a", "fog_b"}
    assert any(e["name"] == "run.finished" for e in events)
    assert any(e["name"] == "node.heartbeat" for e in events)
    assert meta["processes"] == {"groups": 3, "exit_codes": [0, 0, 0]}
    assert sorted(p.name for p in path.iterdir()) == [
        "events.jsonl",
        "model.npz",
        "run.json",
        "scenario.json",
        "summary.json",
    ]


def test_every_group_can_be_started_by_hand_from_the_experiment(tmp_path: Path) -> None:
    """The multi-machine path: each machine runs ``onion_fl node --config exp.yaml``."""
    import textwrap
    import threading

    import yaml

    from onion_fl.experiment.real import run_node

    rows = ["pp,cond,f1"] + [
        f"{s},{'NT'[i % 2]},{s + i}" for s in range(1, 9) for i in range(4)
    ]
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {tmp_path.as_posix()}
            source: {{reader: csv, path: t.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    memory = {"transport": {"name": "memory", "bus": "by-hand"}}
    experiment = {
        "name": "by_hand",
        "topology": {
            "name": "two_fogs",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {"link_up": memory},
                "nodes": [{"id": "fog_a"}, {"id": "fog_b"}],
            },
            "edge": {"link_up": memory},
        },
        "data": {
            "datasets": {"demo": {"descriptor": str(tmp_path / "demo.yaml")}},
            "roles": {"test": 0.25},
        },
        "learning": {
            "model": {"name": "modular_mlp", "adapter_width": 2, "trunk_hidden": [2]},
            "trainer": "stub",
        },
        "rounds": 2,
        "evaluation": {"global": {"every": None}},
        "runtime": {"mode": "real", "timeout": 30, "heartbeat": None},
        "paths": {"runs": str(tmp_path / "runs"), "cache": str(tmp_path / "cache")},
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    from onion_fl.experiment.real import load_scenario
    from onion_fl.experiment.runner import load_data

    scenario, _ = load_scenario(tmp_path / "exp.yaml")
    load_data(scenario.config)  # fill the cache once, before the threads read it
    codes: dict[str, int] = {}

    def node(group: str) -> None:
        codes[group] = run_node(
            group, "manual-run", tmp_path / "exp.yaml", scenario="base", seed=0
        )

    threads = [
        threading.Thread(target=node, args=(g,)) for g in ("fog_a", "fog_b", "cloud")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert codes == {"fog_a": 0, "fog_b": 0, "cloud": 0}
    parts = sorted(p.name for p in (tmp_path / "runs" / "manual-run").iterdir())
    assert parts == [
        "events.cloud.jsonl",
        "events.fog_a.jsonl",
        "events.fog_b.jsonl",
        "model.npz",  # the root group keeps the final global model
    ]
    cloud = (tmp_path / "runs" / "manual-run" / "events.cloud.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"run.finished"' in cloud


def test_unknown_groups_and_scenarios_are_rejected(tmp_path: Path) -> None:
    from onion_fl.experiment.real import load_scenario

    with pytest.raises(ValueError, match="nope"):
        load_scenario(Path("experiments/mix_ab.yaml"), scenario="nope")


# --- robustness of real runs (QA3, #175) ---------------------------------------------


def test_a_message_delivered_twice_is_handled_once() -> None:
    # Two transport specs on one bus subscribe the echo's inbox twice.
    rt = RealRuntime("run-dup", heartbeat_s=None)
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_node(Echo("other"))
    rt.add_link("ping", "echo", transport="memory")
    rt.add_link("echo", "other", transport={"name": "memory", "bus": "default"})

    rt.run(timeout=5)

    handled = [
        e for e in rt.events if e["name"] == "message.delivered" and e["node"] == "echo"
    ]
    assert len(handled) == 1
    assert any(e["name"] == "message.duplicate" for e in rt.events)


# --- what a received message may name (QA1, #173) ----------------------------------


@pytest.mark.parametrize(
    "data",
    [
        b"os:abort\x00x",
        b"os:getcwd\x00x",
        b"json\x00[1, 2]",
        b"json\x00{bad",
        b"\xff\x00x",
    ],
    ids=["plugin_abort", "plugin_call", "not_a_message", "bad_json", "bad_name"],
)
def test_a_received_message_is_decoded_only_by_a_codec_of_the_links(
    data: bytes, monkeypatch
) -> None:
    import onion_fl.core.registry as registry

    def never(path: str):
        raise AssertionError(f"a received message made the runtime import {path!r}")

    monkeypatch.setattr(registry, "_import_spec", never)
    rt = RealRuntime("run-codec", heartbeat_s=None)
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "codec"})

    rt._on_deliver("echo", data)  # neither runs code nor raises

    (rejected,) = [e for e in rt.events if e["name"] == "message.rejected"]
    assert "undecodable" in rejected["tags"]["reason"]


# --- error isolation (#300) ----------------------------------------------------------


class Faulty(Echo):
    """Raises on every message it receives."""

    def on_message(self, msg, ctx):
        ctx.send(Message(kind="control", src=self.id, dst=msg.src))  # never leaves
        raise RuntimeError("boom")


class Unencodable(Ping):
    """Sends a message no codec can encode, then a good one."""

    def on_start(self, ctx):
        ctx.send(Message(kind="control", src=self.id, dst="echo", meta={"x": {1, 2}}))
        super().on_start(ctx)


def _errors(rt: RealRuntime) -> list[dict]:
    return [e for e in rt.events if e["name"] == "node.error"]


def test_a_node_that_raises_is_recorded_and_the_run_goes_on() -> None:
    rt = RealRuntime("run-raise", heartbeat_s=None)
    rt.add_node(Ping("ping"))
    rt.add_node(Faulty("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "raise"})

    rt.run(timeout=5)

    (error,) = _errors(rt)
    assert error["node"] == "echo"
    assert error["tags"] == {"handler": "on_message", "error": "RuntimeError: boom"}
    # What the failed handler sent is dropped; the other node runs to its end.
    assert [e["node"] for e in rt.events if e["name"] == "message.sent"] == ["ping"]
    assert any(e["name"] == "tick.fired" for e in rt.events)
    assert not any(e["name"] == "runtime.timeout" for e in rt.events)


def test_a_failing_after_handle_hook_keeps_the_run_going() -> None:
    rt = RealRuntime("run-hook", heartbeat_s=None)
    rt.add_node(Ping("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "hook"})

    def full_disk(node_id: str) -> None:
        raise OSError("no space left on device")

    rt.after_handle.append(full_disk)

    rt.run(timeout=5)

    errors = _errors(rt)
    assert errors and all(
        e["tags"] == {"handler": "after", "error": "OSError: no space left on device"}
        for e in errors
    )
    # Every handler still ran and what it sent still left: ping, then the echo.
    sent = [e["node"] for e in rt.events if e["name"] == "message.sent"]
    assert sent == ["ping", "echo"]
    assert len([e for e in rt.events if e["name"] == "message.delivered"]) == 2
    assert any(e["name"] == "tick.fired" for e in rt.events)


def test_a_message_that_cannot_be_encoded_is_recorded_and_not_sent() -> None:
    rt = RealRuntime("run-encode", heartbeat_s=None)
    rt.add_node(Unencodable("ping"))
    rt.add_node(Echo("echo"))
    rt.add_link("ping", "echo", transport={"name": "memory", "bus": "encode"})

    rt.run(timeout=5)

    (error,) = _errors(rt)
    assert error["node"] == "ping" and error["tags"]["handler"] == "send"
    assert error["tags"]["error"].startswith("MessageError: ")
    # The good message after it still leaves and is answered.
    sent = [e["node"] for e in rt.events if e["name"] == "message.sent"]
    assert sent == ["ping", "echo"]
    assert any(e["name"] == "tick.fired" for e in rt.events)
