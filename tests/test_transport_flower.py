"""The Flower transport: each frame in a Flower ``Message`` (issue #204).

The links are the gRPC transport's (tested in ``test_transport_grpc.py``);
these tests check the envelope and that it travels. Every server listens on
127.0.0.1 with an ephemeral port, and every test has its own discovery
folder. Without ``flwr`` only the error that asks for the extra is tested.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from onion_fl.core.registry import PluginError
from onion_fl.transports import transports
from onion_fl.transports.flower import FlowerTransport, node_id


def test_it_is_a_registered_transport_even_without_flower() -> None:
    assert transports.get("flower").title == "Flower"


def test_without_flower_it_asks_for_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("flwr", "flwr.app"):  # importing them raises
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(PluginError, match=r"onion-fl\[flower\]"):
        FlowerTransport()


def test_an_inbox_has_one_stable_63_bit_node_id() -> None:
    a = node_id("onionfl/r/fog_a/inbox")

    assert a == node_id("onionfl/r/fog_a/inbox") != node_id("onionfl/r/fog_b/inbox")
    assert 0 <= a < 2**63


class Inbox:
    def __init__(self) -> None:
        self.items: list[bytes] = []
        self._cv = threading.Condition()

    def __call__(self, data: bytes) -> None:
        with self._cv:
            self.items.append(data)
            self._cv.notify_all()

    def wait(self, count: int, timeout: float = 10.0) -> list[bytes]:
        with self._cv:
            self._cv.wait_for(lambda: len(self.items) >= count, timeout)
            return list(self.items)


@pytest.fixture
def make(tmp_path: Path):
    pytest.importorskip("flwr", minversion="1.21")
    pytest.importorskip("grpc")

    def make(**params) -> FlowerTransport:
        return FlowerTransport(directory=str(tmp_path / "peers"), **params)

    return make


def test_a_frame_travels_as_a_flower_message(make) -> None:
    from flwr.common.serde import message_from_proto
    from flwr.proto.message_pb2 import Message as ProtoMessage

    wire = make()._wrap("onionfl/r/fog/inbox", b"\x00frame")
    message = message_from_proto(ProtoMessage.FromString(wire))

    assert message.metadata.group_id == "onionfl/r/fog/inbox"
    assert message.metadata.dst_node_id == node_id("onionfl/r/fog/inbox")
    assert message.metadata.message_type == "query"
    assert message.content["onionfl"]["frame"] == b"\x00frame"


def test_an_envelope_for_another_inbox_or_not_from_flower_is_refused(make) -> None:
    t = make()
    wire = t._wrap("a", b"x")

    assert t._unwrap("a", wire) == b"x"
    with pytest.raises(ValueError, match="reached 'b'"):
        t._unwrap("b", wire)
    with pytest.raises(ValueError, match="not a Flower message"):
        t._unwrap("a", b"\xff\xff\xff")


def test_a_message_goes_straight_to_the_process_of_its_inbox(make) -> None:
    a, b = make(), make()
    got = Inbox()
    b.on_receive("onionfl/r/fog/inbox", got)
    a.start()
    b.start()
    try:
        a.send("onionfl/r/fog/inbox", b"\x00hello")
        assert got.wait(1) == [b"\x00hello"]
    finally:
        a.stop()
        b.stop()
    # the counters count the frame, not the envelope, as every transport does
    assert a.stats["bytes_sent"] == 6
    assert b.stats == {"sent": 0, "bytes_sent": 0, "received": 1, "bytes_received": 6}


def test_messages_to_one_inbox_arrive_in_order(make) -> None:
    a, b = make(), make()
    got = Inbox()
    b.on_receive("in", got)
    b.start()
    a.start()
    try:
        for i in range(50):
            a.send("in", str(i).encode())
        assert got.wait(50) == [str(i).encode() for i in range(50)]
    finally:
        a.stop()
        b.stop()


def test_large_messages_pass(make) -> None:
    a, b = make(), make()
    got = Inbox()
    b.on_receive("big", got)
    a.start()
    b.start()
    try:
        payload = bytes(8 * 2**20)  # above gRPC's default 4 MiB limit
        a.send("big", payload)
        assert got.wait(1, timeout=30) == [payload]
    finally:
        a.stop()
        b.stop()


def test_it_does_not_mix_with_the_grpc_transport(make, tmp_path: Path) -> None:
    from onion_fl.transports.grpc import GrpcTransport

    grpc = GrpcTransport(directory=str(tmp_path / "peers"), flush_timeout=0.2)
    flower = make()
    got = Inbox()
    flower.on_receive("in", got)
    flower.start()
    grpc.start()
    try:
        grpc.send("in", b"plain")  # another service: never delivered
        time.sleep(0.5)
    finally:
        grpc.stop()
        flower.stop()
    assert got.items == []
