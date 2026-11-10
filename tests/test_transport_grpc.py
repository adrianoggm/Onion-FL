"""The gRPC transport: direct delivery, order per inbox, reconnection (issue #196).

Every server listens on 127.0.0.1 with an ephemeral port, and every test has
its own discovery folder.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("grpc")

from onion_fl.transports import transports  # noqa: E402
from onion_fl.transports.grpc import GrpcTransport  # noqa: E402


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


def make(tmp_path: Path, **params) -> GrpcTransport:
    return GrpcTransport(directory=str(tmp_path / "peers"), **params)


def test_it_is_a_registered_transport() -> None:
    assert transports.get("grpc").title == "gRPC"


def test_a_message_goes_straight_to_the_process_of_its_inbox(tmp_path: Path) -> None:
    a, b = make(tmp_path), make(tmp_path)
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
    assert b.endpoint is None and a.stats["bytes_sent"] == 6
    assert b.stats == {"sent": 0, "bytes_sent": 0, "received": 1, "bytes_received": 6}


def test_it_listens_on_localhost_by_default(tmp_path: Path) -> None:
    t = make(tmp_path)
    t.on_receive("x", lambda data: None)
    t.start()
    try:
        assert t.endpoint.startswith("127.0.0.1:")
        assert (tmp_path / "peers" / "x").read_text() == t.endpoint
    finally:
        t.stop()
    assert not (tmp_path / "peers" / "x").exists()


def test_messages_to_one_inbox_arrive_in_order(tmp_path: Path) -> None:
    a, b = make(tmp_path), make(tmp_path)
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


def test_a_message_waits_for_a_receiver_that_is_not_up_yet(tmp_path: Path) -> None:
    a, b = make(tmp_path), make(tmp_path)
    got = Inbox()
    a.start()
    try:
        a.send("late", b"kept")
        time.sleep(0.3)
        b.on_receive("late", got)
        b.start()
        assert got.wait(1) == [b"kept"]
    finally:
        a.stop()
        b.stop()


def test_a_receiver_that_restarts_on_another_port_is_found_again(
    tmp_path: Path,
) -> None:
    a = make(tmp_path)
    first, second = Inbox(), Inbox()
    b = make(tmp_path)
    b.on_receive("node", first)
    a.start()
    b.start()
    try:
        a.send("node", b"one")
        assert first.wait(1) == [b"one"]
        b.stop()
        a.send("node", b"two")  # nobody listens: retried
        time.sleep(0.3)
        c = make(tmp_path)
        c.on_receive("node", second)
        c.start()
        try:
            assert second.wait(1) == [b"two"]
        finally:
            c.stop()
    finally:
        a.stop()
    assert first.items == [b"one"]


def test_large_messages_pass(tmp_path: Path) -> None:
    a, b = make(tmp_path), make(tmp_path)
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


def test_stop_drops_what_cannot_be_delivered_within_the_flush(tmp_path: Path) -> None:
    a = make(tmp_path, flush_timeout=0.2)
    a.start()
    a.send("nobody", b"lost")
    started = time.monotonic()
    a.stop()
    assert time.monotonic() - started < 3


@pytest.mark.parametrize("params", [{"port": -1}, {"call_timeout": 0}, {"workers": 0}])
def test_grpc_params_are_validated(params) -> None:
    with pytest.raises(ValueError):
        transports.create("grpc", params)
