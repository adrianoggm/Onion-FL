from __future__ import annotations

"""The MQTT transport against a real broker (``ONIONFL_MQTT=host:port``).

Skips when no broker is reachable.
"""

import os
import socket
import threading
import uuid

import pytest

from onion_fl.transports.mqtt import MqttTransport

BROKER = os.environ.get("ONIONFL_MQTT", "localhost:1883")


def _broker_up() -> bool:
    host, _, port = BROKER.partition(":")
    try:
        with socket.create_connection((host, int(port or 1883)), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _broker_up(), reason=f"no MQTT broker at {BROKER}")


@pytest.mark.parametrize("nodelay", [True, False])
def test_both_connections_follow_the_links_nagle(nodelay: bool) -> None:
    transport = MqttTransport(broker=BROKER, nodelay=nodelay)
    transport.on_receive(f"onionfl/test-{uuid.uuid4().hex}/node/inbox", print)
    transport.start()
    try:
        for client in (transport._listener, transport._publisher):
            sock = client.socket()
            assert bool(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)) is (
                nodelay
            )
    finally:
        transport.stop()


def test_a_burst_of_large_messages_to_its_own_inbox_arrives() -> None:
    """One process sends a model to each of its own children at once.

    30 messages of 150 KB (one update of the architectures experiment) to an
    inbox the same transport listens on. Over one connection that carries both
    directions, Docker Desktop's port forwarding on Windows stalls once about
    1 MB is in flight each way, and nothing arrives (#192).
    """
    count, size = 30, 150_000
    inbox = f"onionfl/test-{uuid.uuid4().hex}/node/inbox"
    got: list[bytes] = []
    done = threading.Event()

    def receive(data: bytes) -> None:
        got.append(data)
        if len(got) == count:
            done.set()

    transport = MqttTransport(broker=BROKER, qos=1)
    transport.on_receive(inbox, receive)
    transport.start()
    try:
        payloads = [bytes([i]) * size for i in range(count)]
        for payload in payloads:
            transport.send(inbox, payload)
        assert done.wait(20), f"{len(got)} of {count} messages arrived"
    finally:
        transport.stop()
    assert got == payloads  # all of them, in order
    assert transport.stats["bytes_sent"] == transport.stats["bytes_received"]
