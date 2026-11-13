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

from onion_fl.core.message import Message  # noqa: E402
from onion_fl.core.node import Node  # noqa: E402
from onion_fl.runtime.real import RealRuntime  # noqa: E402
from onion_fl.transports import transports  # noqa: E402
from onion_fl.transports.grpc import GrpcTransport  # noqa: E402
from tests.timing import slack  # noqa: E402


class Inbox:
    def __init__(self) -> None:
        self.items: list[bytes] = []
        self.principals: list[str | None] = []
        self._cv = threading.Condition()

    def __call__(self, data: bytes, principal: str | None = None) -> None:
        with self._cv:
            self.principals.append(principal)
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
    assert time.monotonic() - started < slack(3)


@pytest.mark.parametrize("params", [{"port": -1}, {"call_timeout": 0}, {"workers": 0}])
def test_grpc_params_are_validated(params) -> None:
    (field,) = params
    with pytest.raises(ValueError, match=rf"\n{field}\n"):
        transports.create("grpc", params)


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- across machines (issue #199) ---------------------------------------------------------------


def test_grpc_finds_its_peers_through_the_map_and_never_the_folder(
    tmp_path: Path,
) -> None:
    a_port, b_port = free_port(), free_port()
    peers = {"box/a": f"127.0.0.1:{a_port}", "box/b": f"127.0.0.1:{b_port}"}
    folder = tmp_path / "never"
    a = GrpcTransport(port=a_port, peers=peers, directory=str(folder))
    b = GrpcTransport(port=b_port, peers=peers, directory=str(folder))
    got = threading.Event()
    seen = []
    b.on_receive("box/b", lambda data: (seen.append(data), got.set()))
    a.on_receive("box/a", lambda data: None)
    a.start()
    b.start()
    try:
        a.send("box/b", b"hello")
        assert got.wait(10)
    finally:
        a.stop()
        b.stop()

    assert seen == [b"hello"] and not folder.exists()


def test_a_real_runtime_can_give_every_grpc_link_its_listen_and_peers() -> None:
    runtime = RealRuntime(
        "r",
        heartbeat_s=None,
        grpc={"listen": "0.0.0.0:50999", "peers": {"a": "10.0.0.1:50051"}},
    )
    runtime.add_link("a", "b", transport={"name": "grpc", "call_timeout": 3})

    (carrier,) = runtime._transports.values()
    assert (carrier.params.host, carrier.params.port) == ("0.0.0.0", 50999)
    assert carrier.params.peers == {"onionfl/r/a/inbox": "10.0.0.1:50051"}
    assert carrier.params.call_timeout == 3


# --- mutual TLS (issue #227) ---------------------------------------------------------------


@pytest.fixture
def pki(tmp_path: Path):
    from tests.certs import Pki

    return Pki(tmp_path / "pki")


def tls_of(pki, name: str) -> dict:
    cert, key = pki.issue(name, ["127.0.0.1", "localhost"])
    return {"ca": str(pki.ca), "cert": str(cert), "key": str(key)}


def test_grpc_over_mutual_tls_delivers(tmp_path: Path, pki) -> None:
    a = make(tmp_path, tls=tls_of(pki, "a"))
    b = make(tmp_path, tls=tls_of(pki, "b"))
    got = Inbox()
    b.on_receive("secure", got)
    a.start()
    b.start()
    try:
        a.send("secure", b"sealed")
        assert got.wait(1) == [b"sealed"]
    finally:
        a.stop()
        b.stop()
    assert got.principals == ["a"]  # the CN of the sender's certificate


def _call(channel, address: str = "secure") -> None:
    from onion_fl.transports.grpc import ADDRESS, GrpcTransport

    with channel:
        channel.unary_unary(f"/{GrpcTransport.SERVICE}/Deliver")(
            b"intruder", timeout=3, metadata=((ADDRESS, address),)
        )


def test_grpc_tls_rejects_clients_without_a_valid_certificate(
    tmp_path: Path, pki
) -> None:
    import grpc

    from tests.certs import Pki

    server = make(tmp_path, tls=tls_of(pki, "server"))
    got = Inbox()
    server.on_receive("secure", got)
    server.start()
    stranger = Pki(tmp_path / "other", name="another-ca")
    foreign_cert, foreign_key = stranger.issue("stranger", ["127.0.0.1"])
    ca = pki.ca.read_bytes()
    try:
        clients = {
            "plain": grpc.insecure_channel(server.endpoint),
            "tls without a certificate": grpc.secure_channel(
                server.endpoint, grpc.ssl_channel_credentials(root_certificates=ca)
            ),
            "a certificate another CA signed": grpc.secure_channel(
                server.endpoint,
                grpc.ssl_channel_credentials(
                    root_certificates=ca,
                    private_key=foreign_key.read_bytes(),
                    certificate_chain=foreign_cert.read_bytes(),
                ),
            ),
        }
        for what, channel in clients.items():
            with pytest.raises(grpc.RpcError) as refused:
                _call(channel)
            assert refused.value.code() == grpc.StatusCode.UNAVAILABLE, what
    finally:
        server.stop()
    assert got.items == []  # nothing an unauthenticated client sent got in


def test_a_plain_grpc_transport_cannot_talk_to_a_tls_one(tmp_path: Path, pki) -> None:
    plain = make(tmp_path, flush_timeout=0.2)
    secure = make(tmp_path, tls=tls_of(pki, "secure"))
    got = Inbox()
    secure.on_receive("secure", got)
    plain.start()
    secure.start()
    try:
        plain.send("secure", b"plain")
        assert got.wait(1, timeout=1.0) == []
    finally:
        plain.stop()
        secure.stop()


@pytest.mark.parametrize(
    ("tls", "error"),
    [
        # mutual: the certificate is required
        ({"ca": "ca.pem"}, "give tls.cert and tls.key"),
        ({"ca": "ca.pem", "cert": "c.pem"}, "give cert and key together"),
        (
            {"ca": "-----BEGIN CERTIFICATE-----\nMIIB...", "cert": "c", "key": "k"},
            "not its contents",
        ),
        ({"ca": "ca.pem", "cert": "c", "key": "k", "password": "x"}, "tls.password"),
    ],
)
def test_grpc_tls_takes_paths_only(tls, error) -> None:
    with pytest.raises(ValueError, match=error):
        transports.create("grpc", {"tls": tls})


def test_grpc_tls_files_are_read_on_start_and_missing_ones_are_named(
    tmp_path: Path,
) -> None:
    t = make(tmp_path, tls={"ca": "nowhere/ca.pem", "cert": "c.pem", "key": "k.key"})
    with pytest.raises(ConnectionError, match="ca 'nowhere/ca.pem'"):
        t.start()


class Recorder(Node):
    stopped = True  # passive

    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.got: list[Message] = []

    def on_message(self, msg, ctx) -> None:
        self.got.append(msg)


class Forger(Node):
    """Says it is the operator, and claims the principal too."""

    stopped = False

    def on_start(self, ctx) -> None:
        meta = {"promote": 1, "override": True, "principal": "operator"}
        ctx.send(Message(kind="control", src=self.id, dst="cloud", meta=meta))
        ctx.set_timer(1.0, "done")

    def on_timer(self, name, ctx) -> None:
        self.stopped = True


@pytest.mark.parametrize("common_name", ["operator", "intruder"])
def test_the_runtime_tells_who_tls_authenticated_never_who_the_sender_says(
    tmp_path: Path, pki, common_name: str
) -> None:
    """The principal comes from the client certificate, never from ``meta``
    (#227): the coordinator binds an order's ``src`` to it."""
    folder = str(tmp_path / "peers")

    def link(name: str) -> dict:
        return {"name": "grpc", "directory": folder, "tls": tls_of(pki, name)}

    here = RealRuntime("orders", hosted={"cloud"}, heartbeat_s=None)
    there = RealRuntime("orders", hosted={"operator"}, heartbeat_s=None)
    cloud = Recorder("cloud")
    here.add_node(cloud)
    here.add_node(Forger("operator"))
    here.add_link("operator", "cloud", transport=link("cloud"))
    there.add_node(Recorder("cloud"))
    there.add_node(Forger("operator"))
    there.add_link("operator", "cloud", transport=link(common_name))
    here.start()
    try:
        there.run(timeout=10)
    finally:
        here.stop()

    (order,) = cloud.got
    assert order.src == "operator" and order.meta["promote"] == 1
    assert order.meta["principal"] == common_name
