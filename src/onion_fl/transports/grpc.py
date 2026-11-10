from __future__ import annotations

"""gRPC transport: direct links, no broker (spec §3, issue #196).

Every transport runs one gRPC server on ``host`` (``127.0.0.1`` by default,
an ephemeral port unless ``port`` says otherwise) and serves the inboxes
listened on through it. A message goes straight from the sender's process to
the receiver's: a logical link is one connection, with no broker in between.

The service is generic: one unary method, ``/onionfl.Transport/Deliver``,
takes the frame the runtime already built (bytes in, empty bytes out) and the
inbox in the ``onionfl-address`` metadata. No ``.proto`` is needed.

Discovery is a folder (``directory``, under the system temp by default):
each transport writes there, per inbox it serves, a file with its
``host:port``, and the sender reads it. That covers every process of one
machine (the ``local`` backend). Across machines (the ``distributed``
backend, I14) the launcher gives ``peers`` instead, an explicit map from each
inbox to the ``host:port`` that serves it, and the folder is not used.

The guarantees of MQTT with QoS 1:

- **delivery per node**: one queue and one thread per inbox, so the messages
  to a node arrive in the order they were sent, and a message that cannot be
  delivered is kept and retried, not lost;
- **reconnection**: a failed delivery retries with an exponential backoff
  (``reconnect_min`` to ``reconnect_max``), and reads the inbox's file again,
  so a receiver that is not up yet, or that restarted on another port, is
  found. A message may then arrive twice; the runtime drops the duplicate by
  its ``msg_id``.

On ``stop``, what is still queued gets one more try each within
``flush_timeout``: the ``stop`` that travels down the tree leaves as the
sender finishes.

``tls`` (#227) turns every link into mutual TLS: the server presents
``tls.cert`` and accepts only clients whose certificate ``tls.ca`` signed, and
each client checks the server's certificate against the same CA and the
peer's address. Both certificates are required: an unauthenticated client,
plain or TLS without a certificate, never completes the handshake, so
nothing it sends reaches an inbox. The link holds paths only (see
``onion_fl.transports.tls``). Each frame is handed over with ``principal``,
the common name of the client certificate that sent it.
"""

import os
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent import futures
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator

from onion_fl.core.registry import PluginError
from onion_fl.transports import transports
from onion_fl.transports.tls import TlsParams, read

Callback = Callable[[bytes], None]
METHOD = "/onionfl.Transport/Deliver"
ADDRESS = "onionfl-address"
UNLIMITED = [
    ("grpc.max_send_message_length", -1),
    ("grpc.max_receive_message_length", -1),
]


class GrpcParams(BaseModel):
    host: str = Field("127.0.0.1", description="Interfaz en la que escucha")
    port: int = Field(0, ge=0, le=65535, description="0: un puerto libre")
    directory: str | None = Field(
        None, description="Carpeta de descubrimiento; por defecto, en el temporal"
    )
    peers: dict[str, str] | None = Field(
        None,
        description=(
            "Buzón -> host:puerto que lo sirve, entre máquinas; sustituye a la carpeta"
        ),
    )
    call_timeout: float = Field(10.0, gt=0, description="Plazo de cada envío (s)")
    reconnect_min: float = Field(0.05, gt=0, description="Primer reintento (s)")
    reconnect_max: float = Field(5.0, gt=0, description="Reintento máximo (s)")
    flush_timeout: float = Field(5.0, ge=0, description="Vaciado al parar (s)")
    workers: int = Field(4, ge=1, description="Hilos del servidor")
    tls: TlsParams | None = Field(
        None,
        description="TLS mutuo: rutas de la CA, del certificado y de la clave",
    )

    @model_validator(mode="after")
    def _mutual(self) -> GrpcParams:
        if self.tls is not None and self.tls.cert is None:
            raise ValueError("grpc tls is mutual: give tls.cert and tls.key")
        return self


def _require_grpc() -> Any:
    try:
        import grpc
    except ImportError:
        raise PluginError(
            "the grpc transport needs grpcio: pip install 'onion-fl[grpc]'"
        ) from None
    return grpc


def _common_name(context: Any) -> str | None:
    """The CN of the client certificate gRPC verified, if it has one."""
    names = context.auth_context().get("x509_common_name") or []
    return names[0].decode("utf-8") if names else None


@transports.register(
    "grpc",
    title="gRPC",
    description=(
        "Enlaces directos entre procesos, sin broker; un servidor gRPC por "
        "proceso, descubierto por una carpeta local o por un mapa de direcciones."
    ),
    params=GrpcParams,
)
class GrpcTransport:
    def __init__(self, **params: Any) -> None:
        self._grpc = _require_grpc()
        self.params = GrpcParams(**params)
        self.directory = Path(
            self.params.directory or Path(tempfile.gettempdir()) / "onionfl-grpc"
        )
        self.stats = {"sent": 0, "bytes_sent": 0, "received": 0, "bytes_received": 0}
        self.endpoint: str | None = None
        self._inboxes: dict[str, list[Callback]] = {}
        self._outboxes: dict[str, queue.Queue] = {}
        self._threads: list[threading.Thread] = []
        self._channels: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._flush_until = 0.0
        self._server: Any = None
        self._credentials: Any = None  # the channels', once TLS files are read

    def _tls(self) -> tuple[bytes, bytes, bytes] | None:
        """The CA, certificate and key, read when the transport starts."""
        tls = self.params.tls
        if tls is None:
            return None
        return read(tls.ca, "ca"), read(tls.cert, "cert"), read(tls.key, "key")

    # --- receiving -----------------------------------------------------------

    def _file(self, address: str) -> Path:
        return self.directory / quote(address, safe="")

    def _announce(self, address: str) -> None:
        if self.params.peers is not None:  # the launcher's map says where it is
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._file(address)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(self.endpoint or "", encoding="utf-8")
        os.replace(partial, path)

    def on_receive(self, address: str, callback: Callback) -> None:
        with self._lock:
            first = address not in self._inboxes
            self._inboxes.setdefault(address, []).append(callback)
        if first and self.endpoint is not None:
            self._announce(address)

    def _deliver(self, request: bytes, context: Any) -> bytes:
        address = dict(context.invocation_metadata()).get(ADDRESS, "")
        with self._lock:
            callbacks = list(self._inboxes.get(address, []))
            if callbacks:
                self.stats["received"] += 1
                self.stats["bytes_received"] += len(request)
        if not callbacks:  # the sender read a stale file: let it look again
            context.abort(self._grpc.StatusCode.NOT_FOUND, f"no inbox {address!r} here")
        if self.params.tls is None:
            for callback in callbacks:
                callback(request)
            return b""
        principal = _common_name(context)
        for callback in callbacks:
            callback(request, principal=principal)
        return b""

    def start(self) -> None:
        grpc = self._grpc
        self._stopping.clear()
        handler = grpc.method_handlers_generic_handler(
            "onionfl.Transport",
            {"Deliver": grpc.unary_unary_rpc_method_handler(self._deliver)},
        )
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.params.workers),
            handlers=[handler],
            options=UNLIMITED,
        )
        bound = f"{self.params.host}:{self.params.port}"
        files = self._tls()
        if files is None:
            port = self._server.add_insecure_port(bound)
        else:
            ca, cert, key = files
            server = grpc.ssl_server_credentials(
                [(key, cert)], root_certificates=ca, require_client_auth=True
            )
            port = self._server.add_secure_port(bound, server)
            self._credentials = grpc.ssl_channel_credentials(
                root_certificates=ca, private_key=key, certificate_chain=cert
            )
        if port == 0:
            raise ConnectionError(f"gRPC could not listen on {bound}")
        self.endpoint = f"{self.params.host}:{port}"
        self._server.start()
        with self._lock:
            addresses = list(self._inboxes)
        for address in addresses:
            self._announce(address)

    # --- sending -------------------------------------------------------------

    def send(self, address: str, data: bytes) -> None:
        with self._lock:
            self.stats["sent"] += 1
            self.stats["bytes_sent"] += len(data)
            outbox = self._outboxes.get(address)
            if outbox is None:
                outbox = self._outboxes[address] = queue.Queue()
                thread = threading.Thread(
                    target=self._drain, args=(address, outbox), daemon=True
                )
                self._threads.append(thread)
                thread.start()
        outbox.put(data)

    def _resolve(self, address: str) -> str | None:
        if self.params.peers is not None:
            return self.params.peers.get(address)
        try:
            return self._file(address).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _channel(self, endpoint: str) -> Any:
        with self._lock:
            if endpoint not in self._channels:
                if self._credentials is None:
                    channel = self._grpc.insecure_channel(endpoint, options=UNLIMITED)
                else:
                    channel = self._grpc.secure_channel(
                        endpoint, self._credentials, options=UNLIMITED
                    )
                self._channels[endpoint] = channel
            return self._channels[endpoint]

    def _try(self, address: str, data: bytes) -> bool:
        timeout = self.params.call_timeout
        if self._stopping.is_set():
            timeout = min(timeout, self._flush_until - time.monotonic())
            if timeout <= 0:  # the flush is over: drop what is left
                return False
        endpoint = self._resolve(address)
        if endpoint is None:
            return False
        try:
            call = self._channel(endpoint).unary_unary(METHOD)
            call(data, timeout=timeout, metadata=((ADDRESS, address),))
        except (self._grpc.RpcError, ValueError):  # ValueError: channel closed
            return False
        return True

    def _drain(self, address: str, outbox: queue.Queue) -> None:
        while True:
            data = outbox.get()
            if data is None:
                return
            delay = self.params.reconnect_min
            while not self._try(address, data):
                if self._stopping.is_set():  # one try each while flushing
                    break
                self._stopping.wait(delay)
                delay = min(delay * 2, self.params.reconnect_max)

    def stop(self) -> None:
        self._flush_until = time.monotonic() + self.params.flush_timeout
        self._stopping.set()
        with self._lock:
            outboxes, threads = list(self._outboxes.values()), list(self._threads)
            self._outboxes.clear()
            self._threads.clear()
        for outbox in outboxes:
            outbox.put(None)
        for thread in threads:
            thread.join(max(0.0, self._flush_until - time.monotonic()))
        with self._lock:
            addresses = list(self._inboxes)
            channels = list(self._channels.values())
            self._channels.clear()
        for address in addresses:
            if self.params.peers is None and self._resolve(address) == self.endpoint:
                self._file(address).unlink(missing_ok=True)
        for channel in channels:
            channel.close()
        if self._server is not None:
            self._server.stop(grace=None).wait()
            self._server = None
        self.endpoint = None
