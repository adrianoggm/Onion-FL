from __future__ import annotations

"""MQTT transport over paho (spec §9.2).

Each node listens on ``onionfl/<run_id>/<node_id>/inbox``. Broker and QoS are
set per link in the topology (``transport: {name: mqtt, broker: host:port,
qos: 1}``). paho reconnects with an exponential backoff, and every inbox is
subscribed again on each connect, so a broker restart does not lose a node.

A transport holds two connections to the broker: one listens on the inboxes,
the other publishes. A process often sends to nodes it hosts itself (a fog to
its edges), and with one connection a burst of large messages then flows both
ways on the same socket at once. Docker Desktop's port forwarding on Windows
stalls for good once about 1 MB is in flight each way (#192); each connection
carrying one direction avoids it, and costs nothing elsewhere.

``tls`` turns on MQTTS (#227): the broker's certificate is checked against
``tls.ca`` and the broker's host name, and ``tls.cert`` with ``tls.key``
authenticate the client by certificate. ``username`` with ``password_env``
(an environment variable's name) or ``password_file`` (a path) authenticate it
by password. The link holds names and paths, never the secret: it ends up in
the ``config_id`` and in ``run.json``. A broker that refuses the credentials
makes ``start`` fail at once, with the broker's reason.

Both connections turn Nagle's algorithm off (``TCP_NODELAY``) unless the link
says ``nodelay: false``. With Nagle, the last partial segment of a message
waits until the previous one is acknowledged: a round trip whenever messages
leave closer together than that, as an edge host's updates or a fog's models
do. On the emulated lab it added 11 to 35 ms to an update and up to 50 ms to
a burst of control messages (#198).
"""

import os
import socket
import ssl
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from onion_fl.transports import transports
from onion_fl.transports.tls import TlsParams, env_name, expand, path_only, read

Callback = Callable[[bytes], None]


class MqttParams(BaseModel):
    broker: str = Field("localhost:1883", description="host:port del broker")
    qos: int = Field(1, ge=0, le=2)
    keepalive: int = Field(30, gt=0)
    reconnect_min: float = Field(1.0, gt=0, description="Primer reintento (s)")
    reconnect_max: float = Field(30.0, gt=0, description="Reintento máximo (s)")
    connect_timeout: float = Field(10.0, gt=0)
    tls: TlsParams | None = Field(
        None, description="MQTTS: rutas de la CA y, para TLS mutuo, del certificado"
    )
    username: str | None = Field(None, description="Usuario en el broker")
    password_env: str | None = Field(
        None, description="Variable de entorno con la contraseña (su nombre)"
    )
    password_file: str | None = Field(
        None, description="Ruta del fichero con la contraseña"
    )

    _env = field_validator("password_env")(lambda v: env_name(v))
    _file = field_validator("password_file")(lambda v: path_only(v))

    @model_validator(mode="after")
    def _credentials(self) -> MqttParams:
        if self.password_env is not None and self.password_file is not None:
            raise ValueError("give password_env or password_file, not both")
        has_password = self.password_env is not None or self.password_file is not None
        if has_password and self.username is None:
            raise ValueError("a password needs a username")
        return self

    nodelay: bool = Field(
        True,
        description="TCP_NODELAY en las dos conexiones (sin Nagle); false lo deja",
        exclude_if=lambda v: v,  # the default keeps existing ids
    )

    @field_validator("broker")
    @classmethod
    def _host_port(cls, value: str) -> str:
        host, sep, port = value.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"broker must be host:port, got {value!r}")
        return value


@transports.register(
    "mqtt",
    title="MQTT",
    description="Un inbox por nodo en un broker MQTT; QoS y broker por enlace.",
    params=MqttParams,
)
class MqttTransport:
    def __init__(self, **params: Any) -> None:
        self.params = MqttParams(**params)
        host, _, port = self.params.broker.rpartition(":")
        self.host, self.port = host, int(port)
        self.stats = {"sent": 0, "bytes_sent": 0, "received": 0, "bytes_received": 0}
        self._inboxes: dict[str, list[Callback]] = {}
        self._listening = threading.Event()  # connected, every inbox acknowledged
        self._publishing = threading.Event()
        self._lock = threading.Lock()
        self._pending: set[int] = set()  # subscriptions not acknowledged yet
        self._listener: Any = None
        self._publisher: Any = None
        self._refused: str | None = None  # the broker's reason, if it said no

    def _password(self) -> str | None:
        params = self.params
        if params.password_env is not None:
            value = os.environ.get(params.password_env)
            if value is None:
                raise ConnectionError(
                    f"MQTT password: environment variable {params.password_env} "
                    "is not set"
                )
            return value
        if params.password_file is not None:
            text = read(params.password_file, "password_file").decode("utf-8")
            return text.rstrip("\r\n")
        return None

    def _tls_context(self) -> ssl.SSLContext | None:
        tls = self.params.tls
        if tls is None:
            return None
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # checks name and chain
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        try:
            context.load_verify_locations(cadata=read(tls.ca, "ca").decode("ascii"))
            if tls.cert is not None:
                context.load_cert_chain(expand(tls.cert), expand(tls.key))
        except (ssl.SSLError, OSError, ValueError) as exc:
            raise ConnectionError(f"MQTT TLS: {exc}") from None
        return context

    def _accepted(self, reason_code: Any) -> bool:
        # paho 2 gives a ReasonCode; a bare int (0: accepted) is taken too
        if not getattr(reason_code, "is_failure", reason_code != 0):
            return True
        self._refused = str(reason_code)
        return False

    def on_receive(self, address: str, callback: Callback) -> None:
        first = address not in self._inboxes
        self._inboxes.setdefault(address, []).append(callback)
        if first and self._listening.is_set():
            self._listener.subscribe(address, self.params.qos)

    def _on_connect(
        self, client, userdata, flags, reason_code, properties=None
    ) -> None:
        if not self._accepted(reason_code):
            return
        with self._lock:
            self._pending.clear()
            for address in self._inboxes:  # again after every reconnect
                _, mid = client.subscribe(address, self.params.qos)
                self._pending.add(mid)
            if not self._pending:
                self._listening.set()

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties) -> None:
        with self._lock:
            self._pending.discard(mid)
            if not self._pending:
                self._listening.set()

    def _on_disconnect(self, *args: Any) -> None:
        self._listening.clear()

    def _on_message(self, client, userdata, message) -> None:
        data = bytes(message.payload)
        self.stats["received"] += 1
        self.stats["bytes_received"] += len(data)
        for callback in self._inboxes.get(message.topic, []):
            callback(data)

    def _on_publisher_connect(
        self, client, userdata, flags, reason_code, properties=None
    ) -> None:
        if self._accepted(reason_code):
            self._publishing.set()

    def _client(
        self, context: ssl.SSLContext | None, password: str | None, **callbacks: Any
    ) -> Any:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        for name, callback in callbacks.items():
            setattr(client, name, callback)
        if context is not None:
            client.tls_set_context(context)
        if self.params.username is not None:
            client.username_pw_set(self.params.username, password)
        if self.params.nodelay:  # on every socket, reconnections included
            client.on_socket_open = _no_delay
        client.reconnect_delay_set(self.params.reconnect_min, self.params.reconnect_max)
        client.connect_async(self.host, self.port, self.params.keepalive)
        client.loop_start()
        return client

    def start(self) -> None:
        """Connect both ways; return once every inbox is subscribed."""
        context, password = self._tls_context(), self._password()
        self._refused = None
        self._listener = self._client(
            context,
            password,
            on_connect=self._on_connect,
            on_subscribe=self._on_subscribe,
            on_disconnect=self._on_disconnect,
            on_message=self._on_message,
        )
        self._publisher = self._client(
            context,
            password,
            on_connect=self._on_publisher_connect,
            on_disconnect=lambda *args: self._publishing.clear(),
        )
        deadline = time.monotonic() + self.params.connect_timeout
        for ready in (self._listening, self._publishing):
            while not ready.wait(0.05):
                if self._refused is not None:
                    self.stop()
                    raise ConnectionError(
                        f"MQTT broker {self.params.broker} refused the connection: "
                        f"{self._refused}"
                    )
                if time.monotonic() > deadline:
                    self.stop()
                    secure = " (TLS handshake included)" if context else ""
                    raise ConnectionError(
                        f"MQTT broker {self.params.broker} did not answer{secure}"
                    )

    def send(self, address: str, data: bytes) -> None:
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += len(data)
        self._publisher.publish(address, data, self.params.qos)

    def stop(self) -> None:
        for client in (self._listener, self._publisher):
            if client is not None:
                client.loop_stop()
                client.disconnect()
        self._listener = self._publisher = None


def _no_delay(client: Any, userdata: Any, sock: Any) -> None:
    """paho's ``on_socket_open``: send each segment at once (no Nagle)."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
