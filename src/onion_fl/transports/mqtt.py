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
"""

import threading
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator

from onion_fl.transports import transports

Callback = Callable[[bytes], None]


class MqttParams(BaseModel):
    broker: str = Field("localhost:1883", description="host:port del broker")
    qos: int = Field(1, ge=0, le=2)
    keepalive: int = Field(30, gt=0)
    reconnect_min: float = Field(1.0, gt=0, description="Primer reintento (s)")
    reconnect_max: float = Field(30.0, gt=0, description="Reintento máximo (s)")
    connect_timeout: float = Field(10.0, gt=0)

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

    def on_receive(self, address: str, callback: Callback) -> None:
        first = address not in self._inboxes
        self._inboxes.setdefault(address, []).append(callback)
        if first and self._listening.is_set():
            self._listener.subscribe(address, self.params.qos)

    def _on_connect(
        self, client, userdata, flags, reason_code, properties=None
    ) -> None:
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

    def _client(self, **callbacks: Any) -> Any:
        import paho.mqtt.client as mqtt

        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        for name, callback in callbacks.items():
            setattr(client, name, callback)
        client.reconnect_delay_set(self.params.reconnect_min, self.params.reconnect_max)
        client.connect_async(self.host, self.port, self.params.keepalive)
        client.loop_start()
        return client

    def start(self) -> None:
        """Connect both ways; return once every inbox is subscribed."""
        self._listener = self._client(
            on_connect=self._on_connect,
            on_subscribe=self._on_subscribe,
            on_disconnect=self._on_disconnect,
            on_message=self._on_message,
        )
        self._publisher = self._client(
            on_connect=lambda *args: self._publishing.set(),
            on_disconnect=lambda *args: self._publishing.clear(),
        )
        timeout = self.params.connect_timeout
        if not (self._listening.wait(timeout) and self._publishing.wait(timeout)):
            raise ConnectionError(f"MQTT broker {self.params.broker} did not answer")

    def send(self, address: str, data: bytes) -> None:
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += len(data)
        self._publisher.publish(address, data, self.params.qos)

    def stop(self) -> None:
        for client in (self._listener, self._publisher):
            if client is not None:
                client.loop_stop()
                client.disconnect()
