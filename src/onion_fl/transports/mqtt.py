from __future__ import annotations

"""MQTT transport over paho (spec §9.2).

Each node listens on ``onionfl/<run_id>/<node_id>/inbox``. Broker and QoS are
set per link in the topology (``transport: {name: mqtt, broker: host:port,
qos: 1}``). paho reconnects with an exponential backoff, and every inbox is
subscribed again on each connect, so a broker restart does not lose a node.
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
        self._connected = threading.Event()
        self._client: Any = None

    def on_receive(self, address: str, callback: Callback) -> None:
        first = address not in self._inboxes
        self._inboxes.setdefault(address, []).append(callback)
        if first and self._connected.is_set():
            self._client.subscribe(address, self.params.qos)

    def _on_connect(
        self, client, userdata, flags, reason_code, properties=None
    ) -> None:
        for address in self._inboxes:  # again after every reconnect
            client.subscribe(address, self.params.qos)
        self._connected.set()

    def _on_disconnect(self, *args: Any) -> None:
        self._connected.clear()

    def _on_message(self, client, userdata, message) -> None:
        data = bytes(message.payload)
        self.stats["received"] += 1
        self.stats["bytes_received"] += len(data)
        for callback in self._inboxes.get(message.topic, []):
            callback(data)

    def start(self) -> None:
        import paho.mqtt.client as mqtt

        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(
            self.params.reconnect_min, self.params.reconnect_max
        )
        self._client.connect_async(self.host, self.port, self.params.keepalive)
        self._client.loop_start()
        if not self._connected.wait(self.params.connect_timeout):
            raise ConnectionError(f"MQTT broker {self.params.broker} did not answer")

    def send(self, address: str, data: bytes) -> None:
        self.stats["sent"] += 1
        self.stats["bytes_sent"] += len(data)
        self._client.publish(address, data, self.params.qos)

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
