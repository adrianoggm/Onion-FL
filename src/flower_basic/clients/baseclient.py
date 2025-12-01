from __future__ import annotations

"""Base MQTT component used by federated clients and bridges."""

import json
from typing import Iterable

import paho.mqtt.client as mqtt


class BaseMQTTComponent:
    """Common MQTT setup with tagging, subscriptions and safe publish."""

    def __init__(
        self,
        tag: str,
        mqtt_broker: str,
        mqtt_port: int,
        subscriptions: Iterable[str] | None = None,
    ) -> None:
        self.tag = tag
        self._subscriptions = list(subscriptions or [])
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect_wrapper
        self.mqtt.on_message = self._on_message_wrapper
        self.mqtt.connect(mqtt_broker, mqtt_port, keepalive=60)
        self.mqtt.loop_start()

    # Overridable hooks -------------------------------------------------
    def on_message(
        self, client, userdata, msg
    ) -> None:  # pragma: no cover - to be overridden
        """Handle MQTT message (override in subclasses)."""
        return None

    # Wrappers ----------------------------------------------------------
    def _on_connect_wrapper(self, client, userdata, flags, rc, properties=None):
        for topic in self._subscriptions:
            client.subscribe(topic)
        print(
            f"{self.tag} MQTT connected (rc={rc}). Subscribed to {self._subscriptions}"
        )

    def _on_message_wrapper(self, client, userdata, msg):
        try:
            self.on_message(client, userdata, msg)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"{self.tag} Error handling MQTT message: {exc}")

    # Helpers -----------------------------------------------------------
    def publish_json(self, topic: str, payload) -> None:
        """Publish JSON payload with tagging."""
        try:
            self.mqtt.publish(topic, json.dumps(payload))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"{self.tag} Error publishing to {topic}: {exc}")

    def stop_mqtt(self) -> None:
        try:
            self.mqtt.loop_stop()
            self.mqtt.disconnect()
        except Exception:
            pass
