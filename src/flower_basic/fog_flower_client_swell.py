from __future__ import annotations

"""
Fog bridge client for SWELL model.

Receives partial aggregates over MQTT (from broker_fog) and forwards them
to the Flower server as a NumPyClient. Uses SwellMLP to keep parameter
names consistent for ordering.
"""

import json
import os
import time
from typing import List

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt
from pathlib import Path

# Support running as script: add src to path and import absolute package
try:
    from .swell_model import SwellMLP, get_parameters, set_parameters
except Exception:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flower_basic.swell_model import SwellMLP, get_parameters, set_parameters


MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", "fl/partial")


class FogClientSwell(fl.client.NumPyClient):
    def __init__(
        self,
        server_address: str,
        input_dim: int,
        region: str,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        partial_topic: str = PARTIAL_TOPIC,
    ):
        self.server_address = server_address
        self.model = SwellMLP(input_dim=input_dim)
        self.param_names = list(self.model.state_dict().keys())
        self.partial_weights = None
        self.partial_topic = partial_topic
        self.region = region
        self.tag = f"[BRIDGE {self.region}]"

        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_partial
        self.mqtt.connect(mqtt_broker, mqtt_port)
        self.mqtt.loop_start()
        print(f"{self.tag} Listening for partials on {self.partial_topic}")

    def _on_partial(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            self.partial_weights = data.get("partial_weights")
            region = data.get("region", "unknown")
            if region != self.region:
                # Ignorar agregados de otras regiones; otro fog_bridge los consumirá
                return
            print(f"{self.tag} Partial aggregate received for region={region}")
        except Exception as e:
            print(f"{self.tag} Error processing partial: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(self.partial_topic)
            print(f"{self.tag} Subscribed to: {self.partial_topic}")
        else:
            print(f"{self.tag} MQTT connect error: {rc}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config):
        set_parameters(self.model, parameters)
        timeout = 60
        waited = 0
        while self.partial_weights is None and waited < timeout:
            time.sleep(0.5)
            waited += 0.5
        if self.partial_weights is None:
            print(f"{self.tag} Timeout waiting for partial")
            return get_parameters(self.model), 1, {}
        partial_list = [np.array(self.partial_weights[name], dtype=np.float32) for name in self.param_names]
        self.partial_weights = None
        num_samples = 1000
        print(f"{self.tag} Forwarding partial to central server")
        return partial_list, num_samples, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fog bridge client for SWELL")
    ap.add_argument("--input_dim", type=int, required=True, help="Feature dimension (from manifest)")
    ap.add_argument("--server", default="localhost:8080")
    ap.add_argument("--region", required=True, help="Fog region id this bridge represents (e.g., fog_0)")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-partial", default=PARTIAL_TOPIC)
    args = ap.parse_args()

    print("[FOG_CLIENT_SWELL] Starting bridge client...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=FogClientSwell(
            args.server,
            args.input_dim,
            region=args.region,
            mqtt_broker=args.mqtt_broker,
            mqtt_port=args.mqtt_port,
            partial_topic=args.topic_partial,
        ),
    )


if __name__ == "__main__":
    main()
