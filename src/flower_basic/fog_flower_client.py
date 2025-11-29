"""
Fog bridge client (generic).

Receives partial aggregates over MQTT (from broker_fog) and forwards them to
the central Flower server as a NumPyClient. Enables fog → cloud bridging.
"""

import json
import os
import time
from typing import List

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt

from .model import ECGModel, get_parameters, set_parameters

# -----------------------------------------------------------------------------
# MQTT CONFIG
# -----------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PARTIAL_TOPIC = os.getenv(
    "MQTT_TOPIC_PARTIAL", "fl/partial"
)  # Topic where broker_fog publishes partials


# -----------------------------------------------------------------------------
# Cliente Flower para nodo fog (puente MQTT-Flower)
# -----------------------------------------------------------------------------
class FogClient(fl.client.NumPyClient):
    """
    Flower NumPyClient acting as a bridge between MQTT fog broker and central server.
    """

    def __init__(
        self,
        server_address: str,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        partial_topic: str = PARTIAL_TOPIC,
    ):
        self.server_address = server_address
        self.model = ECGModel()
        self.param_names = list(self.model.state_dict().keys())
        self.partial_weights = None
        self.partial_topic = partial_topic

        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_partial
        self.mqtt.connect(mqtt_broker, mqtt_port)
        self.mqtt.loop_start()
        print(
            f"[FOG_CLIENT] Connected to MQTT broker, listening for partials on {self.partial_topic}"
        )

    def _on_partial(self, client, userdata, msg):
        """
        Process partial aggregates published by the fog broker.
        """
        try:
            data = json.loads(msg.payload.decode())
            self.partial_weights = data.get("partial_weights")
            region = data.get("region", "unknown")
            print(f"[FOG_CLIENT] Partial aggregate received for region={region}")
        except Exception as e:
            print(f"[FOG_CLIENT] Error processing partial aggregate: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connect callback."""
        if rc == 0:
            client.subscribe(self.partial_topic)
            print(f"[FOG_CLIENT] Suscrito al topic: {self.partial_topic}")
        else:
            print(f"[FOG_CLIENT] MQTT connection error: {rc}")

    # Interfaz Flower NumPyClient
    def get_parameters(self, config):
        """Return current model parameters."""
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config):
        """
        Training entrypoint for Flower: waits for MQTT partial and returns it.
        """
        # Cargar nuevos parámetros globales del servidor central
        set_parameters(self.model, parameters)

        # Esperar agregado parcial de clientes locales vía MQTT
        timeout_count = 0
        while (
            self.partial_weights is None and timeout_count < 60
        ):  # 30 segundos timeout
            time.sleep(0.5)
            timeout_count += 1

        if self.partial_weights is None:
            print("[FOG_CLIENT] Timeout esperando agregado parcial")
            return get_parameters(self.model), 1, {}

        # Convertir dict parcial a lista de parámetros en orden correcto
        partial_list = [
            np.array(self.partial_weights[name], dtype=np.float32)
            for name in self.param_names
        ]
        self.partial_weights = None

        # Return partial aggregate as if it were local training
        num_samples = 1000  # approximate: aggregated clients
        print(
            f"[FOG_CLIENT] Sending partial aggregate to central server ({num_samples} samples)"
        )
        return partial_list, num_samples, {}

    def evaluate(self, parameters, config):
        """
        Evaluación no implementada para nodos fog.

        Los nodos fog solo actúan como puentes, no realizan evaluación local.
        """
        return 0.0, 0, {}


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: Inicializar cliente fog
# -----------------------------------------------------------------------------
def main():
    """
    Inicia el cliente fog que conecta:
    - MQTT (para recibir agregados parciales del broker fog)
    - Flower gRPC (para comunicarse con el servidor central)

    Este es el puente entre la capa fog (MQTT) y la capa central (Flower).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Fog bridge client (generic)")
    ap.add_argument("--server", default="localhost:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-partial", default=PARTIAL_TOPIC)
    args = ap.parse_args()

    print("[FOG_CLIENT] Iniciando cliente puente fog-central...")
    print(f"[FOG_CLIENT] Conectando al servidor Flower en {args.server}")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=FogClient(
            args.server,
            mqtt_broker=args.mqtt_broker,
            mqtt_port=args.mqtt_port,
            partial_topic=args.topic_partial,
        ),
    )


if __name__ == "__main__":
    main()
