"""
fog_flower_client.py - Cliente Puente entre MQTT y Flower

Este módulo implementa un cliente Flower que actúa como puente entre:
1. El broker fog (broker_fog.py) que envía agregados parciales por MQTT
2. El servidor central Flower (server.py) que coordina el aprendizaje federado

ARQUITECTURA:
- Recibe agregados parciales de broker_fog.py vía MQTT topic 'fl/partial'
- Los reenvía al servidor central usando el protocolo gRPC de Flower
- Permite que el fog computing se integre con el framework Flower

FLUJO:
1. Broker fog acumula K actualizaciones locales → agregado parcial
2. Este cliente recibe el agregado parcial vía MQTT
3. Lo convierte a formato Flower y lo envía al servidor central
4. El servidor central agrega múltiples regiones → modelo global
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
# CONFIGURACIÓN MQTT
# -----------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", "fl/partial")  # Topic donde broker_fog publica agregados parciales


# -----------------------------------------------------------------------------
# Cliente Flower para nodo fog (puente MQTT-Flower)
# -----------------------------------------------------------------------------
class FogClient(fl.client.NumPyClient):
    """
    Cliente Flower que actúa como puente entre el broker fog MQTT y el servidor central.

    Funciones principales:
    1. Se conecta al servidor Flower central vía gRPC
    2. Recibe agregados parciales del broker fog vía MQTT
    3. Los reenvía al servidor central en formato Flower
    4. Permite integración fog computing + Flower framework
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

        # Configurar suscriptor MQTT para recibir agregados parciales
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_partial
        self.mqtt.connect(mqtt_broker, mqtt_port)
        self.mqtt.loop_start()
        print(
            f"[FOG_CLIENT] Conectado a MQTT broker, escuchando agregados parciales en {self.partial_topic}"
        )

    def _on_partial(self, client, userdata, msg):
        """
        Callback MQTT que procesa agregados parciales del broker fog.

        Recibe el mensaje JSON con los pesos agregados de K clientes de una región
        y los almacena para enviar al servidor central en la próxima ronda.
        """
        try:
            data = json.loads(msg.payload.decode())
            self.partial_weights = data.get("partial_weights")
            region = data.get("region", "unknown")
            print(f"[FOG_CLIENT] Agregado parcial recibido para region={region}")
        except Exception as e:
            print(f"[FOG_CLIENT] Error procesando agregado parcial: {e}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback MQTT para conexión exitosa."""
        if rc == 0:
            client.subscribe(self.partial_topic)
            print(f"[FOG_CLIENT] Suscrito al topic: {self.partial_topic}")
        else:
            print(f"[FOG_CLIENT] Error de conexión MQTT: {rc}")

    # Interfaz Flower NumPyClient
    def get_parameters(self, config):
        """Devuelve parámetros actuales del modelo."""
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config):
        """
        Método principal de entrenamiento para Flower.

        En lugar de entrenar localmente, este cliente:
        1. Recibe parámetros globales del servidor central
        2. Espera agregado parcial del broker fog vía MQTT
        3. Devuelve el agregado parcial como "entrenamiento local"

        Esto permite que el servidor central vea los agregados fog como clientes regulares.
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

        # Devolver agregado parcial como si fuera entrenamiento local
        # num_samples representa la suma de muestras de los K clientes agregados
        num_samples = 1000  # Aproximadamente K=3 clientes con ~333 muestras cada uno
        print(
            f"[FOG_CLIENT] Enviando agregado parcial al servidor central ({num_samples} muestras)"
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
