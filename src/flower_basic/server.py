#!/usr/bin/env python3
"""
Central FL server with fog computing (Flower gRPC + MQTT).

Receives partial aggregates from fog nodes via Flower gRPC and performs
global aggregation (FedAvg). Publishes the global model via MQTT.
"""

import json
import os
from typing import Any, List, Optional, Tuple

import flwr as fl
import paho.mqtt.client as mqtt

from .model import ECGModel
from .utils import state_dict_to_numpy

# -----------------------------------------------------------------------------
# MQTT CONFIGURATION
# -----------------------------------------------------------------------------
UPDATE_TOPIC = "fl/partial"  # Fog nodes publish partial aggregates here
MODEL_TOPIC = "fl/global_model"  # Publish global model here
MQTT_BROKER = "localhost"  # Local MQTT broker

# Environment overrides (optional)
try:
    UPDATE_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", UPDATE_TOPIC)
    MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", MODEL_TOPIC)
    MQTT_BROKER = os.getenv("MQTT_BROKER", MQTT_BROKER)
except Exception:
    pass


# -----------------------------------------------------------------------------
# FEDAVG STRATEGY WITH MQTT INTEGRATION
# -----------------------------------------------------------------------------
class MQTTFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg strategy that also publishes global models via MQTT.
    """

    def __init__(self, model: ECGModel, mqtt_client: Optional[mqtt.Client], **kwargs):
        """Initialize FedAvg with MQTT client."""
        super().__init__(**kwargs)
        self.global_model = model
        self.mqtt = mqtt_client
        self.param_names = list(model.state_dict().keys())

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, fl.common.FitRes]],
        failures,
    ) -> Optional[fl.common.Parameters]:
        """Aggregate fog updates and publish the global model."""
        print(f"\n[SERVER] === RONDA {server_round} DE AGREGACIÓN ===")
        print(f"[SERVER] Received {len(results)} partial updates")

        # Ejecutar agregación FedAvg estándar
        new_parameters = super().aggregate_fit(server_round, results, failures)

        if new_parameters is None:
            print("[SERVER] ERROR: aggregation failed, parameters None")
            return None

        try:
            # Debug: Ver el tipo de new_parameters
            print(f"[SERVER] DEBUG: Tipo de parámetros: {type(new_parameters)}")

            # Manejar diferentes tipos de parámetros según versión de Flower
            if isinstance(new_parameters, tuple):
                # Es una tupla (versión antigua): (parameters, fit_metrics_dict)
                parameters_obj = new_parameters[0]
                print(
                    f"[SERVER] DEBUG: Extraída tupla, tipo interno: {type(parameters_obj)}"
                )
            else:
                parameters_obj = new_parameters

            # Convertir parámetros a arrays numpy
            if hasattr(parameters_obj, "tensors"):
                # Flower 1.8+
                param_arrays = parameters_obj.tensors
                param_arrays = [
                    fl.common.bytes_to_ndarray(tensor) for tensor in param_arrays
                ]
                print("[SERVER] DEBUG: Usando parameters_obj.tensors")
            else:
                # Flower anterior - usar la función de conversión
                param_arrays = fl.common.parameters_to_ndarrays(parameters_obj)
                print("[SERVER] DEBUG: Usando parameters_to_ndarrays")

            # Crear state_dict
            state_dict = {}
            for i, name in enumerate(self.param_names):
                if i < len(param_arrays):
                    state_dict[name] = param_arrays[i]

            print(f"[SERVER] DEBUG: State dict creado con {len(state_dict)} parámetros")

            # Serializar para MQTT
            payload = {
                "round": server_round,
                "global_weights": state_dict_to_numpy(state_dict),
            }

            # Publicar modelo global via MQTT
            if self.mqtt is not None:
                self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))
                print(
                    f"[SERVER] ✅ Modelo global publicado en MQTT topic: {MODEL_TOPIC}"
                )
            else:
                print("[SERVER] MQTT no disponible, saltando publicación")

            print("[SERVER] Modelo global agregado exitosamente")

        except Exception as e:
            print(f"[SERVER] ERROR en publicación MQTT: {e}")
            import traceback

            traceback.print_exc()
            print("[SERVER] Continuando sin publicación MQTT")

        return new_parameters


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main(argv=None):
    """
    Función principal que configura e inicia el servidor central.
    """
    import argparse

    global MQTT_BROKER, MODEL_TOPIC
    ap = argparse.ArgumentParser(description="Central FL server with MQTT publishing")
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883"))
    )
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    args, _ = ap.parse_known_args(argv)

    print(f"[SERVER] Central server started at {args.server_addr}")
    print("[SERVER] Aggregating partial updates from fog nodes")

    # Inicializar modelo ECG
    model = ECGModel()

    # Configurar cliente MQTT
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        MQTT_BROKER = args.mqtt_broker
        MODEL_TOPIC = args.topic_global
        mqtt_client.connect(MQTT_BROKER, args.mqtt_port)
        mqtt_client.loop_start()
        print(f"[SERVER] Connected to MQTT broker at {MQTT_BROKER}")
    except Exception as e:
        print(f"[SERVER] MQTT connection failed: {e}, continuing without MQTT")
        mqtt_client = None

    # Crear estrategia FedAvg personalizada
    strategy = MQTTFedAvg(
        model=model,
        mqtt_client=mqtt_client,
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # Sin evaluación para evitar problemas
        min_fit_clients=1,
        min_evaluate_clients=0,
        min_available_clients=1,
    )

    print("[SERVER] Waiting for fog clients...")

    # Iniciar servidor Flower
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
