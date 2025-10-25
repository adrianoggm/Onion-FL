#!/usr/bin/env python3
"""
server.py - Servidor Central de Aprendizaje Federado con Fog Computing

Este servidor actúa como el nodo central en la arquitectura de aprendizaje federado
con fog computing. Recibe agregados parciales de los nodos fog via Flower gRPC y
realiza la agregación global usando el algoritmo FedAvg.

ARQUITECTURA:
- Servidor Flower que maneja comunicación gRPC con nodos fog
- Cliente MQTT que publica modelos globales a todos los clientes
- Estrategia FedAvg personalizada que integra MQTT con Flower

FLUJO:
1. Nodos fog se conectan via Flower gRPC como clientes
2. Cada nodo fog envía agregado parcial de su región (K clientes)
3. Servidor agrega todos los parciales usando FedAvg
4. Publica modelo global actualizado via MQTT topic 'fl/global_model'
5. Proceso se repite para múltiples rondas de entrenamiento

CONFIGURACIÓN:
- Puerto Flower: localhost:8080 (gRPC)
- Broker MQTT: localhost:1883
- Rondas: 3 por defecto
- Clientes mínimos: 1 (un nodo fog por región)
"""

import json
import os
from typing import Any, List, Optional, Tuple

import flwr as fl
import paho.mqtt.client as mqtt

from .model import ECGModel
from .utils import state_dict_to_numpy

# -----------------------------------------------------------------------------
# CONFIGURACIÓN MQTT
# -----------------------------------------------------------------------------
UPDATE_TOPIC = "fl/partial"  # Nodos fog publican agregados parciales aquí
MODEL_TOPIC = "fl/global_model"  # Publicamos el modelo global aquí
MQTT_BROKER = "localhost"  # Broker MQTT local

# Environment overrides (optional)
try:
    UPDATE_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", UPDATE_TOPIC)
    MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", MODEL_TOPIC)
    MQTT_BROKER = os.getenv("MQTT_BROKER", MQTT_BROKER)
except Exception:
    pass


# -----------------------------------------------------------------------------
# ESTRATEGIA FEDAVG PERSONALIZADA CON INTEGRACIÓN MQTT
# -----------------------------------------------------------------------------
class MQTTFedAvg(fl.server.strategy.FedAvg):
    """
    Estrategia FedAvg personalizada que integra MQTT para fog computing.

    Extiende FedAvg estándar para:
    - Publicar modelos globales via MQTT después de cada agregación
    - Manejar comunicación con múltiples regiones fog
    - Proporcionar logging detallado del proceso de agregación
    """

    def __init__(self, model: ECGModel, mqtt_client: Optional[mqtt.Client], **kwargs):
        """
        Inicializa estrategia FedAvg con cliente MQTT.

        Args:
            model: Modelo ECG CNN para referencia de parámetros
            mqtt_client: Cliente MQTT configurado para publicación
            **kwargs: Parámetros estándar de FedAvg
        """
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
        """
        Agrega parámetros de nodos fog y publica modelo global.

        Args:
            server_round: Número de ronda actual
            results: Lista de resultados de nodos fog
            failures: Lista de fallos (no usada)

        Returns:
            Parámetros agregados o None si falla
        """
        print(f"\n[SERVER] === RONDA {server_round} DE AGREGACIÓN ===")
        print(f"[SERVER] Recibidas {len(results)} actualizaciones parciales")

        # Ejecutar agregación FedAvg estándar
        new_parameters = super().aggregate_fit(server_round, results, failures)

        if new_parameters is None:
            print("[SERVER] ERROR: Agregación falló, parámetros None")
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
def main():
    """
    Función principal que configura e inicia el servidor central.
    """
    import argparse
    ap = argparse.ArgumentParser(description="Central FL server with MQTT publishing")
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    args = ap.parse_args()

    print(f"[SERVER] Servidor central iniciado en {args.server_addr}")
    print("[SERVER] Agregando actualizaciones parciales de nodos fog")

    # Inicializar modelo ECG
    model = ECGModel()

    # Configurar cliente MQTT
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        global MQTT_BROKER, MODEL_TOPIC
        MQTT_BROKER = args.mqtt_broker
        MODEL_TOPIC = args.topic_global
        mqtt_client.connect(MQTT_BROKER, args.mqtt_port)
        mqtt_client.loop_start()
        print(f"[SERVER] Conectado a MQTT broker en {MQTT_BROKER}")
    except Exception as e:
        print(f"[SERVER] MQTT conexión falló: {e}, continuando sin MQTT")
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

    print("[SERVER] Esperando clientes fog...")

    # Iniciar servidor Flower
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
