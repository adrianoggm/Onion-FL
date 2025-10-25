#!/usr/bin/env python3
"""Broker Fog - Nodo de Agregación Regional.

Este componente actúa como un nodo fog que agrega actualizaciones de múltiples
clientes locales antes de enviarlas al servidor central. Implementa agregación
jerárquica en la arquitectura de aprendizaje federado.

Funcionalidades:
- Recibe actualizaciones de modelo de clientes locales via MQTT
- Agrega K actualizaciones por región usando promedio ponderado
- Publica agregados parciales para que los fog clients los reenvíen al servidor central
- Maneja múltiples regiones simultáneamente

Flujo:
1. Escucha en topic 'fl/updates' para recibir actualizaciones de clientes
2. Acumula actualizaciones hasta tener K por región
3. Computa promedio ponderado de los K modelos
4. Publica agregado parcial en topic 'fl/partial'
5. Resetea buffer y espera próximas actualizaciones

Configuración:
- K=3: Número de clientes por región antes de agregar
- Soporte para múltiples regiones concurrentes
"""

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import argparse
import os

# -----------------------------------------------------------------------------
import paho.mqtt.client as mqtt

# CONFIGURACIÓN MQTT Y PARÁMETROS DE AGREGACIÓN
# -----------------------------------------------------------------------------
UPDATE_TOPIC = "fl/updates"  # clientes publican actualizaciones aquí
PARTIAL_TOPIC = "fl/partial"  # este broker publica agregados parciales aquí
GLOBAL_TOPIC = "fl/global_model"  # (opcional) para re-publicar modelo global

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Número de actualizaciones por región antes de computar agregado parcial
K = 3

# -----------------------------------------------------------------------------
# BUFFERS DE AGREGACIÓN POR REGIÓN
# -----------------------------------------------------------------------------
# Cada región mantiene su propio buffer de actualizaciones
buffers = defaultdict(list)

# Environment overrides (optional)
try:
    UPDATE_TOPIC = os.getenv("MQTT_TOPIC_UPDATES", UPDATE_TOPIC)
    PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", PARTIAL_TOPIC)
    GLOBAL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", GLOBAL_TOPIC)
    MQTT_BROKER = os.getenv("MQTT_BROKER", MQTT_BROKER)
    MQTT_PORT = int(os.getenv("MQTT_PORT", str(MQTT_PORT)))
    K = int(os.getenv("FOG_K", str(K)))
except Exception:
    pass


# -----------------------------------------------------------------------------
# Utility: element‐wise weighted average of a list of dicts {param_name: list}
# -----------------------------------------------------------------------------
def weighted_average(
    updates: List[Dict[str, Any]], weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Realiza un promedio ponderado de actualizaciones de modelo por región.

    Args:
        updates: Lista de diccionarios con parámetros del modelo
                Cada dict tiene formato {param_name: numpy_array_serializable}
        weights: Pesos para el promedio (opcional). Si None, usa promedio uniforme.

    Returns:
        Diccionario con parámetros promediados para agregar al servidor central
    """
    n = len(updates)
    if weights is None:
        weights = [1.0 / n] * n
    avg = {}

    # Para cada parámetro del modelo...
    for key in updates[0]:
        # Stack all parameter tensors for this key
        param_arrays = [np.array(up[key]) for up in updates]
        stacked = np.stack(param_arrays, axis=0)  # Shape: (n_updates, *param_shape)

        # Compute weighted average along the first axis
        weights_array = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
        avg[key] = (stacked * weights_array).sum(axis=0).tolist()

    return avg


# -----------------------------------------------------------------------------
# MQTT CALLBACK: Manejo de actualizaciones de clientes locales
# -----------------------------------------------------------------------------
def on_update(client, userdata, msg):
    """
    Callback que procesa actualizaciones de modelos enviadas por clientes locales.

    Recibe actualizaciones vía MQTT del topic 'fl/updates', las almacena por región
    y cuando acumula K actualizaciones computa un agregado parcial que envía al
    servidor central vía topic 'fl/partial'.
    """
    try:
        payload = json.loads(msg.payload.decode())

        # Extraer información del mensaje del cliente
        region = payload.get("region", "default_region")
        weights = payload.get("weights", {})
        client_id = payload.get("client_id", "unknown")

        if not weights:
            print(f"[BROKER] Received empty weights from {client_id}")
            return

        # Almacenar actualización en buffer de la región
        buffers[region].append(weights)
        print(
            f"[BROKER] Actualización recibida de cliente={client_id}, region={region}. "
            f"Buffer: {len(buffers[region])}/{K}"
        )

        # Cuando se acumulan K actualizaciones, computar agregado parcial
        if len(buffers[region]) >= K:
            partial = weighted_average(buffers[region])
            buffers[region].clear()

            # Publicar agregado parcial hacia el servidor central
            msg = {
                "region": region,
                "partial_weights": partial,
                "timestamp": time.time(),
            }
            client.publish(PARTIAL_TOPIC, json.dumps(msg))
            print(f"[BROKER] Agregado parcial publicado para region={region}")

    except Exception as e:
        print(f"[BROKER ERROR] Error procesando actualización: {e}")
        print(f"[BROKER ERROR] Mensaje: {msg.payload.decode()[:200]}...")


# -----------------------------------------------------------------------------
# THREAD: listen for partials and forward to Flower aggregator
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# FUNCIÓN PRINCIPAL: Inicialización del broker fog
# -----------------------------------------------------------------------------
def main():
    """
    Función principal que inicializa el broker fog MQTT.

    Este broker:
    1. Se conecta al broker MQTT local (localhost:1883)
    2. Se suscribe al topic 'fl/updates' para recibir actualizaciones de clientes
    3. Acumula K actualizaciones por región en buffers
    4. Computa agregados parciales y los publica en 'fl/partial'
    5. Los agregados parciales son consumidos por el servidor central
    """
    # CLI/env configuration
    global K, MQTT_BROKER, MQTT_PORT, UPDATE_TOPIC, PARTIAL_TOPIC, GLOBAL_TOPIC
    parser = argparse.ArgumentParser(description="Fog broker (regional aggregator)")
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("FOG_K", K)),
        help="Updates per region before computing partial aggregate",
    )
    parser.add_argument(
        "--mqtt-broker",
        default=os.getenv("MQTT_BROKER", MQTT_BROKER),
        help="MQTT broker host",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=int(os.getenv("MQTT_PORT", MQTT_PORT)),
        help="MQTT broker port",
    )
    parser.add_argument(
        "--topic-updates",
        default=os.getenv("MQTT_TOPIC_UPDATES", UPDATE_TOPIC),
        help="Topic for client updates -> broker",
    )
    parser.add_argument(
        "--topic-partial",
        default=os.getenv("MQTT_TOPIC_PARTIAL", PARTIAL_TOPIC),
        help="Topic for broker partials -> fog bridge",
    )
    parser.add_argument(
        "--topic-global",
        default=os.getenv("MQTT_TOPIC_GLOBAL", GLOBAL_TOPIC),
        help="Topic for global model publish",
    )
    args = parser.parse_args()

    K = max(1, int(args.k))
    MQTT_BROKER = args.mqtt_broker
    MQTT_PORT = int(args.mqtt_port)
    UPDATE_TOPIC = args.topic_updates
    PARTIAL_TOPIC = args.topic_partial
    GLOBAL_TOPIC = args.topic_global

    # Configurar cliente MQTT con callback API v2
    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqttc.on_connect = lambda c, u, f, rc, p=None: c.subscribe(UPDATE_TOPIC)
    mqttc.on_message = on_update
    mqttc.connect(MQTT_BROKER, MQTT_PORT)
    print(f"[BROKER] Broker fog iniciado. Escuchando actualizaciones en {UPDATE_TOPIC}")
    print(
        f"[BROKER] Agregando K={K} actualizaciones por región antes de enviar al servidor central"
    )
    mqttc.loop_forever()


if __name__ == "__main__":
    main()
