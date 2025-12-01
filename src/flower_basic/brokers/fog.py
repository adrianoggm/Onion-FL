#!/usr/bin/env python3
"""Fog broker (regional aggregator).

This component aggregates local client updates per fog region before sending a
partial model to the central server. It implements the fog layer aggregation in
the hierarchical FL architecture.

Flow:
1. Subscribe to 'fl/updates' to receive client updates.
2. Buffer K updates per region.
3. Compute weighted average.
4. Publish partial aggregate to 'fl/partial'.
5. Clear buffer and wait for the next batch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import paho.mqtt.client as mqtt

from flower_basic.telemetry import create_counter, init_otel, start_span

# MQTT CONFIG AND AGGREGATION PARAMETERS
UPDATE_TOPIC = "fl/updates"  # clients publish local updates here
PARTIAL_TOPIC = "fl/partial"  # broker publishes partial aggregates here
GLOBAL_TOPIC = "fl/global_model"  # (optional) republish global model

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Number of updates per region before computing partial aggregate
# K_MAP allows per-region thresholds (e.g., {"fog_1": 2, "fog_2": 3})
K = 3
K_MAP: Dict[str, int] = {}

# BUFFERS PER REGION
# Each region has its own buffer of updates
buffers = defaultdict(list)

# Environment overrides (optional)
try:
    UPDATE_TOPIC = os.getenv("MQTT_TOPIC_UPDATES", UPDATE_TOPIC)
    PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", PARTIAL_TOPIC)
    GLOBAL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", GLOBAL_TOPIC)
    MQTT_BROKER = os.getenv("MQTT_BROKER", MQTT_BROKER)
    MQTT_PORT = int(os.getenv("MQTT_PORT", str(MQTT_PORT)))
    K = int(os.getenv("FOG_K", str(K)))
    _k_map_env = os.getenv("FOG_K_MAP")
    if _k_map_env:
        try:
            parsed = json.loads(_k_map_env)
            K_MAP = {str(k): int(v) for k, v in parsed.items()}
        except Exception:
            K_MAP = {}
except Exception:
    pass

# Telemetry - initialized lazily in main() to avoid import-time side effects
TRACER = None
METER = None
COUNTER_UPDATE = None
COUNTER_PARTIAL = None


def _init_telemetry():
    """Initialize telemetry. Called from main() to ensure proper service name."""
    global TRACER, METER, COUNTER_UPDATE, COUNTER_PARTIAL
    TRACER, METER = init_otel("fog-broker")
    COUNTER_UPDATE = create_counter(
        METER, "broker.updates.received", "Local updates received per region"
    )
    COUNTER_PARTIAL = create_counter(
        METER, "broker.partials.published", "Partial aggregates published per region"
    )


def weighted_average(
    updates: List[Dict[str, Any]], weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """Compute weighted average of model updates per region.

    Args:
        updates: List of dicts {param_name: numpy_array_serializable}
        weights: Optional weights. If None, use uniform average.

    Returns:
        Dict with averaged parameters to send to central server
    """
    n = len(updates)
    if weights is None:
        weights = [1.0 / n] * n
    avg = {}

    # For each model parameter...
    for key in updates[0]:
        # Stack all parameter tensors for this key
        param_arrays = [np.array(up[key]) for up in updates]
        stacked = np.stack(param_arrays, axis=0)  # Shape: (n_updates, *param_shape)

        # Compute weighted average along the first axis
        weights_array = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
        avg[key] = (stacked * weights_array).sum(axis=0).tolist()

    return avg


def on_update(client, userdata, msg):
    """Handle local client updates and emit partial aggregates per region."""
    try:
        payload = json.loads(msg.payload.decode())

        region = payload.get("region", "default_region")
        weights = payload.get("weights", {})
        client_id = payload.get("client_id", "unknown")

        if not weights:
            print(f"[BROKER] Received empty weights from {client_id}")
            return

        with start_span(
            TRACER, "broker.on_update", {"region": region, "client_id": client_id}
        ):
            buffers[region].append(weights)
            region_k = K_MAP.get(region, K)
            if COUNTER_UPDATE:
                COUNTER_UPDATE.add(1, {"region": region})
            print(
                f"[BROKER] Update received from client={client_id}, region={region}. "
                f"Buffer: {len(buffers[region])}/{region_k}"
            )

            if len(buffers[region]) >= region_k:
                partial = weighted_average(buffers[region])
                buffers[region].clear()

                msg = {
                    "region": region,
                    "partial_weights": partial,
                    "timestamp": time.time(),
                }
                client.publish(PARTIAL_TOPIC, json.dumps(msg))
                if COUNTER_PARTIAL:
                    COUNTER_PARTIAL.add(1, {"region": region})
                print(f"[BROKER] Partial aggregate published for region={region}")

    except Exception as e:
        print(f"[BROKER ERROR] Error procesando actualización: {e}")
        print(f"[BROKER ERROR] Mensaje: {msg.payload.decode()[:200]}...")


def main():
    """Start fog broker MQTT loop."""
    global K, MQTT_BROKER, MQTT_PORT, UPDATE_TOPIC, PARTIAL_TOPIC, GLOBAL_TOPIC
    
    # Initialize telemetry for this service
    _init_telemetry()
    
    parser = argparse.ArgumentParser(description="Fog broker (regional aggregator)")
    parser.add_argument(
        "--k",
        type=int,
        default=int(os.getenv("FOG_K", K)),
        help="Updates per region before computing partial aggregate",
    )
    parser.add_argument(
        "--k-map",
        type=str,
        default=os.getenv("FOG_K_MAP"),
        help="JSON mapping {region: k_value} to override K per region",
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
    if args.k_map:
        try:
            parsed = json.loads(args.k_map)
            K_MAP.clear()
            K_MAP.update({str(k): max(1, int(v)) for k, v in parsed.items()})
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[BROKER] Ignorando k-map inválido: {exc}")

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
