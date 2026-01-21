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
import atexit
import json
import os
import signal
import time
from collections import defaultdict
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt

from flower_basic.logging_utils import enable_timestamped_print
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_span,
    start_consumer_span,
    start_producer_span,
    start_server_span,
    start_linked_consumer_span,
    start_linked_producer_span,
    inject_trace_context,
    extract_trace_context,
    SpanKind,
)
from flower_basic.prometheus_metrics import (
    start_metrics_server,
    get_metrics_port_from_env,
    push_metrics_to_gateway,
    BROKER_CLIENTS_PER_REGION,
    BROKER_AGGREGATIONS,
    BROKER_BUFFER_SIZE,
    BROKER_UPDATES_RECEIVED,
    BROKER_PARTIALS_PUBLISHED,
    BROKER_CLIENT_CONTRIBUTION,
    set_broker_clients,
    FOG_REGION_ACCURACY,
    FOG_REGION_LOSS,
    FOG_REGION_SAMPLES,
    FOG_REGION_MODEL_NORM,
    FOG_REGION_MODEL_MEAN,
    FOG_REGION_MODEL_STD,
)

# MQTT CONFIG AND AGGREGATION PARAMETERS
UPDATE_TOPIC = "fl/updates"  # clients publish local updates here
PARTIAL_TOPIC = "fl/partial"  # broker publishes partial aggregates here
GLOBAL_TOPIC = "fl/global_model"  # (optional) republish global model

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Number of updates per region before computing partial aggregate
# K_MAP allows per-region thresholds (e.g., {"fog_1": 2, "fog_2": 3})
K = 3
K_MAP: dict[str, int] = {}

# BUFFERS PER REGION
# Each region has its own buffer of updates
buffers = defaultdict(list)

# Track unique clients per region
clients_per_region: dict[str, set] = defaultdict(set)

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
COUNTER_UPDATES_RECEIVED = None
COUNTER_PARTIALS_PUBLISHED = None
HIST_AGGREGATION_TIME = None
GAUGE_BUFFER_SIZE = None
GAUGE_CLIENT_CONTRIBUTION = None
COUNTER_AGGREGATIONS_TOTAL = None
GAUGE_CLIENTS_PER_REGION = None


def _init_telemetry():
    """Initialize telemetry. Called from main() to ensure proper service name."""
    global TRACER, METER
    global COUNTER_UPDATES_RECEIVED, COUNTER_PARTIALS_PUBLISHED, HIST_AGGREGATION_TIME, GAUGE_BUFFER_SIZE
    global GAUGE_CLIENT_CONTRIBUTION, COUNTER_AGGREGATIONS_TOTAL, GAUGE_CLIENTS_PER_REGION

    TRACER, METER = init_otel("fog-broker")

    COUNTER_UPDATES_RECEIVED = create_counter(
        METER, "fl_broker_updates_received_total", "Local updates received from clients"
    )
    COUNTER_PARTIALS_PUBLISHED = create_counter(
        METER,
        "fl_broker_partials_published_total",
        "Partial aggregates published to bridge",
    )
    HIST_AGGREGATION_TIME = create_histogram(
        METER,
        "fl_broker_aggregation_duration_seconds",
        "Time to aggregate client updates",
        "s",
    )
    GAUGE_BUFFER_SIZE = create_gauge(
        METER, "fl_broker_buffer_size", "Current buffer size per region", "1"
    )
    GAUGE_CLIENT_CONTRIBUTION = create_gauge(
        METER,
        "fl_broker_client_contribution",
        "Number of samples contributed by each client",
        "1",
    )
    COUNTER_AGGREGATIONS_TOTAL = create_counter(
        METER, "fl_broker_aggregations_total", "Total regional aggregations performed"
    )
    GAUGE_CLIENTS_PER_REGION = create_gauge(
        METER,
        "fl_broker_clients_per_region",
        "Number of unique clients per fog region",
        "1",
    )


def weighted_average(
    updates: list[dict[str, Any]], weights: list[float] | None = None
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute weighted average of model updates per region.

    Args:
        updates: List of dicts {param_name: numpy_array_serializable}
        weights: Optional weights. If None, use uniform average.

    Returns:
        Tuple of:
        - Dict with averaged parameters to send to central server
        - Dict with centroid statistics (norm, mean, std)
    """
    n = len(updates)
    if weights is None:
        weights = [1.0 / n] * n
    avg = {}

    # For each model parameter...
    all_params = []  # Collect all flattened parameters for stats
    for key in updates[0]:
        # Stack all parameter tensors for this key
        param_arrays = [np.array(up[key]) for up in updates]
        stacked = np.stack(param_arrays, axis=0)  # Shape: (n_updates, *param_shape)

        # Compute weighted average along the first axis
        weights_array = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
        averaged = (stacked * weights_array).sum(axis=0)
        avg[key] = averaged.tolist()
        all_params.append(averaged.flatten())

    # Calculate centroid statistics
    all_weights = np.concatenate(all_params)
    centroid_stats = {
        "norm": float(np.linalg.norm(all_weights)),  # L2 norm
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "num_params": len(all_weights),
    }

    return avg, centroid_stats


def on_update(client, userdata, msg):
    """Handle local client updates and emit partial aggregates per region."""
    try:
        payload = json.loads(msg.payload.decode())

        region = payload.get("region", "default_region")
        weights = payload.get("weights", {})
        client_id = payload.get("client_id", "unknown")
        num_samples = payload.get("num_samples", 0)  # Track contribution
        trace_context = payload.get("trace_context", {})  # Extract trace context

        if not weights:
            print(f"[BROKER] Received empty weights from {client_id}")
            return

        # Use linked CONSUMER span to continue trace from swell-client
        with start_linked_consumer_span(
            TRACER,
            "broker.receive_update",
            trace_context,
            source_service="swell-client",
            attributes={
                "region": region,
                "client_id": client_id,
                "num_samples": num_samples,
            },
        ):
            # Store weights along with metadata for weighted aggregation
            buffers[region].append(
                {
                    "weights": weights,
                    "num_samples": num_samples,
                    "client_id": client_id,
                    "trace_context": trace_context,
                }
            )
            region_k = K_MAP.get(region, K)

            # Track unique clients per region
            clients_per_region[region].add(client_id)

            # Record metrics (OTEL)
            record_metric(
                COUNTER_UPDATES_RECEIVED, 1, {"region": region, "client_id": client_id}
            )
            record_metric(GAUGE_BUFFER_SIZE, len(buffers[region]), {"region": region})
            record_metric(
                GAUGE_CLIENT_CONTRIBUTION,
                num_samples,
                {"region": region, "client_id": client_id},
            )
            record_metric(
                GAUGE_CLIENTS_PER_REGION,
                len(clients_per_region[region]),
                {"region": region},
            )

            # Record Prometheus metrics
            BROKER_UPDATES_RECEIVED.labels(region=region).inc()
            BROKER_BUFFER_SIZE.labels(region=region).set(len(buffers[region]))
            BROKER_CLIENT_CONTRIBUTION.labels(client_id=client_id, region=region).set(
                num_samples
            )
            BROKER_CLIENTS_PER_REGION.labels(region=region).set(
                len(clients_per_region[region])
            )

            print(
                f"[BROKER] Update received from client={client_id}, region={region}, samples={num_samples}. "
                f"Buffer: {len(buffers[region])}/{region_k}"
            )

            if len(buffers[region]) >= region_k:
                agg_start = time.time()

                # Extract weights and compute sample-weighted average
                weight_list = [item["weights"] for item in buffers[region]]
                sample_counts = [item["num_samples"] for item in buffers[region]]
                total_samples = sum(sample_counts)

                # Use sample counts as weights for FedAvg
                if total_samples > 0:
                    weights_for_avg = [s / total_samples for s in sample_counts]
                else:
                    weights_for_avg = None

                # Use SERVER span for aggregation processing
                with start_server_span(
                    TRACER,
                    "broker.aggregate",
                    attributes={
                        "region": region,
                        "num_clients": len(weight_list),
                        "total_samples": total_samples,
                    },
                ):
                    partial, centroid_stats = weighted_average(
                        weight_list, weights_for_avg
                    )
                    agg_duration = time.time() - agg_start

                # Record centroid (model) metrics for this fog region
                FOG_REGION_MODEL_NORM.labels(region=region).set(centroid_stats["norm"])
                FOG_REGION_MODEL_MEAN.labels(region=region).set(centroid_stats["mean"])
                FOG_REGION_MODEL_STD.labels(region=region).set(centroid_stats["std"])
                FOG_REGION_SAMPLES.labels(region=region).set(total_samples)

                print(
                    f"[BROKER] Centroid stats for {region}: norm={centroid_stats['norm']:.4f}, "
                    f"mean={centroid_stats['mean']:.6f}, std={centroid_stats['std']:.4f}, "
                    f"params={centroid_stats['num_params']}"
                )

                # Log contribution breakdown
                for item in buffers[region]:
                    contrib_pct = (
                        (item["num_samples"] / total_samples * 100)
                        if total_samples > 0
                        else 0
                    )
                    print(
                        f"[BROKER] Client {item['client_id']} contributed {item['num_samples']} samples ({contrib_pct:.1f}%)"
                    )

                buffers[region].clear()
                record_metric(GAUGE_BUFFER_SIZE, 0, {"region": region})

                # Record Prometheus metrics after aggregation
                BROKER_BUFFER_SIZE.labels(region=region).set(0)

                # Use PRODUCER span with context propagation to fog-bridge
                with start_linked_producer_span(
                    TRACER,
                    "broker.publish_partial",
                    target_service="fog-bridge",
                    attributes={"region": region, "total_samples": total_samples},
                ) as (span, trace_ctx):
                    msg_payload = {
                        "region": region,
                        "partial_weights": partial,
                        "total_samples": total_samples,
                        "timestamp": time.time(),
                        "trace_context": trace_ctx,  # Propagate trace context
                    }
                    client.publish(PARTIAL_TOPIC, json.dumps(msg_payload))

                # Record aggregation metrics (OTEL)
                record_metric(COUNTER_PARTIALS_PUBLISHED, 1, {"region": region})
                record_metric(COUNTER_AGGREGATIONS_TOTAL, 1, {"region": region})
                if HIST_AGGREGATION_TIME:
                    HIST_AGGREGATION_TIME.record(agg_duration, {"region": region})

                # Record Prometheus metrics
                BROKER_PARTIALS_PUBLISHED.labels(region=region).inc()
                BROKER_AGGREGATIONS.labels(region=region).inc()

                print(
                    f"[BROKER] Partial aggregate published for region={region} (total samples: {total_samples})"
                )

    except Exception as e:
        print(f"[BROKER ERROR] Error procesando actualización: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Start fog broker MQTT loop."""
    global K, MQTT_BROKER, MQTT_PORT, UPDATE_TOPIC, PARTIAL_TOPIC, GLOBAL_TOPIC

    enable_timestamped_print()

    # Initialize telemetry for this service
    _init_telemetry()

    # Start Prometheus metrics server
    metrics_port = get_metrics_port_from_env(default=8001, component="BROKER")
    start_metrics_server(port=metrics_port)

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

    # Cleanup function to push metrics before exit
    def cleanup(*args):
        print("[BROKER] Pushing metrics before shutdown...")
        push_metrics_to_gateway(
            job="flower-broker", grouping_key={"component": "broker"}
        )
        shutdown_telemetry()

    # Register cleanup for various termination signals
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(0)))
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), exit(0)))

    try:
        mqttc.loop_forever()
    except KeyboardInterrupt:
        print("[BROKER] Shutting down...")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
