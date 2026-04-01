#!/usr/bin/env python3
"""SWEET Fog broker (regional aggregator)."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
from collections import defaultdict

import paho.mqtt.client as mqtt

from flower_basic.brokers.federated_base import (
    BrokerCallbacks,
    BrokerConfig,
    BrokerTelemetryHandles,
    handle_client_update,
    parse_k_map,
    weighted_average,
)
from flower_basic.prometheus_metrics import (
    BROKER_AGGREGATIONS,
    BROKER_BUFFER_SIZE,
    BROKER_CLIENT_CONTRIBUTION,
    BROKER_CLIENTS_PER_REGION,
    BROKER_PARTIALS_PUBLISHED,
    BROKER_UPDATES_RECEIVED,
    FOG_REGION_MODEL_MEAN,
    FOG_REGION_MODEL_NORM,
    FOG_REGION_MODEL_STD,
    FOG_REGION_SAMPLES,
    get_metrics_port_from_env,
    start_metrics_server,
)
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    shutdown_telemetry,
)

UPDATE_TOPIC = "fl/updates"
PARTIAL_TOPIC = "fl/partial"
GLOBAL_TOPIC = "fl/global_model"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

K = 1
K_MAP: dict[str, int] = {}

buffers = defaultdict(list)
clients_per_region: dict[str, set] = defaultdict(set)

try:
    UPDATE_TOPIC = os.getenv("MQTT_TOPIC_UPDATES", UPDATE_TOPIC)
    PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", PARTIAL_TOPIC)
    GLOBAL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", GLOBAL_TOPIC)
    MQTT_BROKER = os.getenv("MQTT_BROKER", MQTT_BROKER)
    MQTT_PORT = int(os.getenv("MQTT_PORT", str(MQTT_PORT)))
    K = int(os.getenv("FOG_K", str(K)))
    K_MAP = parse_k_map(os.getenv("FOG_K_MAP"), broker_tag="[SWEET_FOG_BROKER]")
except Exception:
    pass

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

    TRACER, METER = init_otel("sweet-fog-broker")

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


def shutdown_broker_runtime() -> None:
    """Flush broker telemetry without pushing broker gauges to Pushgateway."""
    shutdown_telemetry()


def _broker_config() -> BrokerConfig:
    return BrokerConfig(
        broker_tag="[SWEET_FOG_BROKER]",
        source_service="sweet-client",
        target_service="sweet-fog-bridge",
        partial_topic=PARTIAL_TOPIC,
        default_k=K,
        k_map=K_MAP,
        use_round_metadata=False,
    )


def _broker_telemetry() -> BrokerTelemetryHandles:
    return BrokerTelemetryHandles(
        tracer=TRACER,
        counter_updates_received=COUNTER_UPDATES_RECEIVED,
        counter_partials_published=COUNTER_PARTIALS_PUBLISHED,
        hist_aggregation_time=HIST_AGGREGATION_TIME,
        gauge_buffer_size=GAUGE_BUFFER_SIZE,
        gauge_client_contribution=GAUGE_CLIENT_CONTRIBUTION,
        counter_aggregations_total=COUNTER_AGGREGATIONS_TOTAL,
        gauge_clients_per_region=GAUGE_CLIENTS_PER_REGION,
    )


def _record_prometheus_update(
    region: str, client_id: str, num_samples: int, buffer_size: int, num_clients: int
) -> None:
    BROKER_UPDATES_RECEIVED.labels(region=region).inc()
    BROKER_BUFFER_SIZE.labels(region=region).set(buffer_size)
    BROKER_CLIENT_CONTRIBUTION.labels(client_id=client_id, region=region).set(
        num_samples
    )
    BROKER_CLIENTS_PER_REGION.labels(region=region).set(num_clients)


def _record_prometheus_buffer_cleared(region: str) -> None:
    BROKER_BUFFER_SIZE.labels(region=region).set(0)


def _record_prometheus_aggregation(region: str) -> None:
    BROKER_PARTIALS_PUBLISHED.labels(region=region).inc()
    BROKER_AGGREGATIONS.labels(region=region).inc()


def _record_region_model_metrics(
    region: str, centroid_stats: dict[str, float], total_samples: int
) -> None:
    FOG_REGION_MODEL_NORM.labels(region=region).set(centroid_stats["norm"])
    FOG_REGION_MODEL_MEAN.labels(region=region).set(centroid_stats["mean"])
    FOG_REGION_MODEL_STD.labels(region=region).set(centroid_stats["std"])
    FOG_REGION_SAMPLES.labels(region=region).set(total_samples)


def _broker_callbacks() -> BrokerCallbacks:
    return BrokerCallbacks(
        record_prometheus_update=_record_prometheus_update,
        record_prometheus_buffer_cleared=_record_prometheus_buffer_cleared,
        record_prometheus_aggregation=_record_prometheus_aggregation,
        record_region_model_metrics=_record_region_model_metrics,
    )


def on_update(client, userdata, msg):
    """Handle local client updates and emit partial aggregates per region."""
    handle_client_update(
        client=client,
        msg=msg,
        config=_broker_config(),
        telemetry=_broker_telemetry(),
        callbacks=_broker_callbacks(),
        buffers=buffers,
        clients_per_region=clients_per_region,
        weighted_average_fn=weighted_average,
    )


def main():
    """Start SWEET fog broker MQTT loop."""
    global K, MQTT_BROKER, MQTT_PORT, UPDATE_TOPIC, PARTIAL_TOPIC, GLOBAL_TOPIC

    _init_telemetry()

    metrics_port = get_metrics_port_from_env(default=8001, component="SWEET_BROKER")
    start_metrics_server(port=metrics_port)

    parser = argparse.ArgumentParser(
        description="SWEET Fog broker (regional aggregator)"
    )
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
    K_MAP.clear()
    K_MAP.update(parse_k_map(args.k_map, broker_tag="[SWEET_FOG_BROKER]"))

    MQTT_BROKER = args.mqtt_broker
    MQTT_PORT = int(args.mqtt_port)
    UPDATE_TOPIC = args.topic_updates
    PARTIAL_TOPIC = args.topic_partial
    GLOBAL_TOPIC = args.topic_global

    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqttc.on_connect = lambda c, u, f, rc, p=None: c.subscribe(UPDATE_TOPIC)
    mqttc.on_message = on_update
    mqttc.connect(MQTT_BROKER, MQTT_PORT)
    print(
        f"[SWEET_FOG_BROKER] Broker fog iniciado. Escuchando actualizaciones en {UPDATE_TOPIC}"
    )
    print(
        f"[SWEET_FOG_BROKER] Agregando K={K} actualizaciones por región antes de enviar al servidor central"
    )

    def cleanup(*_args):
        print("[SWEET_FOG_BROKER] Shutting down telemetry...")
        shutdown_broker_runtime()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(0)))
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), exit(0)))

    try:
        mqttc.loop_forever()
    except KeyboardInterrupt:
        print("[SWEET_FOG_BROKER] Shutting down...")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
