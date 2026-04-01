from __future__ import annotations

"""
Fog bridge client for SWEET model.

Receives partial aggregates over MQTT (from sweet_fog broker) and forwards them
to the Flower server as a NumPyClient. Uses SweetMLP to keep parameter
names consistent for ordering.
"""

import argparse
import os

import flwr as fl

from flower_basic.clients.fog_bridge_base import BaseFogBridgeClient
from flower_basic.runtime_protocol import PartialAggregateEnvelope
from flower_basic.sweet_model import SweetMLP, get_parameters, set_parameters
from flower_basic.telemetry import (
    create_counter,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
)

# Telemetry - initialized lazily in main() to avoid import-time side effects
TRACER = None
METER = None
COUNTER_PARTIALS_RECEIVED = None
COUNTER_FORWARDS_TO_SERVER = None
COUNTER_TIMEOUTS = None
HIST_WAIT_TIME = None


def _init_telemetry():
    """Initialize telemetry. Called from main() to ensure proper service name."""
    global TRACER, METER
    global COUNTER_PARTIALS_RECEIVED, COUNTER_FORWARDS_TO_SERVER, COUNTER_TIMEOUTS, HIST_WAIT_TIME

    TRACER, METER = init_otel("sweet-fog-bridge")

    COUNTER_PARTIALS_RECEIVED = create_counter(
        METER,
        "fl_bridge_partials_received_total",
        "Partial aggregates received from fog",
    )
    COUNTER_FORWARDS_TO_SERVER = create_counter(
        METER, "fl_bridge_forwards_total", "Aggregates forwarded to central server"
    )
    COUNTER_TIMEOUTS = create_counter(
        METER, "fl_bridge_timeouts_total", "Timeouts waiting for partial aggregates"
    )
    HIST_WAIT_TIME = create_histogram(
        METER, "fl_bridge_wait_seconds", "Time waiting for partial aggregates", "s"
    )


MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
PARTIAL_TOPIC = os.getenv("MQTT_TOPIC_PARTIAL", "fl/partial")


class FogClientSweet(BaseFogBridgeClient):
    def __init__(
        self,
        server_address: str,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        region: str,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        partial_topic: str = PARTIAL_TOPIC,
    ):
        super().__init__(
            server_address=server_address,
            model=SweetMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
            ),
            get_parameters_fn=get_parameters,
            set_parameters_fn=set_parameters,
            region=region,
            tag=f"[SWEET_BRIDGE {region}]",
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            partial_topic=partial_topic,
            tracer=TRACER,
            partial_source_service="sweet-fog-broker",
            server_target_service="server-sweet",
        )

    def on_partial_received(self, envelope: PartialAggregateEnvelope) -> None:
        if COUNTER_PARTIALS_RECEIVED:
            record_metric(COUNTER_PARTIALS_RECEIVED, 1, {"region": self.region})

    def on_wait_completed(self, wait_duration: float) -> None:
        if HIST_WAIT_TIME:
            HIST_WAIT_TIME.record(wait_duration, {"region": self.region})

    def on_timeout(self) -> None:
        if COUNTER_TIMEOUTS:
            record_metric(COUNTER_TIMEOUTS, 1, {"region": self.region})

    def on_forwarded(self, num_samples: int) -> None:
        if COUNTER_FORWARDS_TO_SERVER:
            record_metric(COUNTER_FORWARDS_TO_SERVER, 1, {"region": self.region})


def main():
    _init_telemetry()

    ap = argparse.ArgumentParser(description="Fog bridge client for SWEET")
    ap.add_argument(
        "--input-dim", type=int, required=True, help="Feature dimension (from manifest)"
    )
    ap.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions",
    )
    ap.add_argument(
        "--num-classes", type=int, default=3, help="Number of output classes"
    )
    ap.add_argument("--server", default="localhost:8080")
    ap.add_argument(
        "--region",
        required=True,
        help="Fog region id this bridge represents (e.g., fog_0)",
    )
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-partial", default=PARTIAL_TOPIC)
    args = ap.parse_args()

    print("[SWEET_FOG_BRIDGE] Starting bridge client...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=FogClientSweet(
            args.server,
            args.input_dim,
            args.hidden_dims,
            args.num_classes,
            region=args.region,
            mqtt_broker=args.mqtt_broker,
            mqtt_port=args.mqtt_port,
            partial_topic=args.topic_partial,
        ),
    )

    shutdown_telemetry()


if __name__ == "__main__":
    main()
