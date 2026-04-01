from __future__ import annotations

"""
Fog bridge client for SWELL model.

Receives partial aggregates over MQTT (from broker_fog) and forwards them
to the Flower server as a NumPyClient. Uses SwellMLP to keep parameter
names consistent for ordering.
"""

import argparse
import os

import flwr as fl

from flower_basic.clients.fog_bridge_base import BaseFogBridgeClient
from flower_basic.logging_utils import enable_timestamped_print
from flower_basic.runtime_protocol import PartialAggregateEnvelope
from flower_basic.swell_model import SwellMLP, get_parameters, set_parameters
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

    TRACER, METER = init_otel("fog-bridge")

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


class FogClientSwell(BaseFogBridgeClient):
    def __init__(
        self,
        server_address: str,
        input_dim: int,
        region: str,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        partial_topic: str = PARTIAL_TOPIC,
    ):
        super().__init__(
            server_address=server_address,
            model=SwellMLP(input_dim=input_dim),
            get_parameters_fn=get_parameters,
            set_parameters_fn=set_parameters,
            region=region,
            tag=f"[BRIDGE {region}]",
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            partial_topic=partial_topic,
            tracer=TRACER,
            partial_source_service="fog-broker",
            server_target_service="server-swell",
        )

    def build_partial_metadata(
        self, envelope: PartialAggregateEnvelope
    ) -> dict[str, object]:
        return {
            "expected_round": envelope.expected_round or 0,
            "round_min": envelope.round_min or 0,
            "round_max": envelope.round_max or 0,
            "stale_update_count": envelope.stale_update_count,
            "future_update_count": envelope.future_update_count,
            "max_delay_seconds": envelope.max_delay_seconds,
            "mean_delay_seconds": envelope.mean_delay_seconds,
            "stale_policy": envelope.stale_policy or "accept",
        }

    def build_timeout_metrics(self) -> dict[str, object]:
        return {"timeout": True, "region": self.region}

    def on_partial_received(self, envelope: PartialAggregateEnvelope) -> None:
        record_metric(COUNTER_PARTIALS_RECEIVED, 1, {"region": self.region})

    def on_wait_completed(self, wait_duration: float) -> None:
        if HIST_WAIT_TIME:
            HIST_WAIT_TIME.record(wait_duration, {"region": self.region})

    def on_timeout(self) -> None:
        record_metric(COUNTER_TIMEOUTS, 1, {"region": self.region})

    def on_forwarded(self, num_samples: int) -> None:
        record_metric(COUNTER_FORWARDS_TO_SERVER, 1, {"region": self.region})


def main():
    enable_timestamped_print()
    _init_telemetry()

    ap = argparse.ArgumentParser(description="Fog bridge client for SWELL")
    ap.add_argument(
        "--input_dim", type=int, required=True, help="Feature dimension (from manifest)"
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

    shutdown_telemetry()


if __name__ == "__main__":
    main()
