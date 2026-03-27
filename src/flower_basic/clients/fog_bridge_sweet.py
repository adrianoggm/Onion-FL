from __future__ import annotations

"""
Fog bridge client for SWEET model.

Receives partial aggregates over MQTT (from sweet_fog broker) and forwards them
to the Flower server as a NumPyClient. Uses SweetMLP to keep parameter
names consistent for ordering.
"""

import argparse
import os
import time

import flwr as fl
import numpy as np

from flower_basic.clients.baseclient import BaseMQTTComponent
from flower_basic.runtime_protocol import decode_partial_aggregate_message
from flower_basic.sweet_model import SweetMLP, get_parameters, set_parameters
from flower_basic.telemetry import (
    create_counter,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_linked_client_span,
    start_linked_consumer_span,
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


class FogClientSweet(BaseMQTTComponent, fl.client.NumPyClient):
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
        self.server_address = server_address
        self.model = SweetMLP(
            input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes
        )
        self.param_names = list(self.model.state_dict().keys())
        self.partial_weights = None
        self.partial_trace_context = None  # Store trace context from partial
        self.partial_topic = partial_topic
        self.region = region
        self.tag = f"[SWEET_BRIDGE {self.region}]"
        super().__init__(
            tag=self.tag,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            subscriptions=[self.partial_topic],
        )
        print(f"{self.tag} Listening for partials on {self.partial_topic}")

    def on_message(self, client, userdata, msg):
        try:
            envelope = decode_partial_aggregate_message(msg.payload)
            if envelope is None:
                print(f"{self.tag} Ignoring malformed partial payload")
                return
            if envelope.region != self.region:
                return
            self.partial_weights = envelope.weights
            self.partial_trace_context = envelope.trace_context
            # Use linked CONSUMER span to continue trace from sweet-fog-broker
            with start_linked_consumer_span(
                TRACER,
                "bridge.receive_partial",
                self.partial_trace_context,
                source_service="sweet-fog-broker",
                attributes={"region": envelope.region},
            ):
                if COUNTER_PARTIALS_RECEIVED:
                    record_metric(COUNTER_PARTIALS_RECEIVED, 1, {"region": self.region})
                print(
                    f"{self.tag} Partial aggregate received for region={envelope.region}"
                )
        except Exception as e:
            print(f"{self.tag} Error processing partial: {e}")

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config):
        start_wait = time.time()

        # Use linked CLIENT span to continue the trace and show dependency to server-sweet
        # This links from the sweet-fog-broker trace through to the server
        with start_linked_client_span(
            TRACER,
            "bridge.forward_to_server",
            "server-sweet",
            trace_context=self.partial_trace_context,
            attributes={"region": self.region},
        ) as span:
            set_parameters(self.model, parameters)
            timeout = 60
            waited = 0
            while self.partial_weights is None and waited < timeout:
                time.sleep(0.5)
                waited += 0.5

            wait_duration = time.time() - start_wait
            if HIST_WAIT_TIME:
                HIST_WAIT_TIME.record(wait_duration, {"region": self.region})

            if self.partial_weights is None:
                print(f"{self.tag} Timeout waiting for partial")
                if span:
                    span.set_attribute("status", "timeout")
                if COUNTER_TIMEOUTS:
                    record_metric(COUNTER_TIMEOUTS, 1, {"region": self.region})
                return get_parameters(self.model), 1, {}

            partial_list = [
                np.array(self.partial_weights[name], dtype=np.float32)
                for name in self.param_names
            ]
            self.partial_weights = None
            self.partial_trace_context = None  # Clear trace context after use
            num_samples = 1000
            print(f"{self.tag} Forwarding partial to central server")
            if span:
                span.set_attribute("status", "forwarded")
                span.set_attribute("num_samples", num_samples)
            if COUNTER_FORWARDS_TO_SERVER:
                record_metric(COUNTER_FORWARDS_TO_SERVER, 1, {"region": self.region})
            return partial_list, num_samples, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


def main():
    # Initialize telemetry for this service
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

    # Ensure all telemetry is flushed before exit
    shutdown_telemetry()


if __name__ == "__main__":
    main()
