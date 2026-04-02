"""SWEET central server with MQTT publishing.

Uses SweetMLP to maintain parameter names consistent with clients.
Follows SWELL architecture: Fog Bridges -> Central Server -> Global Model Broadcast
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt
import torch

from flower_basic.datasets.federated_common import load_manifest_eval_data
from flower_basic.datasets.sweet_federated import load_node_split
from flower_basic.prometheus_metrics import (
    get_metrics_port_from_env,
    push_metrics_to_gateway,
    start_metrics_server,
)
from flower_basic.servers.federated_base import FederatedMQTTStrategyBase
from flower_basic.sweet_model import SweetMLP
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    shutdown_telemetry,
)
from flower_basic.training.local import evaluate_classifier_arrays

MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
TAG = "[SERVER_SWEET]"

# Telemetry - initialized lazily in main() to avoid import-time side effects
TRACER = None
METER = None
COUNTER_ROUNDS = None
COUNTER_AGGREGATIONS = None
HIST_AGGREGATION_TIME = None
GAUGE_GLOBAL_ACCURACY = None
GAUGE_GLOBAL_LOSS = None
GAUGE_ACTIVE_CLIENTS = None


def _init_telemetry():
    """Initialize telemetry. Called from main() to ensure proper service name."""
    global TRACER, METER
    global COUNTER_ROUNDS, COUNTER_AGGREGATIONS, HIST_AGGREGATION_TIME
    global GAUGE_GLOBAL_ACCURACY, GAUGE_GLOBAL_LOSS, GAUGE_ACTIVE_CLIENTS

    TRACER, METER = init_otel("server-sweet")

    COUNTER_ROUNDS = create_counter(
        METER, "fl_rounds_total", "Total number of FL rounds completed"
    )
    COUNTER_AGGREGATIONS = create_counter(
        METER, "fl_aggregations_total", "Total number of model aggregations"
    )
    HIST_AGGREGATION_TIME = create_histogram(
        METER,
        "fl_aggregation_duration_seconds",
        "Time to aggregate client updates",
        "s",
    )
    GAUGE_GLOBAL_ACCURACY = create_gauge(
        METER, "fl_global_accuracy", "Current global model accuracy", "1"
    )
    GAUGE_GLOBAL_LOSS = create_gauge(
        METER, "fl_global_loss", "Current global model loss", "1"
    )
    GAUGE_ACTIVE_CLIENTS = create_gauge(
        METER, "fl_active_clients", "Number of active clients in current round", "1"
    )


class MQTTFedAvgSweet(FederatedMQTTStrategyBase):
    def __init__(
        self,
        model: SweetMLP,
        mqtt_client: mqtt.Client | None,
        eval_data: tuple[np.ndarray, np.ndarray] | None = None,
        total_rounds: int = 10,
        **kwargs,
    ):
        super().__init__(
            model=model,
            mqtt_client=mqtt_client,
            eval_data=eval_data,
            total_rounds=total_rounds,
            tracer=TRACER,
            model_topic=MODEL_TOPIC,
            publish_target_service="sweet-client",
            server_label="sweet",
            tag=TAG,
            **kwargs,
        )

    def before_aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> None:
        self.record_active_clients_measurement(
            server_round,
            num_results=len(results),
            active_clients_gauge=GAUGE_ACTIVE_CLIENTS,
        )

    def evaluate_aggregated_state(
        self, server_round: int, state_dict: dict[str, Any]
    ) -> tuple[float, float]:
        torch_state = {name: torch.tensor(value) for name, value in state_dict.items()}
        self.global_model.load_state_dict(torch_state, strict=False)
        return _evaluate_global(self.global_model, self.eval_data)

    def on_final_evaluation(
        self, server_round: int, evaluation: tuple[float, float]
    ) -> None:
        loss, acc = evaluation

        self.finalize_standard_evaluation(
            server_round,
            loss=loss,
            accuracy=acc,
            accuracy_gauge=GAUGE_GLOBAL_ACCURACY,
            loss_gauge=GAUGE_GLOBAL_LOSS,
        )

    def record_aggregation_metrics(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        aggregation_time: float,
    ) -> None:
        self.record_standard_round_metrics(
            server_round,
            num_results=len(results),
            aggregation_time=aggregation_time,
            rounds_counter=COUNTER_ROUNDS,
            aggregations_counter=COUNTER_AGGREGATIONS,
            aggregation_histogram=HIST_AGGREGATION_TIME,
        )


def _load_eval_data(manifest_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load test data from all nodes for centralized evaluation."""
    return load_manifest_eval_data(
        manifest_path,
        load_split=load_node_split,
        tag=TAG,
    )


def _evaluate_global(
    model: SweetMLP, data: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    """Evaluate global model on test data."""
    result = evaluate_classifier_arrays(model, data, batch_size=256)
    return result.loss, result.accuracy


def main():
    global MQTT_BROKER, MODEL_TOPIC

    _init_telemetry()

    metrics_port = get_metrics_port_from_env(default=8000, component="SERVER")
    start_metrics_server(port=metrics_port)

    ap = argparse.ArgumentParser(description="SWEET central Flower server")
    ap.add_argument("--input-dim", type=int, required=True, help="Feature dimension")
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--server-addr", default="0.0.0.0:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883"))
    )
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    ap.add_argument("--manifest", type=str, help="Path to manifest.json")
    ap.add_argument("--min-fit-clients", type=int, default=1)
    ap.add_argument("--min-available-clients", type=int, default=None)
    args = ap.parse_args()

    model = SweetMLP(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=args.num_classes,
    )
    eval_data = _load_eval_data(Path(args.manifest)) if args.manifest else None
    if eval_data is None and args.manifest:
        print(f"{TAG} No test data found for central eval")

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        MQTT_BROKER = args.mqtt_broker
        MODEL_TOPIC = args.topic_global
        mqtt_client.connect(MQTT_BROKER, args.mqtt_port)
        mqtt_client.loop_start()
        print(f"{TAG} Connected to MQTT at {MQTT_BROKER}")
    except Exception as exc:
        print(f"{TAG} MQTT connection failed: {exc}")
        mqtt_client = None

    min_available = (
        args.min_available_clients
        if args.min_available_clients is not None
        else args.min_fit_clients
    )
    strategy = MQTTFedAvgSweet(
        model=model,
        mqtt_client=mqtt_client,
        eval_data=eval_data,
        total_rounds=args.rounds,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=min_available,
    )

    print(f"{TAG} " + "=" * 62)
    print(f"{TAG}  SWEET Federated Learning Server (Transfer Learning)")
    print(f"{TAG}  - Model input_dim: {args.input_dim}")
    print(f"{TAG}  - Hidden dims: {args.hidden_dims}")
    print(f"{TAG}  - Num classes: {args.num_classes}")
    print(f"{TAG}  - Rounds: {args.rounds}")
    print(f"{TAG}  - Min clients: {args.min_fit_clients}")
    print(f"{TAG}  - Evaluation: {'ENABLED' if eval_data is not None else 'DISABLED'}")
    print(f"{TAG} " + "=" * 62)
    print(f"{TAG} Waiting for fog clients...")
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    push_metrics_to_gateway(job="sweet-server", grouping_key={"component": "server"})
    shutdown_telemetry()


if __name__ == "__main__":
    main()
