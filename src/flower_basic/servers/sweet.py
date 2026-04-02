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
    FL_ACCURACY,
    FL_ACTIVE_CLIENTS,
    FL_AGGREGATIONS,
    FL_LOSS,
    FL_ROUND_DURATION,
    FL_ROUNDS,
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
    record_metric,
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
            tag=TAG,
            **kwargs,
        )

    def describe_eval_data(self) -> str | None:
        if self.eval_data is None:
            return None
        features, labels = self.eval_data
        unique_labels = np.unique(labels)
        return f"{features.shape[0]} samples, {len(unique_labels)} classes"

    def before_aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> None:
        if GAUGE_ACTIVE_CLIENTS:
            record_metric(
                GAUGE_ACTIVE_CLIENTS, len(results), {"round": str(server_round)}
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

        self.history["round"].append(server_round)
        self.history["loss"].append(loss)
        self.history["accuracy"].append(acc)

        if GAUGE_GLOBAL_ACCURACY:
            record_metric(GAUGE_GLOBAL_ACCURACY, acc, {"round": str(server_round)})
        if GAUGE_GLOBAL_LOSS:
            record_metric(GAUGE_GLOBAL_LOSS, loss, {"round": str(server_round)})

        FL_ACCURACY.labels(server="sweet").set(acc)
        FL_LOSS.labels(server="sweet").set(loss)
        self._print_final_evaluation(loss, acc)

    def record_aggregation_metrics(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        aggregation_time: float,
    ) -> None:
        if COUNTER_AGGREGATIONS:
            record_metric(COUNTER_AGGREGATIONS, 1, {"round": str(server_round)})
        if COUNTER_ROUNDS:
            record_metric(COUNTER_ROUNDS, 1)
        if HIST_AGGREGATION_TIME:
            HIST_AGGREGATION_TIME.record(aggregation_time, {"round": str(server_round)})

        FL_ROUNDS.labels(server="sweet").inc()
        FL_AGGREGATIONS.labels(server="sweet").inc()
        FL_ACTIVE_CLIENTS.labels(server="sweet").set(len(results))
        FL_ROUND_DURATION.labels(server="sweet").observe(aggregation_time)

    def _print_final_evaluation(self, loss: float, acc: float):
        """Print final evaluation results after federated learning completes."""
        features, labels = self.eval_data
        n_samples = features.shape[0]
        n_classes = len(np.unique(labels))

        print(
            f"\n{TAG} ╔══════════════════════════════════════════════════════════════╗"
        )
        print(f"{TAG} ║     FEDERATED LEARNING COMPLETE - FINAL MODEL EVALUATION     ║")
        print(f"{TAG} ╠══════════════════════════════════════════════════════════════╣")
        print(f"{TAG} ║                                                              ║")
        print(f"{TAG} ║  Test Dataset:                                               ║")
        print(
            f"{TAG} ║    - Samples: {n_samples:>10,}                                   ║"
        )
        print(
            f"{TAG} ║    - Classes: {n_classes:>10}                                   ║"
        )
        print(f"{TAG} ║                                                              ║")
        print(f"{TAG} ║  Model Performance:                                          ║")
        print(f"{TAG} ║    ┌────────────────────────────────────────────────┐        ║")
        print(
            f"{TAG} ║    │  Loss:     {loss:>10.4f}                        │        ║"
        )
        print(
            f"{TAG} ║    │  Accuracy: {acc:>10.4f}  ({acc*100:>6.2f}%)            │        ║"
        )
        print(f"{TAG} ║    └────────────────────────────────────────────────┘        ║")
        print(f"{TAG} ║                                                              ║")

        bar_len = 40
        filled = int(acc * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"{TAG} ║  Accuracy: [{bar}]  ║")
        print(f"{TAG} ║             0%                                   100%        ║")
        print(f"{TAG} ║                                                              ║")
        print(
            f"{TAG} ╚══════════════════════════════════════════════════════════════╝\n"
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
