"""SWELL central server with MQTT publishing.

Uses SwellMLP to maintain parameter names consistent with clients.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_basic.datasets.swell_federated import load_node_split
from flower_basic.swell_model import SwellMLP
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_span,
    start_server_span,
    start_producer_span,
    start_linked_producer_span,
    inject_trace_context,
    SpanKind,
)
from flower_basic.prometheus_metrics import (
    start_metrics_server,
    FL_ROUNDS,
    FL_ACCURACY,
    FL_LOSS,
    FL_ACTIVE_CLIENTS,
    FL_AGGREGATIONS,
    FL_ROUND_DURATION,
    get_metrics_port_from_env,
    push_metrics_to_gateway,
)

MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
TAG = "[SERVER_SWELL]"

# Telemetry - initialized lazily in main() to avoid import-time side effects
TRACER = None
METER = None
# Metrics
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

    TRACER, METER = init_otel("server-swell")

    # Counters
    COUNTER_ROUNDS = create_counter(
        METER, "fl_rounds_total", "Total number of FL rounds completed"
    )
    COUNTER_AGGREGATIONS = create_counter(
        METER, "fl_aggregations_total", "Total number of model aggregations"
    )

    # Histograms
    HIST_AGGREGATION_TIME = create_histogram(
        METER,
        "fl_aggregation_duration_seconds",
        "Time to aggregate client updates",
        "s",
    )

    # Gauges (using UpDownCounter)
    GAUGE_GLOBAL_ACCURACY = create_gauge(
        METER, "fl_global_accuracy", "Current global model accuracy", "1"
    )
    GAUGE_GLOBAL_LOSS = create_gauge(
        METER, "fl_global_loss", "Current global model loss", "1"
    )
    GAUGE_ACTIVE_CLIENTS = create_gauge(
        METER, "fl_active_clients", "Number of active clients in current round", "1"
    )


class MQTTFedAvgSwell(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model: SwellMLP,
        mqtt_client: mqtt.Client | None,
        eval_data: tuple[np.ndarray, np.ndarray] | None = None,
        total_rounds: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_model = model
        self.mqtt = mqtt_client
        self.param_names = list(model.state_dict().keys())
        self.eval_data = eval_data
        self.total_rounds = total_rounds
        self.history = {"round": [], "loss": [], "accuracy": []}

        # Log evaluation data status
        if self.eval_data is not None:
            X, y = self.eval_data
            unique_labels = np.unique(y)
            print(
                f"{TAG} Evaluation data loaded: {X.shape[0]} samples, {len(unique_labels)} classes"
            )
        else:
            print(
                f"{TAG} WARNING: No evaluation data - model quality will NOT be verified!"
            )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> fl.common.Parameters | None:
        import time

        start_time = time.time()

        print(f"\n{TAG} ╔══════════════════════════════════════════╗")
        print(
            f"{TAG} ║           ROUND {server_round}/{self.total_rounds}                        ║"
        )
        print(f"{TAG} ╚══════════════════════════════════════════╝")
        print(
            f"{TAG} Received results from {len(results)} fog nodes, {len(failures)} failures"
        )

        # Record number of active clients
        record_metric(GAUGE_ACTIVE_CLIENTS, len(results), {"round": str(server_round)})

        # Use SERVER span to show this service handles incoming requests from fog-bridge
        with start_server_span(
            TRACER,
            "server.aggregate_fit",
            attributes={"round": server_round, "num_results": len(results)},
        ):
            new_parameters = super().aggregate_fit(server_round, results, failures)

        if new_parameters is None:
            print(f"{TAG} Aggregation returned None")
            return None

        try:
            parameters_obj = new_parameters
            if isinstance(new_parameters, tuple):
                parameters_obj = new_parameters[0]

            if hasattr(parameters_obj, "tensors"):
                param_arrays = [
                    fl.common.bytes_to_ndarray(t) for t in parameters_obj.tensors
                ]
            else:
                param_arrays = fl.common.parameters_to_ndarrays(parameters_obj)

            state_dict = {}
            for i, name in enumerate(self.param_names):
                if i < len(param_arrays):
                    state_dict[name] = param_arrays[i]

            if self.mqtt is not None:
                # Use PRODUCER span with context propagation to clients
                with start_linked_producer_span(
                    TRACER,
                    "server.publish_global_model",
                    target_service="swell-client",
                    attributes={"round": server_round},
                ) as (span, trace_ctx):
                    payload = {
                        "round": server_round,
                        "global_weights": {
                            k: v.tolist() for k, v in state_dict.items()
                        },
                        "trace_context": trace_ctx,  # Propagate trace context
                    }
                    self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))
                print(f"{TAG} Published global model on {MODEL_TOPIC}")

            # Only evaluate on the FINAL round after all federated aggregation is complete
            if server_round == self.total_rounds:
                if self.eval_data is not None:
                    try:
                        torch_state = {
                            k: torch.tensor(v) for k, v in state_dict.items()
                        }
                        self.global_model.load_state_dict(torch_state, strict=False)
                        loss, acc = _evaluate_global(self.global_model, self.eval_data)

                        # Store final metrics
                        self.history["round"].append(server_round)
                        self.history["loss"].append(loss)
                        self.history["accuracy"].append(acc)

                        # Record global metrics for telemetry (OTEL)
                        record_metric(
                            GAUGE_GLOBAL_ACCURACY, acc, {"round": str(server_round)}
                        )
                        record_metric(
                            GAUGE_GLOBAL_LOSS, loss, {"round": str(server_round)}
                        )

                        # Record Prometheus metrics
                        FL_ACCURACY.labels(server="swell").set(acc)
                        FL_LOSS.labels(server="swell").set(loss)

                        # Print final evaluation summary
                        self._print_final_evaluation(loss, acc)
                    except Exception as eval_exc:
                        print(f"{TAG} Final evaluation failed: {eval_exc}")
                        import traceback

                        traceback.print_exc()
                else:
                    print(
                        f"{TAG} ⚠ FINAL ROUND: No test data available for evaluation!"
                    )
            else:
                print(
                    f"{TAG} Round {server_round}/{self.total_rounds} complete. Evaluation will run after final round."
                )
        except Exception as e:
            print(f"{TAG} MQTT publish failed: {e}")

        # Record aggregation metrics (OTEL)
        aggregation_time = time.time() - start_time
        record_metric(COUNTER_AGGREGATIONS, 1, {"round": str(server_round)})
        record_metric(COUNTER_ROUNDS, 1)
        if HIST_AGGREGATION_TIME:
            HIST_AGGREGATION_TIME.record(aggregation_time, {"round": str(server_round)})

        # Record Prometheus metrics
        FL_ROUNDS.labels(server="swell").inc()
        FL_AGGREGATIONS.labels(server="swell").inc()
        FL_ACTIVE_CLIENTS.labels(server="swell").set(len(results))
        FL_ROUND_DURATION.labels(server="swell").observe(aggregation_time)

        return new_parameters

    def _print_final_evaluation(self, loss: float, acc: float):
        """Print final evaluation results after federated learning completes."""
        X, y = self.eval_data
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

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

        # Visual accuracy bar
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
    print(f"{TAG} Loading evaluation data from manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    Xs = []
    ys = []
    for node_id in nodes.keys():
        test_path = base / node_id / "test.npz"
        if not test_path.exists():
            print(f"{TAG}   Node {node_id}: test.npz NOT FOUND at {test_path}")
            continue
        X, y, _ = load_node_split(test_path)
        if X.size > 0:
            Xs.append(X)
            ys.append(y)
            print(f"{TAG}   Node {node_id}: loaded {X.shape[0]} test samples")
        else:
            print(f"{TAG}   Node {node_id}: test.npz is empty")
    if not Xs:
        print(f"{TAG} WARNING: No test data found for centralized evaluation!")
        return None
    total_X = np.concatenate(Xs, axis=0)
    total_y = np.concatenate(ys, axis=0)
    print(
        f"{TAG} Total evaluation data: {total_X.shape[0]} samples, {total_X.shape[1]} features"
    )
    return total_X, total_y


def _evaluate_global(
    model: SwellMLP, data: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float]:
    X, y = data
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    total = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            count += xb.size(0)
    loss = total / max(count, 1)
    acc = correct / max(count, 1)
    return loss, acc


def main():
    global MQTT_BROKER, MODEL_TOPIC

    # Initialize telemetry for this service
    _init_telemetry()

    # Start Prometheus metrics server
    metrics_port = get_metrics_port_from_env(default=8000, component="SERVER")
    start_metrics_server(port=metrics_port)

    ap = argparse.ArgumentParser(description="SWELL central Flower server")
    ap.add_argument("--input_dim", type=int, required=True, help="Feature dimension")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883"))
    )
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    ap.add_argument("--manifest", type=str, help="Path to manifest.json")
    ap.add_argument("--min-fit-clients", type=int, default=1)
    ap.add_argument("--min-available-clients", type=int, default=None)
    args = ap.parse_args()

    model = SwellMLP(input_dim=args.input_dim)
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
    except Exception as e:
        print(f"{TAG} MQTT connection failed: {e}")
        mqtt_client = None

    min_available = (
        args.min_available_clients
        if args.min_available_clients is not None
        else args.min_fit_clients
    )
    strategy = MQTTFedAvgSwell(
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

    print(f"{TAG} ══════════════════════════════════════════════════════════════")
    print(f"{TAG}  SWELL Federated Learning Server")
    print(f"{TAG}  - Model input_dim: {args.input_dim}")
    print(f"{TAG}  - Rounds: {args.rounds}")
    print(f"{TAG}  - Min clients: {args.min_fit_clients}")
    print(f"{TAG}  - Evaluation: {'ENABLED' if eval_data is not None else 'DISABLED'}")
    print(f"{TAG} ══════════════════════════════════════════════════════════════")
    print(f"{TAG} Waiting for fog clients...")
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Push metrics to Pushgateway before exit (ensures metrics persist)
    push_metrics_to_gateway(job="flower-server", grouping_key={"component": "server"})

    # Ensure all telemetry is flushed before exit
    shutdown_telemetry()


if __name__ == "__main__":
    main()
