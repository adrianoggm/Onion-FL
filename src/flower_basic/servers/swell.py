"""SWELL central server with MQTT publishing.

Uses SwellMLP to maintain parameter names consistent with clients.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_basic.datasets.swell_federated import load_node_split
from flower_basic.logging_utils import enable_timestamped_print
from flower_basic.prometheus_metrics import (
    FL_ACCURACY,
    FL_ACTIVE_CLIENTS,
    FL_AGGREGATIONS,
    FL_CONFUSION_MATRIX,
    FL_GLOBAL_TEST_SAMPLES,
    FL_GLOBAL_TRAIN_SAMPLES,
    FL_GLOBAL_VAL_SAMPLES,
    FL_LOSS,
    FL_ROUND_DURATION,
    FL_ROUNDS,
    get_metrics_port_from_env,
    push_metrics_to_gateway,
    start_metrics_server,
)
from flower_basic.runtime_protocol import summarize_staleness_metrics
from flower_basic.servers.federated_base import FederatedMQTTStrategyBase
from flower_basic.swell_model import SwellMLP
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
)

MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
TAG = "[SERVER_SWELL]"

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

    TRACER, METER = init_otel("server-swell")

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


class MQTTFedAvgSwell(FederatedMQTTStrategyBase):
    def __init__(
        self,
        model: SwellMLP,
        mqtt_client: mqtt.Client | None,
        eval_data: tuple[np.ndarray, np.ndarray] | None = None,
        total_rounds: int = 3,
        **kwargs,
    ):
        super().__init__(
            model=model,
            mqtt_client=mqtt_client,
            eval_data=eval_data,
            total_rounds=total_rounds,
            tracer=TRACER,
            model_topic=MODEL_TOPIC,
            publish_target_service="swell-client",
            tag=TAG,
            **kwargs,
        )

    def initialize_history(self) -> dict[str, list[Any]]:
        return {
            "round": [],
            "loss": [],
            "accuracy": [],
            "stale_updates": [],
            "future_updates": [],
            "max_delay_seconds": [],
            "round_span": [],
        }

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
        staleness = summarize_staleness_metrics(
            [getattr(fit_res, "metrics", {}) for _client_proxy, fit_res in results],
            server_round=server_round,
        )

        print(
            f"{TAG} Staleness summary: stale_updates={staleness.stale_updates}, "
            f"future_updates={staleness.future_updates}, max_round_span={staleness.max_round_span}, "
            f"max_delay={staleness.max_delay_seconds:.2f}s"
        )

        self.history["stale_updates"].append(staleness.stale_updates)
        self.history["future_updates"].append(staleness.future_updates)
        self.history["max_delay_seconds"].append(staleness.max_delay_seconds)
        self.history["round_span"].append(staleness.max_round_span)
        record_metric(GAUGE_ACTIVE_CLIENTS, len(results), {"round": str(server_round)})

    def evaluate_aggregated_state(
        self, server_round: int, state_dict: dict[str, Any]
    ) -> tuple[float, float, np.ndarray, list[int]]:
        torch_state = {name: torch.tensor(value) for name, value in state_dict.items()}
        self.global_model.load_state_dict(torch_state, strict=False)
        return _evaluate_global(self.global_model, self.eval_data)

    def on_final_evaluation(
        self,
        server_round: int,
        evaluation: tuple[float, float, np.ndarray, list[int]],
    ) -> None:
        loss, acc, confusion_matrix, labels = evaluation

        self.history["round"].append(server_round)
        self.history["loss"].append(loss)
        self.history["accuracy"].append(acc)

        record_metric(GAUGE_GLOBAL_ACCURACY, acc, {"round": str(server_round)})
        record_metric(GAUGE_GLOBAL_LOSS, loss, {"round": str(server_round)})

        FL_ACCURACY.labels(server="swell").set(acc)
        FL_LOSS.labels(server="swell").set(loss)
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                FL_CONFUSION_MATRIX.labels(
                    server="swell",
                    true_label=str(true_label),
                    pred_label=str(pred_label),
                ).set(int(confusion_matrix[true_index, pred_index]))

        self._print_final_evaluation(loss, acc)

    def record_aggregation_metrics(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        aggregation_time: float,
    ) -> None:
        record_metric(COUNTER_AGGREGATIONS, 1, {"round": str(server_round)})
        record_metric(COUNTER_ROUNDS, 1)
        if HIST_AGGREGATION_TIME:
            HIST_AGGREGATION_TIME.record(aggregation_time, {"round": str(server_round)})

        FL_ROUNDS.labels(server="swell").inc()
        FL_AGGREGATIONS.labels(server="swell").inc()
        FL_ACTIVE_CLIENTS.labels(server="swell").set(len(results))
        FL_ROUND_DURATION.labels(server="swell").observe(aggregation_time)

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
    print(f"{TAG} Loading evaluation data from manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    all_features = []
    all_labels = []

    for node_id in nodes.keys():
        test_path = base / node_id / "test.npz"
        if not test_path.exists():
            print(f"{TAG}   Node {node_id}: test.npz NOT FOUND at {test_path}")
            continue

        features, labels, _ = load_node_split(test_path)
        if features.size == 0:
            print(f"{TAG}   Node {node_id}: test.npz is empty")
            continue

        all_features.append(features)
        all_labels.append(labels)
        print(f"{TAG}   Node {node_id}: loaded {features.shape[0]} test samples")

    if not all_features:
        print(f"{TAG} WARNING: No test data found for centralized evaluation!")
        return None

    total_features = np.concatenate(all_features, axis=0)
    total_labels = np.concatenate(all_labels, axis=0)
    print(
        f"{TAG} Total evaluation data: {total_features.shape[0]} samples, "
        f"{total_features.shape[1]} features"
    )
    return total_features, total_labels


def _load_manifest_split_counts(manifest_path: Path) -> tuple[int, int, int]:
    """Load total train/val/test sample counts from aggregated node splits."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    totals = {"train": 0, "val": 0, "test": 0}

    for node_id in nodes.keys():
        for split_name in ("train", "val", "test"):
            split_path = base / node_id / f"{split_name}.npz"
            if not split_path.exists():
                continue
            features, _, _ = load_node_split(split_path)
            totals[split_name] += int(features.shape[0])

    return totals["train"], totals["val"], totals["test"]


def _evaluate_global(
    model: SwellMLP, data: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float, np.ndarray, list[int]]:
    features, labels = data
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    dataset = TensorDataset(
        torch.from_numpy(features).float(), torch.from_numpy(labels).long()
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    count = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            total_loss += float(loss.item()) * int(batch_features.size(0))
            predictions = torch.argmax(logits, dim=1)
            correct += int((predictions == batch_labels).sum().item())
            count += int(batch_features.size(0))
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())

    loss = total_loss / max(count, 1)
    accuracy = correct / max(count, 1)
    y_true = torch.cat(all_labels).numpy() if all_labels else np.array([], dtype=int)
    y_pred = (
        torch.cat(all_predictions).numpy() if all_predictions else np.array([], dtype=int)
    )
    label_values = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not label_values:
        label_values = [0, 1]

    label_index = {label: idx for idx, label in enumerate(label_values)}
    confusion_matrix = np.zeros((len(label_values), len(label_values)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion_matrix[label_index[true_label], label_index[pred_label]] += 1

    return loss, accuracy, confusion_matrix, label_values


def main():
    global MQTT_BROKER, MODEL_TOPIC

    enable_timestamped_print()
    _init_telemetry()

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
    if args.manifest:
        train_n, val_n, test_n = _load_manifest_split_counts(Path(args.manifest))
        FL_GLOBAL_TRAIN_SAMPLES.labels(server="swell").set(train_n)
        FL_GLOBAL_VAL_SAMPLES.labels(server="swell").set(val_n)
        FL_GLOBAL_TEST_SAMPLES.labels(server="swell").set(test_n)
        print(
            f"{TAG} Manifest split totals: train={train_n}, val={val_n}, test={test_n}"
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

    push_metrics_to_gateway(job="flower-server", grouping_key={"component": "server"})
    shutdown_telemetry()


if __name__ == "__main__":
    main()
