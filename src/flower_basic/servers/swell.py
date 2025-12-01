"""SWELL central server with MQTT publishing.

Uses SwellMLP to maintain parameter names consistent with clients.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_basic.telemetry import (
    create_counter,
    create_histogram,
    create_gauge,
    record_metric,
    init_otel,
    start_span,
    shutdown_telemetry,
)
from flower_basic.datasets.swell_federated import load_node_split
from flower_basic.swell_model import SwellMLP

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
        METER, "fl_aggregation_duration_seconds", "Time to aggregate client updates", "s"
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
        mqtt_client: Optional[mqtt.Client],
        eval_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_model = model
        self.mqtt = mqtt_client
        self.param_names = list(model.state_dict().keys())
        self.eval_data = eval_data

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[Any, fl.common.FitRes]],
        failures,
    ) -> Optional[fl.common.Parameters]:
        import time
        start_time = time.time()
        
        print(f"\n[SERVER_SWELL] === Round {server_round} ===")
        
        # Record number of active clients
        record_metric(GAUGE_ACTIVE_CLIENTS, len(results), {"round": str(server_round)})
        
        with start_span(TRACER, "server.aggregate_fit", {"round": server_round, "num_results": len(results)}):
            new_parameters = super().aggregate_fit(server_round, results, failures)

        if new_parameters is None:
            print(f"{TAG} Aggregation returned None")
            return None

        try:
            parameters_obj = new_parameters
            if isinstance(new_parameters, tuple):
                parameters_obj = new_parameters[0]

            if hasattr(parameters_obj, "tensors"):
                param_arrays = [fl.common.bytes_to_ndarray(t) for t in parameters_obj.tensors]
            else:
                param_arrays = fl.common.parameters_to_ndarrays(parameters_obj)

            state_dict = {}
            for i, name in enumerate(self.param_names):
                if i < len(param_arrays):
                    state_dict[name] = param_arrays[i]

            payload = {"round": server_round, "global_weights": {k: v.tolist() for k, v in state_dict.items()}}
            if self.mqtt is not None:
                self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))
                print(f"{TAG} Published global model on {MODEL_TOPIC}")

            if self.eval_data is not None:
                try:
                    torch_state = {k: torch.tensor(v) for k, v in state_dict.items()}
                    self.global_model.load_state_dict(torch_state, strict=False)
                    loss, acc = _evaluate_global(self.global_model, self.eval_data)
                    print(f"{TAG} Global eval -> loss: {loss:.4f} | acc: {acc:.4f}")
                    
                    # Record global metrics
                    record_metric(GAUGE_GLOBAL_ACCURACY, acc, {"round": str(server_round)})
                    record_metric(GAUGE_GLOBAL_LOSS, loss, {"round": str(server_round)})
                except Exception as eval_exc:
                    print(f"{TAG} Global eval failed: {eval_exc}")
        except Exception as e:
            print(f"{TAG} MQTT publish failed: {e}")

        # Record aggregation metrics
        aggregation_time = time.time() - start_time
        record_metric(COUNTER_AGGREGATIONS, 1, {"round": str(server_round)})
        record_metric(COUNTER_ROUNDS, 1)
        if HIST_AGGREGATION_TIME:
            HIST_AGGREGATION_TIME.record(aggregation_time, {"round": str(server_round)})
        
        return new_parameters


def _load_eval_data(manifest_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    Xs = []
    ys = []
    for node_id in nodes.keys():
        test_path = base / node_id / "test.npz"
        if not test_path.exists():
            continue
        X, y, _ = load_node_split(test_path)
        if X.size > 0:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return None
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _evaluate_global(model: SwellMLP, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
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
    
    ap = argparse.ArgumentParser(description="SWELL central Flower server")
    ap.add_argument("--input_dim", type=int, required=True, help="Feature dimension")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
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

    min_available = args.min_available_clients if args.min_available_clients is not None else args.min_fit_clients
    strategy = MQTTFedAvgSwell(
        model=model,
        mqtt_client=mqtt_client,
        eval_data=eval_data,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=0,
        min_available_clients=min_available,
    )

    print(f"{TAG} Waiting for fog clients...")
    fl.server.start_server(
        server_address=args.server_addr,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    # Ensure all telemetry is flushed before exit
    shutdown_telemetry()


if __name__ == "__main__":
    main()
