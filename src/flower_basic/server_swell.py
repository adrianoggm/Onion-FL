from __future__ import annotations

"""
SWELL central server with MQTT publishing.
Uses SwellMLP to maintain parameter names consistent with clients.
"""

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

# Support running as a script (no package context)
try:
    from .datasets.swell_federated import load_node_split
    from .swell_model import SwellMLP
except Exception:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flower_basic.datasets.swell_federated import load_node_split
    from flower_basic.swell_model import SwellMLP

MODEL_TOPIC = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
TAG = "[SERVER_SWELL]"


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
        print(f"\n[SERVER_SWELL] === Round {server_round} ===")
        new_parameters = super().aggregate_fit(server_round, results, failures)
        if new_parameters is None:
            print(f"{TAG} Aggregation returned None")
            return None

        try:
            # Handle different return types across Flower versions
            parameters_obj = new_parameters
            if isinstance(new_parameters, tuple):
                # Older versions may return (parameters, metrics)
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

            payload = {
                "round": server_round,
                "global_weights": {k: v.tolist() for k, v in state_dict.items()},
            }
            if self.mqtt is not None:
                self.mqtt.publish(MODEL_TOPIC, json.dumps(payload))
                print(f"{TAG} Published global model on {MODEL_TOPIC}")

            # Optional central evaluation on global test split
            if self.eval_data is not None:
                try:
                    torch_state = {k: torch.tensor(v) for k, v in state_dict.items()}
                    self.global_model.load_state_dict(torch_state, strict=False)
                    loss, acc = _evaluate_global(self.global_model, self.eval_data)
                    print(f"{TAG} Global eval -> loss: {loss:.4f} | acc: {acc:.4f}")
                except Exception as eval_exc:  # pragma: no cover - defensive
                    print(f"{TAG} Global eval failed: {eval_exc}")
        except Exception as e:
            print(f"{TAG} MQTT publish failed: {e}")
        return new_parameters


def _load_eval_data(manifest_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load combined test split from all fog nodes for central evaluation."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = manifest.get("nodes", {})
    base = manifest_path.parent
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
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


def _evaluate_global(
    model: SwellMLP, data: Tuple[np.ndarray, np.ndarray]
) -> Tuple[float, float]:
    """Compute loss/accuracy on provided data tuple."""
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
    ap = argparse.ArgumentParser(description="SWELL central Flower server")
    ap.add_argument(
        "--input_dim", type=int, required=True, help="Feature dimension (from manifest)"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--server_addr", default="0.0.0.0:8080")
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument(
        "--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883"))
    )
    ap.add_argument("--topic-global", default=MODEL_TOPIC)
    ap.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest.json to enable central evaluation",
    )
    ap.add_argument(
        "--min-fit-clients",
        type=int,
        default=1,
        help="Min fog bridges required per round",
    )
    ap.add_argument(
        "--min-available-clients",
        type=int,
        default=None,
        help="Min fog bridges that must be connected",
    )
    args = ap.parse_args()

    model = SwellMLP(input_dim=args.input_dim)
    eval_data = _load_eval_data(Path(args.manifest)) if args.manifest else None
    if eval_data is None and args.manifest:
        print(f"{TAG} No test data found for central eval (manifest: {args.manifest})")

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


if __name__ == "__main__":
    main()
