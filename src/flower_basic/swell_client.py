from __future__ import annotations

"""
SWELL federated local client over MQTT.

Loads per-node train/val/test NPZ splits prepared by
scripts/prepare_swell_federated.py and trains a small MLP locally.
Publishes updated weights to the fog broker over MQTT.
"""

import argparse
import json
import os
import time
from pathlib import Path
import threading

import numpy as np
import paho.mqtt.client as mqtt
import torch
from torch.utils.data import DataLoader, TensorDataset

# Support running as script: add src to path and import absolute package
try:
    from .datasets.swell_federated import load_node_split
    from .swell_model import SwellMLP
except Exception:  # pragma: no cover
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from flower_basic.datasets.swell_federated import load_node_split
    from flower_basic.swell_model import SwellMLP


MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_UPDATES = os.getenv("MQTT_TOPIC_UPDATES", "fl/updates")
TOPIC_GLOBAL_MODEL = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")


class SwellFLClientMQTT:
    def __init__(
        self,
        node_dir: str,
        region: str,
        lr: float = 1e-3,
        batch_size: int = 64,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        topic_updates: str = TOPIC_UPDATES,
        topic_global: str = TOPIC_GLOBAL_MODEL,
    ):
        self.node_dir = Path(node_dir)
        self.region = region
        self.tag = f"[CLIENT {self.region}]"
        self.topic_updates = topic_updates
        self.topic_global = topic_global

        # Load splits
        X_train, y_train, _ = load_node_split(self.node_dir / "train.npz")
        if X_train.size == 0:
            raise RuntimeError("Train split is empty for this node. Check subject assignments.")

        input_dim = X_train.shape[1]
        self.model = SwellMLP(input_dim=input_dim)

        # DataLoaders
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()),
            batch_size=batch_size,
            shuffle=True,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        # Optional validation split
        self.val_loader = None
        val_path = self.node_dir / "val.npz"
        if val_path.exists():
            try:
                X_val, y_val, _ = load_node_split(val_path)
                if X_val.size > 0:
                    self.val_loader = DataLoader(
                        TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()),
                        batch_size=256,
                        shuffle=False,
                    )
            except Exception:
                self.val_loader = None
        # Metrics output path
        self.metrics_path = self.node_dir / "val_metrics.jsonl"

        # MQTT
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(mqtt_broker, mqtt_port, keepalive=60)
        self.mqtt.loop_start()
        self._got_global = False
        # Protect model updates from MQTT callback during training
        self._lock = threading.Lock()
        self._pending_global_state = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe(self.topic_global)
        print(f"{self.tag} MQTT connected (rc={rc}). Subscribed to {self.topic_global}")

    def _on_message(self, client, userdata, msg):
        if msg.topic == self.topic_global:
            try:
                payload = json.loads(msg.payload.decode())
                weights = payload.get("global_weights")
                if weights:
                    # Buffer the global state to apply between rounds (avoid in-place during training)
                    state = {k: torch.tensor(v) for k, v in weights.items() if k in self.model.state_dict()}
                    with self._lock:
                        self._pending_global_state = state
                    self._got_global = True
                    print(f"{self.tag} Global model available (round={payload.get('round','?')})")
            except Exception as e:
                print(f"{self.tag} Error processing global model: {e}")

    def train_one_round(self) -> float:
        with self._lock:
            self.model.train()
            total = 0.0
            n = 0
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * X.size(0)
                n += X.size(0)
            avg_loss = total / max(n, 1)
        print(f"{self.tag} Train loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate_val(self) -> dict:
        if self.val_loader is None:
            return {}
        with self._lock:
            prev_mode = self.model.training
            self.model.eval()
            total = 0.0
            correct = 0
            count = 0
            with torch.no_grad():
                for X, y in self.val_loader:
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    total += loss.item() * X.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y).sum().item()
                    count += X.size(0)
            if prev_mode:
                self.model.train()
        if count == 0:
            return {}
        val_loss = total / count
        val_acc = correct / count
        print(f"{self.tag} Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}")
        return {"val_loss": val_loss, "val_acc": val_acc}

    def publish_update(self, avg_loss: float) -> None:
        with self._lock:
            state = self.model.state_dict()
            weights = {k: v.detach().cpu().numpy().tolist() for k, v in state.items()}
        payload = {
            "client_id": f"swell_client_{id(self)%10000}",
            "region": self.region,
            "weights": weights,
            "num_samples": sum(len(b[0]) for b in self.train_loader),
            "loss": float(avg_loss),
        }
        self.mqtt.publish(self.topic_updates, json.dumps(payload))
        print(f"{self.tag} Local update published to {self.topic_updates}")

    def wait_for_global(self, timeout_s: float = 30.0) -> bool:
        waited = 0.0
        self._got_global = False
        interval = 0.5
        while not self._got_global and waited < timeout_s:
            time.sleep(interval)
            waited += interval
        if not self._got_global:
            print(f"{self.tag} Timeout waiting for global model. Proceeding.")
        return self._got_global

    def run(self, rounds: int = 3, delay: float = 2.0) -> None:
        print(f"{self.tag} Starting {rounds} federated rounds (region={self.region})")
        for r in range(1, rounds + 1):
            print(f"\n=== Round {r}/{rounds} ===")
            avg_loss = self.train_one_round()
            # Validate before publishing update
            val_metrics = self.evaluate_val()
            # Persist metrics
            try:
                rec = {"round": r, "region": self.region, "train_loss": float(avg_loss)}
                rec.update({k: float(v) for k, v in val_metrics.items()})
                import json as _json
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec) + "\n")
            except Exception:
                pass
            self.publish_update(avg_loss)
            self.wait_for_global()
            # Apply pending global safely between rounds
            with self._lock:
                if self._pending_global_state is not None:
                    current = self.model.state_dict()
                    current.update(self._pending_global_state)
                    self.model.load_state_dict(current, strict=False)
                    self._pending_global_state = None
                    print(f"{self.tag} Global model applied")
            if r < rounds:
                time.sleep(delay)
        self.mqtt.loop_stop()
        self.mqtt.disconnect()


def main():
    ap = argparse.ArgumentParser(description="SWELL MQTT federated local client")
    ap.add_argument("--node_dir", required=True, help="Path to node directory with splits (train/val/test npz)")
    ap.add_argument("--region", required=True, help="Region name to tag updates (e.g., fog_0)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-updates", default=TOPIC_UPDATES)
    ap.add_argument("--topic-global", default=TOPIC_GLOBAL_MODEL)
    args = ap.parse_args()

    client = SwellFLClientMQTT(
        args.node_dir,
        args.region,
        lr=args.lr,
        batch_size=args.batch_size,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
        topic_updates=args.topic_updates,
        topic_global=args.topic_global,
    )
    client.run(rounds=args.rounds)


if __name__ == "__main__":
    main()
