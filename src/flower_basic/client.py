"""
Local federated client over MQTT (WESAD/ECG example).

Trains a CNN locally, publishes updates to the fog broker, and receives global
models over MQTT.
"""

import argparse
import json
import os
import time

import paho.mqtt.client as mqtt
import torch

from .datasets import load_wesad_dataset
from .model import ECGModel

# -----------------------------------------------------------------------------
# MQTT CONFIG
# -----------------------------------------------------------------------------
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_UPDATES = os.getenv(
    "MQTT_TOPIC_UPDATES", "fl/updates"
)  # Topic to publish local updates
TOPIC_GLOBAL_MODEL = os.getenv(
    "MQTT_TOPIC_GLOBAL", "fl/global_model"
)  # Topic to receive global models


# -----------------------------------------------------------------------------
# LOCAL MQTT CLIENT
# -----------------------------------------------------------------------------
class FLClientMQTT:
    """Federated MQTT client for local WESAD data."""

    def __init__(self):
        # Initialize ECG CNN model
        self.model = ECGModel()

        # Load and prepare local data
        X_train, X_test, y_train, y_test = load_wesad_dataset()
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=32,
            shuffle=True,
        )

        # Configure MQTT client
        self.mqtt = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_message = self._on_message
        self.mqtt.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        print(f"[CLIENT] Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")

        # Start MQTT loop in a separate thread
        self.mqtt.loop_start()

        # Flag to detect arrival of global model
        self._got_global = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connect callback: subscribe to global models."""
        print(f"[CLIENT] MQTT connected (rc={rc}), subscribing to {TOPIC_GLOBAL_MODEL}")
        client.subscribe(TOPIC_GLOBAL_MODEL)

    def _on_message(self, client, userdata, msg):
        """Handle global model messages from central server."""
        if msg.topic == TOPIC_GLOBAL_MODEL:
            try:
                payload = json.loads(msg.payload.decode())
                if "global_weights" in payload:
                    weights_dict = payload["global_weights"]
                    state_dict = {k: torch.tensor(v) for k, v in weights_dict.items()}
                    self.model.load_state_dict(state_dict, strict=True)
                    round_num = payload.get("round", "?")
                    print(f"[CLIENT] Global model loaded from round {round_num}")
                    self._got_global = True
                else:
                    print("[CLIENT] Invalid payload format received")
            except Exception as e:
                print(f"[CLIENT] Error processing MQTT message: {e}")

    def train_one_round(self):
        """Run one local training round and publish update."""
        # One epoch local training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()  # Binary classification
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for X, y in self.train_loader:
            optimizer.zero_grad()
            # Add channel dim for CNN: (batch, 1, sequence_length)
            X = X.unsqueeze(1)
            logits = self.model(X).squeeze()  # Remove extra dims
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"[CLIENT] Training done. Avg loss: {avg_loss:.4f}")

        # Serialize current weights and publish with metadata
        state = self.model.state_dict()
        payload = {
            "client_id": f"client_{id(self) % 1000}",  # simple client id
            "region": "region_0",  # default region (overridable)
            "weights": {k: v.cpu().numpy().tolist() for k, v in state.items()},
            "num_samples": (
                len(self.train_loader.dataset)
                if hasattr(self.train_loader.dataset, "__len__")
                else len(self.train_loader)
            ),
            "loss": avg_loss,
        }
        # Region override via env (MQTT_REGION) if provided
        payload["region"] = os.getenv("MQTT_REGION", payload["region"])

        self.mqtt.publish(TOPIC_UPDATES, json.dumps(payload))
        print(f"[CLIENT] Local update published to {TOPIC_UPDATES}")

        # Wait for new global model
        print("[CLIENT] Waiting for new global model...")
        while not self._got_global:
            time.sleep(1)
        self._got_global = False
        print("[CLIENT] New global model received, ready for next round")

    def run(self, rounds: int = 5, delay: float = 5.0):
        """Main loop: train → publish → sync → repeat."""
        print(f"[CLIENT] Starting {rounds} federated rounds")
        for rnd in range(1, rounds + 1):
            print(f"\n=== Round {rnd}/{rounds} ===")
            self.train_one_round()
            if rnd < rounds:  # No delay after last round
                time.sleep(delay)

        # Cleanup connections
        print("[CLIENT] Federated training completed, closing connections")
        self.mqtt.loop_stop()
        self.mqtt.disconnect()


# -----------------------------------------------------------------------------
# MAIN: run local client
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MQTT client for WESAD/ECG flow")
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.getenv("CLIENT_ROUNDS", "3")),
        help="Federated rounds",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("CLIENT_DELAY", "2.0")),
        help="Delay between rounds (s)",
    )
    parser.add_argument("--region", help="Override MQTT_REGION (fog node id)")
    args = parser.parse_args()

    if args.region:
        os.environ["MQTT_REGION"] = args.region

    print("=== LOCAL FEDERATED CLIENT ===")
    print("Starting MQTT client for fog computing...")

    client = FLClientMQTT()
    client.run(rounds=args.rounds, delay=args.delay)
