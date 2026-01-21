from __future__ import annotations

"""
SWEET federated local client over MQTT.

Loads per-node train/val/test NPZ splits prepared by
scripts/prepare_sweet_federated.py and trains a SweetMLP locally.
Publishes updated weights to the fog broker over MQTT.

Architecture: Client → Fog Broker → Fog Bridge → Central Server
"""

import argparse
import atexit
import json
import os
import signal
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from flower_basic.clients.baseclient import BaseMQTTComponent
from flower_basic.datasets.sweet_federated import load_node_split
from flower_basic.sweet_model import SweetMLP

# Telemetry
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_span,
    start_linked_producer_span,
    start_linked_consumer_span,
)
from flower_basic.prometheus_metrics import (
    start_metrics_server,
    get_metrics_port_from_env,
    push_metrics_to_gateway,
    CLIENT_TRAIN_SAMPLES,
    CLIENT_VAL_SAMPLES,
    CLIENT_TEST_SAMPLES,
    CLIENT_TRAINING_ROUNDS,
    CLIENT_TRAINING_DURATION,
    CLIENT_LOCAL_LOSS,
    CLIENT_LOCAL_ACCURACY,
)

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_UPDATES = os.getenv("MQTT_TOPIC_UPDATES", "fl/updates")
TOPIC_GLOBAL_MODEL = os.getenv("MQTT_TOPIC_GLOBAL", "fl/global_model")

# Telemetry globals
TRACER = None
METER = None
COUNTER_TRAINING_ROUNDS = None
COUNTER_UPDATES_PUBLISHED = None
COUNTER_GLOBAL_MODELS_RECEIVED = None
HIST_TRAINING_DURATION = None
HIST_TRAINING_LOSS = None
GAUGE_TRAIN_SAMPLES = None
GAUGE_VAL_SAMPLES = None
GAUGE_TEST_SAMPLES = None
COUNTER_BATCHES_PROCESSED = None


def _init_telemetry():
    """Initialize telemetry for SWEET client."""
    global TRACER, METER
    global COUNTER_TRAINING_ROUNDS, COUNTER_UPDATES_PUBLISHED, COUNTER_GLOBAL_MODELS_RECEIVED
    global HIST_TRAINING_DURATION, HIST_TRAINING_LOSS, GAUGE_TRAIN_SAMPLES, GAUGE_VAL_SAMPLES, GAUGE_TEST_SAMPLES
    global COUNTER_BATCHES_PROCESSED

    TRACER, METER = init_otel("sweet-client")

    COUNTER_TRAINING_ROUNDS = create_counter(
        METER, "fl_client_training_rounds_total", "Training rounds completed"
    )
    COUNTER_UPDATES_PUBLISHED = create_counter(
        METER, "fl_client_updates_published_total", "Model updates published"
    )
    COUNTER_GLOBAL_MODELS_RECEIVED = create_counter(
        METER, "fl_client_global_models_received_total", "Global models received"
    )
    HIST_TRAINING_DURATION = create_histogram(
        METER, "fl_client_training_duration_seconds", "Training duration", "s"
    )
    HIST_TRAINING_LOSS = create_histogram(
        METER, "fl_client_training_loss", "Training loss", "1"
    )
    GAUGE_TRAIN_SAMPLES = create_gauge(
        METER, "fl_client_train_samples", "Training samples", "1"
    )
    GAUGE_VAL_SAMPLES = create_gauge(
        METER, "fl_client_val_samples", "Validation samples", "1"
    )
    GAUGE_TEST_SAMPLES = create_gauge(
        METER, "fl_client_test_samples", "Test samples", "1"
    )
    COUNTER_BATCHES_PROCESSED = create_counter(
        METER, "fl_client_batches_processed_total", "Batches processed"
    )


class SweetFLClientMQTT(BaseMQTTComponent):
    """SWEET federated client - follows SWELL architecture pattern."""

    def __init__(
        self,
        node_dir: str,
        region: str,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        subject_id: str | None = None,  # For per-subject client
        lr: float = 1e-3,
        batch_size: int = 32,
        local_epochs: int = 5,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        topic_updates: str = TOPIC_UPDATES,
        topic_global: str = TOPIC_GLOBAL_MODEL,
    ):
        """SWEET FL client using MQTT.

        Args:
            node_dir: Path to fog node directory (e.g., federated_runs/sweet/auto_5nodes/fog_0)
            region: Node identifier (e.g., 'fog_0')
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            num_classes: Number of classes
            subject_id: Optional subject ID for per-subject client (e.g., '123')
            lr: Learning rate
            batch_size: Batch size for training
            local_epochs: Number of local epochs per round
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            topic_updates: MQTT topic for publishing updates
            topic_global: MQTT topic for receiving global model
        """
        self.node_dir = Path(node_dir)
        self.region = region
        self.subject_id = subject_id
        self.tag = (
            f"[CLIENT {self.region}{'_' + self.subject_id if self.subject_id else ''}]"
        )
        self.topic_updates = topic_updates
        self.topic_global = topic_global
        self.local_epochs = local_epochs

        # Load splits - support both aggregated and per-subject
        if subject_id:
            # Per-subject client (like SWELL)
            train_file = self.node_dir / f"subject_{subject_id}" / "train.npz"
            val_file = self.node_dir / f"subject_{subject_id}" / "val.npz"
            test_file = self.node_dir / f"subject_{subject_id}" / "test.npz"
        else:
            # Aggregated node client (legacy)
            train_file = self.node_dir / "train.npz"
            val_file = self.node_dir / "val.npz"
            test_file = self.node_dir / "test.npz"

        X_train, y_train, _ = load_node_split(train_file)
        if X_train.size == 0:
            raise RuntimeError(
                f"Train split is empty for {self.tag}. Check subject assignments."
            )

        # Model
        self.model = SweetMLP(
            input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes
        )

        # Load pre-trained model if available (transfer learning)
        pretrained_path = self.node_dir.parent / "pretrained_model.json"
        if pretrained_path.exists():
            try:
                import json

                with open(pretrained_path, "r") as f:
                    pretrained_data = json.load(f)

                # Convert XGBoost weights to PyTorch (if needed)
                # For now, just log that pre-trained model exists
                print(f"{self.tag} Pre-trained model found at {pretrained_path}")
                print(
                    f"{self.tag} Note: Transfer learning from XGBoost to PyTorch requires weight conversion"
                )
                # TODO: Implement XGBoost → PyTorch weight transfer
            except Exception as e:
                print(f"{self.tag} Warning: Could not load pre-trained model: {e}")

        # DataLoaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Validation split
        self.val_loader = None
        self.num_val_samples = 0
        if val_file.exists():
            val_X, val_y, _ = load_node_split(val_file)
            if val_X.size > 0:
                self.val_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long()
                    ),
                    batch_size=256,
                    shuffle=False,
                )
                self.num_val_samples = len(val_X)

        # Test samples count
        self.num_test_samples = 0
        if test_file.exists():
            test_X, _, _ = load_node_split(test_file)
            self.num_test_samples = len(test_X) if test_X.size > 0 else 0

        super().__init__(
            tag=self.tag,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            subscriptions=[self.topic_global],
        )
        self._got_global = False
        self._lock = threading.Lock()
        self._pending_global_state = None

        self.num_samples = len(X_train)

        print(
            f"{self.tag} Initialized with {self.num_samples} train / "
            f"{self.num_val_samples} val / {self.num_test_samples} test samples"
        )

    def on_message(self, client, userdata, msg):
        """Receive global model from server."""
        if msg.topic == self.topic_global:
            try:
                payload = json.loads(msg.payload.decode())
                weights = payload.get("global_weights")
                trace_context = payload.get("trace_context", {})
                if weights:
                    with start_linked_consumer_span(
                        TRACER,
                        "client.receive_global_model",
                        trace_context,
                        source_service="server-sweet",
                        attributes={
                            "region": self.region,
                            "round": payload.get("round", "?"),
                        },
                    ):
                        state = {
                            k: torch.tensor(v)
                            for k, v in weights.items()
                            if k in self.model.state_dict()
                        }
                        with self._lock:
                            self._pending_global_state = state
                        self._got_global = True
                        if COUNTER_GLOBAL_MODELS_RECEIVED:
                            record_metric(
                                COUNTER_GLOBAL_MODELS_RECEIVED,
                                1,
                                {"region": self.region},
                            )
                        print(
                            f"{self.tag} Global model available (round={payload.get('round','?')})"
                        )
            except Exception as e:
                print(f"{self.tag} Error processing global model: {e}")

    def train_one_round(self) -> float:
        """Train for multiple local epochs."""
        start_time = time.time()

        with start_span(TRACER, "client.train_one_round", {"region": self.region}):
            with self._lock:
                self.model.train()
                total_loss = 0.0
                n = 0
                batch_count = 0

                # Multiple local epochs
                for epoch in range(self.local_epochs):
                    epoch_loss = 0.0
                    epoch_samples = 0
                    for X, y in self.train_loader:
                        self.optimizer.zero_grad()
                        logits = self.model(X)
                        loss = self.criterion(logits, y)
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item() * X.size(0)
                        epoch_samples += X.size(0)
                        batch_count += 1

                    total_loss += epoch_loss
                    n += epoch_samples

                avg_loss = total_loss / max(n, 1)

            # Record metrics
            training_duration = time.time() - start_time
            if COUNTER_TRAINING_ROUNDS:
                record_metric(COUNTER_TRAINING_ROUNDS, 1, {"region": self.region})
            if COUNTER_BATCHES_PROCESSED:
                record_metric(
                    COUNTER_BATCHES_PROCESSED, batch_count, {"region": self.region}
                )
            if HIST_TRAINING_DURATION:
                HIST_TRAINING_DURATION.record(
                    training_duration, {"region": self.region}
                )
            if HIST_TRAINING_LOSS:
                HIST_TRAINING_LOSS.record(avg_loss, {"region": self.region})
            if GAUGE_TRAIN_SAMPLES:
                record_metric(GAUGE_TRAIN_SAMPLES, n, {"region": self.region})

            # Prometheus metrics
            client_id = f"{self.region}_client_{os.getpid() % 10000}"
            CLIENT_TRAINING_ROUNDS.labels(client_id=client_id, region=self.region).inc()
            CLIENT_TRAINING_DURATION.labels(
                client_id=client_id, region=self.region
            ).observe(training_duration)
            CLIENT_LOCAL_LOSS.labels(client_id=client_id, region=self.region).set(
                avg_loss
            )

        print(
            f"{self.tag} Train loss: {avg_loss:.4f} | samples: {n} | batches: {batch_count} | epochs: {self.local_epochs}"
        )
        return avg_loss

    def evaluate_val(self) -> dict:
        """Evaluate on validation set."""
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

        # Prometheus validation accuracy
        client_id = f"{self.region}_client_{os.getpid() % 10000}"
        CLIENT_LOCAL_ACCURACY.labels(client_id=client_id, region=self.region).set(
            val_acc
        )

        print(f"{self.tag} Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}")
        return {"val_loss": val_loss, "val_acc": val_acc}

    def publish_update(self, avg_loss: float, val_acc: float = 0.0) -> None:
        """Publish local weights to fog broker."""
        with start_linked_producer_span(
            TRACER, "client.publish_update", "sweet-fog-broker", {"region": self.region}
        ) as (span, trace_ctx):
            with self._lock:
                state = self.model.state_dict()
                weights = {
                    k: v.detach().cpu().numpy().tolist() for k, v in state.items()
                }

            client_id = f"{self.region}_client_{os.getpid() % 10000}"

            payload = {
                "client_id": client_id,
                "region": self.region,
                "weights": weights,
                "num_samples": self.num_samples,
                "loss": float(avg_loss),
                "val_acc": float(val_acc),
                "trace_context": trace_ctx,
            }
            self.mqtt.publish(self.topic_updates, json.dumps(payload))
            if COUNTER_UPDATES_PUBLISHED:
                record_metric(
                    COUNTER_UPDATES_PUBLISHED,
                    1,
                    {"region": self.region, "client_id": client_id},
                )
        print(
            f"{self.tag} Local update published ({self.num_samples} samples) to {self.topic_updates}"
        )

    def wait_for_global(self, timeout_s: float = 30.0) -> bool:
        """Wait for global model from server."""
        waited = 0.0
        self._got_global = False
        interval = 0.5
        while not self._got_global and waited < timeout_s:
            time.sleep(interval)
            waited += interval
        if not self._got_global:
            print(f"{self.tag} Timeout waiting for global model. Proceeding.")
        return self._got_global

    def run(self, rounds: int = 10, delay: float = 2.0) -> None:
        """Run federated learning rounds."""
        print(f"{self.tag} Starting {rounds} federated rounds (region={self.region})")

        client_id = f"{self.region}_client_{os.getpid() % 10000}"

        # Register dataset metrics
        if GAUGE_TRAIN_SAMPLES:
            record_metric(
                GAUGE_TRAIN_SAMPLES, self.num_samples, {"region": self.region}
            )
        if GAUGE_VAL_SAMPLES:
            record_metric(
                GAUGE_VAL_SAMPLES, self.num_val_samples, {"region": self.region}
            )
        if GAUGE_TEST_SAMPLES:
            record_metric(
                GAUGE_TEST_SAMPLES, self.num_test_samples, {"region": self.region}
            )

        # Prometheus metrics
        CLIENT_TRAIN_SAMPLES.labels(client_id=client_id, region=self.region).set(
            self.num_samples
        )
        CLIENT_VAL_SAMPLES.labels(client_id=client_id, region=self.region).set(
            self.num_val_samples
        )
        CLIENT_TEST_SAMPLES.labels(client_id=client_id, region=self.region).set(
            self.num_test_samples
        )

        for r in range(rounds):
            print(f"\n{self.tag} ===== Round {r+1}/{rounds} =====")

            # Wait for global model
            if self.wait_for_global(timeout_s=60.0):
                with self._lock:
                    if self._pending_global_state:
                        self.model.load_state_dict(
                            self._pending_global_state, strict=False
                        )
                        self._pending_global_state = None

            # Train
            avg_loss = self.train_one_round()

            # Validate
            val_metrics = self.evaluate_val()
            val_acc = val_metrics.get("val_acc", 0.0)

            # Publish to fog broker
            self.publish_update(avg_loss, val_acc)

            time.sleep(delay)

        print(f"{self.tag} Training complete!")


def main():
    parser = argparse.ArgumentParser(description="SWEET Federated Client (MQTT)")
    parser.add_argument("--node-dir", required=True, help="Node data directory")
    parser.add_argument("--region", required=True, help="Region/Node ID (fog_0, etc)")
    parser.add_argument("--subject-id", help="Subject ID for per-subject client")
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--mqtt-broker", default=MQTT_BROKER)
    parser.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    parser.add_argument("--enable-telemetry", action="store_true")
    parser.add_argument("--enable-prometheus", action="store_true")

    args = parser.parse_args()

    # Initialize telemetry
    if args.enable_telemetry:
        _init_telemetry()

    # Start Prometheus
    if args.enable_prometheus:
        port = get_metrics_port_from_env(default=9090)
        start_metrics_server(port)
        print(f"Prometheus metrics at http://localhost:{port}/metrics")

    # Create client
    client = SweetFLClientMQTT(
        node_dir=args.node_dir,
        region=args.region,
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=args.num_classes,
        subject_id=args.subject_id,
        lr=args.lr,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        mqtt_broker=args.mqtt_broker,
        mqtt_port=args.mqtt_port,
    )

    # Cleanup
    def cleanup(*args):
        print(f"\n{client.tag} Shutting down...")
        client.stop_mqtt()
        if args.enable_telemetry:
            shutdown_telemetry()
        if args.enable_prometheus:
            push_metrics_to_gateway()
        exit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Run
    client.run(rounds=args.rounds)


if __name__ == "__main__":
    main()
