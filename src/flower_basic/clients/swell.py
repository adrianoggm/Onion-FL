from __future__ import annotations

"""
SWELL federated local client over MQTT.

Loads per-node train/val/test NPZ splits prepared by
scripts/prepare_swell_federated.py and trains a small MLP locally.
Publishes updated weights to the fog broker over MQTT.
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
from flower_basic.datasets.swell_federated import load_node_split
from flower_basic.swell_model import SwellMLP

# Telemetry (optional)
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
    start_span,
    start_client_span,
    start_server_span,
    start_consumer_span,
    start_producer_span,
    start_linked_producer_span,
    start_linked_consumer_span,
    inject_trace_context,
    extract_trace_context,
    SpanKind,
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

# Telemetry - initialized lazily in main() to avoid import-time side effects
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
GAUGE_CONTRIBUTION_WEIGHT = None
COUNTER_BATCHES_PROCESSED = None


def _init_telemetry():
    """Initialize telemetry. Called from main() to ensure proper service name."""
    global TRACER, METER
    global COUNTER_TRAINING_ROUNDS, COUNTER_UPDATES_PUBLISHED, COUNTER_GLOBAL_MODELS_RECEIVED
    global HIST_TRAINING_DURATION, HIST_TRAINING_LOSS, GAUGE_TRAIN_SAMPLES, GAUGE_VAL_SAMPLES, GAUGE_TEST_SAMPLES
    global GAUGE_CONTRIBUTION_WEIGHT, COUNTER_BATCHES_PROCESSED

    TRACER, METER = init_otel("swell-client")

    COUNTER_TRAINING_ROUNDS = create_counter(
        METER, "fl_client_training_rounds_total", "Training rounds completed by client"
    )
    COUNTER_UPDATES_PUBLISHED = create_counter(
        METER, "fl_client_updates_published_total", "Model updates published to MQTT"
    )
    COUNTER_GLOBAL_MODELS_RECEIVED = create_counter(
        METER,
        "fl_client_global_models_received_total",
        "Global models received from server",
    )
    HIST_TRAINING_DURATION = create_histogram(
        METER,
        "fl_client_training_duration_seconds",
        "Time for local training round",
        "s",
    )
    HIST_TRAINING_LOSS = create_histogram(
        METER, "fl_client_training_loss", "Training loss distribution", "1"
    )
    GAUGE_TRAIN_SAMPLES = create_gauge(
        METER, "fl_client_train_samples", "Number of training samples", "1"
    )
    GAUGE_VAL_SAMPLES = create_gauge(
        METER, "fl_client_val_samples", "Number of validation samples", "1"
    )
    GAUGE_TEST_SAMPLES = create_gauge(
        METER, "fl_client_test_samples", "Number of test samples", "1"
    )
    GAUGE_CONTRIBUTION_WEIGHT = create_gauge(
        METER,
        "fl_client_contribution_weight",
        "Estimated contribution weight (samples/total)",
        "1",
    )
    COUNTER_BATCHES_PROCESSED = create_counter(
        METER,
        "fl_client_batches_processed_total",
        "Total batches processed during training",
    )


class SwellFLClientMQTT(BaseMQTTComponent):
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
            raise RuntimeError(
                "Train split is empty for this node. Check subject assignments."
            )

        input_dim = X_train.shape[1]
        self.model = SwellMLP(input_dim=input_dim)

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
        # Optional validation split
        self.val_loader = None
        self.num_val_samples = 0  # Initialize before conditional
        val_path = self.node_dir / "val.npz"
        if val_path.exists():
            try:
                X_val, y_val, _ = load_node_split(val_path)
                if X_val.size > 0:
                    self.val_loader = DataLoader(
                        TensorDataset(
                            torch.from_numpy(X_val).float(),
                            torch.from_numpy(y_val).long(),
                        ),
                        batch_size=256,
                        shuffle=False,
                    )
                    self.num_val_samples = len(X_val)
            except Exception:
                self.val_loader = None
                self.num_val_samples = 0

        # Load test samples count
        self.num_test_samples = 0
        test_path = self.node_dir / "test.npz"
        if test_path.exists():
            try:
                X_test, _, _ = load_node_split(test_path)
                self.num_test_samples = len(X_test) if X_test.size > 0 else 0
            except Exception:
                self.num_test_samples = 0

        # Metrics output path
        self.metrics_path = self.node_dir / "val_metrics.jsonl"

        super().__init__(
            tag=self.tag,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            subscriptions=[self.topic_global],
        )
        self._got_global = False
        # Protect model updates from MQTT callback during training
        self._lock = threading.Lock()
        self._pending_global_state = None

        # Store sample count for contribution tracking
        self.num_samples = len(X_train)

        # Report dataset distribution
        self.num_samples + self.num_val_samples + self.num_test_samples
        print(
            f"{self.tag} Initialized with {self.num_samples} train / {self.num_val_samples} val / {self.num_test_samples} test samples"
        )

    def on_message(self, client, userdata, msg):
        if msg.topic == self.topic_global:
            try:
                payload = json.loads(msg.payload.decode())
                weights = payload.get("global_weights")
                trace_context = payload.get("trace_context", {})  # Extract trace context
                if weights:
                    # Use linked CONSUMER span to continue trace from server
                    with start_linked_consumer_span(
                        TRACER,
                        "client.receive_global_model",
                        trace_context,
                        source_service="server-swell",
                        attributes={"region": self.region, "round": payload.get("round", "?")}
                    ):
                        # Buffer the global state to apply between rounds (avoid in-place during training)
                        state = {
                            k: torch.tensor(v)
                            for k, v in weights.items()
                            if k in self.model.state_dict()
                        }
                        with self._lock:
                            self._pending_global_state = state
                        self._got_global = True
                        record_metric(
                            COUNTER_GLOBAL_MODELS_RECEIVED, 1, {"region": self.region}
                        )
                        print(
                            f"{self.tag} Global model available (round={payload.get('round','?')})"
                        )
            except Exception as e:
                print(f"{self.tag} Error processing global model: {e}")

    def train_one_round(self) -> float:
        start_time = time.time()

        with start_span(TRACER, "client.train_one_round", {"region": self.region}):
            with self._lock:
                self.model.train()
                total = 0.0
                n = 0
                batch_count = 0
                for X, y in self.train_loader:
                    self.optimizer.zero_grad()
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()
                    total += loss.item() * X.size(0)
                    n += X.size(0)
                    batch_count += 1
                avg_loss = total / max(n, 1)

            # Record metrics (OTEL)
            training_duration = time.time() - start_time
            record_metric(COUNTER_TRAINING_ROUNDS, 1, {"region": self.region})
            record_metric(
                COUNTER_BATCHES_PROCESSED, batch_count, {"region": self.region}
            )
            if HIST_TRAINING_DURATION:
                HIST_TRAINING_DURATION.record(
                    training_duration, {"region": self.region}
                )
            if HIST_TRAINING_LOSS:
                HIST_TRAINING_LOSS.record(avg_loss, {"region": self.region})
            record_metric(GAUGE_TRAIN_SAMPLES, n, {"region": self.region})
            
            # Record Prometheus metrics
            client_id = f"{self.region}_client_{os.getpid() % 10000}"
            CLIENT_TRAINING_ROUNDS.labels(client_id=client_id, region=self.region).inc()
            CLIENT_TRAINING_DURATION.labels(client_id=client_id, region=self.region).observe(training_duration)
            CLIENT_LOCAL_LOSS.labels(client_id=client_id, region=self.region).set(avg_loss)

        print(
            f"{self.tag} Train loss: {avg_loss:.4f} | samples: {n} | batches: {batch_count}"
        )
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
        
        # Record Prometheus validation accuracy
        client_id = f"{self.region}_client_{os.getpid() % 10000}"
        CLIENT_LOCAL_ACCURACY.labels(client_id=client_id, region=self.region).set(val_acc)
        
        print(f"{self.tag} Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}")
        return {"val_loss": val_loss, "val_acc": val_acc}

    def publish_update(self, avg_loss: float, val_acc: float = 0.0) -> None:
        # Use PRODUCER span with context propagation for distributed tracing
        with start_linked_producer_span(TRACER, "client.publish_update", "fog-broker", {"region": self.region}) as (span, trace_ctx):
            with self._lock:
                state = self.model.state_dict()
                weights = {
                    k: v.detach().cpu().numpy().tolist() for k, v in state.items()
                }

            # Use consistent client_id based on region and process id
            client_id = f"{self.region}_client_{os.getpid() % 10000}"

            payload = {
                "client_id": client_id,
                "region": self.region,
                "weights": weights,
                "num_samples": self.num_samples,
                "loss": float(avg_loss),
                "val_acc": float(val_acc),  # Include validation accuracy
                "trace_context": trace_ctx,  # Propagate trace context
            }
            self.mqtt.publish(self.topic_updates, json.dumps(payload))
            record_metric(
                COUNTER_UPDATES_PUBLISHED,
                1,
                {"region": self.region, "client_id": client_id},
            )
        print(
            f"{self.tag} Local update published ({self.num_samples} samples) to {self.topic_updates}"
        )

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
        
        # Client ID for metrics
        client_id = f"{self.region}_client_{os.getpid() % 10000}"

        # Register dataset distribution metrics once at start (OTEL)
        record_metric(GAUGE_TRAIN_SAMPLES, self.num_samples, {"region": self.region})
        record_metric(GAUGE_VAL_SAMPLES, self.num_val_samples, {"region": self.region})
        record_metric(
            GAUGE_TEST_SAMPLES, self.num_test_samples, {"region": self.region}
        )
        
        # Register Prometheus metrics
        CLIENT_TRAIN_SAMPLES.labels(client_id=client_id, region=self.region).set(self.num_samples)
        CLIENT_VAL_SAMPLES.labels(client_id=client_id, region=self.region).set(self.num_val_samples)
        CLIENT_TEST_SAMPLES.labels(client_id=client_id, region=self.region).set(self.num_test_samples)

        for r in range(1, rounds + 1):
            print(f"\n=== Round {r}/{rounds} ===")
            avg_loss = self.train_one_round()
            # Validate before publishing update
            val_metrics = self.evaluate_val()
            val_acc = val_metrics.get("val_acc", 0.0)
            # Persist metrics
            try:
                rec = {"round": r, "region": self.region, "train_loss": float(avg_loss)}
                rec.update({k: float(v) for k, v in val_metrics.items()})
                import json as _json

                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec) + "\n")
            except Exception:
                pass
            self.publish_update(avg_loss, val_acc)  # Include validation accuracy
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
        self.stop_mqtt()


def main():
    # Parse arguments first to get client index for metrics port
    ap = argparse.ArgumentParser(description="SWELL MQTT federated local client")
    ap.add_argument(
        "--node_dir",
        required=True,
        help="Path to node directory with splits (train/val/test npz)",
    )
    ap.add_argument(
        "--region", required=True, help="Region name to tag updates (e.g., fog_0)"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-updates", default=TOPIC_UPDATES)
    ap.add_argument("--topic-global", default=TOPIC_GLOBAL_MODEL)
    ap.add_argument("--client-index", type=int, default=-1, 
                    help="Client index for metrics port (0-99). If -1, uses PID-based port.")
    args = ap.parse_args()

    # Initialize telemetry for this service
    _init_telemetry()
    
    # Start Prometheus metrics server with deterministic port
    base_port = get_metrics_port_from_env(default=8100, component="CLIENT")
    if args.client_index >= 0:
        client_port = base_port + args.client_index
    else:
        client_port = base_port + (os.getpid() % 100)  # Fallback to PID-based
    start_metrics_server(port=client_port)
    
    # Cleanup function to push metrics before exit
    def cleanup(*_args):
        print(f"[CLIENT {args.region}] Pushing metrics before shutdown...")
        push_metrics_to_gateway(
            job="flower-client", 
            grouping_key={"region": args.region, "client_index": str(args.client_index)}
        )
        shutdown_telemetry()
    
    # Register cleanup for various termination signals
    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(0)))
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), exit(0)))

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
    
    try:
        client.run(rounds=args.rounds)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
