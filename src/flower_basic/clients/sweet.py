from __future__ import annotations

"""
SWEET federated local client over MQTT.

Loads per-node train/val/test NPZ splits prepared by
scripts/prepare_sweet_federated.py and trains a SweetMLP locally.
Publishes updated weights to the fog broker over MQTT.

Architecture: Client -> Fog Broker -> Fog Bridge -> Central Server
"""

import argparse
import atexit
import json
import os
import signal
from pathlib import Path

import torch

from flower_basic.clients.federated_base import (
    ClientDataLoaders,
    FederatedMQTTClientBase,
)
from flower_basic.datasets.federated_common import (
    build_client_data,
    resolve_split_paths,
)
from flower_basic.datasets.sweet_federated import load_node_split
from flower_basic.prometheus_metrics import (
    CLIENT_LOCAL_ACCURACY,
    CLIENT_LOCAL_LOSS,
    CLIENT_TEST_SAMPLES,
    CLIENT_TRAIN_SAMPLES,
    CLIENT_TRAINING_DURATION,
    CLIENT_TRAINING_ROUNDS,
    CLIENT_VAL_SAMPLES,
    get_metrics_port_from_env,
    push_metrics_to_gateway,
    start_metrics_server,
)
from flower_basic.runtime_protocol import GlobalModelEnvelope
from flower_basic.sweet_model import SweetMLP
from flower_basic.telemetry import (
    create_counter,
    create_gauge,
    create_histogram,
    init_otel,
    record_metric,
    shutdown_telemetry,
)
from flower_basic.training.local import EvalResult, TrainRoundResult

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


def _build_client_data(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    *,
    batch_size: int,
) -> ClientDataLoaders:
    return build_client_data(
        train_file,
        val_file,
        test_file,
        load_split=load_node_split,
        batch_size=batch_size,
    )


def _log_pretrained_model_hint(tag: str, node_dir: Path) -> None:
    pretrained_path = node_dir.parent / "pretrained_model.json"
    if not pretrained_path.exists():
        return

    try:
        with pretrained_path.open(encoding="utf-8") as handle:
            json.load(handle)
        print(f"{tag} Pre-trained model found at {pretrained_path}")
        print(
            f"{tag} Note: Transfer learning from XGBoost to PyTorch requires weight conversion"
        )
    except Exception as exc:
        print(f"{tag} Warning: Could not load pre-trained model: {exc}")


class SweetFLClientMQTT(FederatedMQTTClientBase):
    """SWEET federated client using the shared MQTT workflow."""

    def __init__(
        self,
        node_dir: str,
        region: str,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        subject_id: str | None = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        local_epochs: int = 5,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        topic_updates: str = TOPIC_UPDATES,
        topic_global: str = TOPIC_GLOBAL_MODEL,
    ):
        self.node_dir = Path(node_dir)
        self.region = region
        self.subject_id = subject_id

        tag = f"[CLIENT {region}{'_' + subject_id if subject_id else ''}]"
        client_id = (
            f"{region}_client_{subject_id}"
            if subject_id
            else f"{region}_client_{os.getpid() % 10000}"
        )

        train_file, val_file, test_file = resolve_split_paths(self.node_dir, subject_id)
        data = _build_client_data(
            train_file, val_file, test_file, batch_size=batch_size
        )
        model = SweetMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        _log_pretrained_model_hint(tag, self.node_dir)

        super().__init__(
            tag=tag,
            region=region,
            client_id=client_id,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            data=data,
            local_epochs=local_epochs,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            topic_updates=topic_updates,
            topic_global=topic_global,
            tracer=TRACER,
            global_source_service="server-sweet",
            update_target_service="sweet-fog-broker",
        )

        print(
            f"{self.tag} Initialized with {self.num_samples} train / "
            f"{self.num_val_samples} val / {self.num_test_samples} test samples"
        )

    def on_global_model_buffered(self, envelope: GlobalModelEnvelope) -> None:
        if COUNTER_GLOBAL_MODELS_RECEIVED:
            record_metric(COUNTER_GLOBAL_MODELS_RECEIVED, 1, {"region": self.region})

    def on_train_round_completed(
        self, result: TrainRoundResult, duration: float
    ) -> None:
        if COUNTER_TRAINING_ROUNDS:
            record_metric(COUNTER_TRAINING_ROUNDS, 1, {"region": self.region})
        if COUNTER_BATCHES_PROCESSED:
            record_metric(
                COUNTER_BATCHES_PROCESSED, result.batch_count, {"region": self.region}
            )
        if HIST_TRAINING_DURATION:
            HIST_TRAINING_DURATION.record(duration, {"region": self.region})
        if HIST_TRAINING_LOSS:
            HIST_TRAINING_LOSS.record(result.avg_loss, {"region": self.region})
        if GAUGE_TRAIN_SAMPLES:
            record_metric(
                GAUGE_TRAIN_SAMPLES, result.num_samples, {"region": self.region}
            )

        CLIENT_TRAINING_ROUNDS.labels(
            client_id=self.client_id, region=self.region
        ).inc()
        CLIENT_TRAINING_DURATION.labels(
            client_id=self.client_id, region=self.region
        ).observe(duration)
        CLIENT_LOCAL_LOSS.labels(client_id=self.client_id, region=self.region).set(
            result.avg_loss
        )

    def on_validation_completed(self, result: EvalResult) -> None:
        CLIENT_LOCAL_ACCURACY.labels(client_id=self.client_id, region=self.region).set(
            result.accuracy
        )

    def on_update_published(self, payload: dict[str, object]) -> None:
        if COUNTER_UPDATES_PUBLISHED:
            record_metric(
                COUNTER_UPDATES_PUBLISHED,
                1,
                {"region": self.region, "client_id": self.client_id},
            )

    def on_dataset_metrics_registered(self) -> None:
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

        CLIENT_TRAIN_SAMPLES.labels(client_id=self.client_id, region=self.region).set(
            self.num_samples
        )
        CLIENT_VAL_SAMPLES.labels(client_id=self.client_id, region=self.region).set(
            self.num_val_samples
        )
        CLIENT_TEST_SAMPLES.labels(client_id=self.client_id, region=self.region).set(
            self.num_test_samples
        )

    def should_wait_for_global(self, round_num: int) -> bool:
        return True

    def global_wait_timeout_seconds(self, round_num: int) -> float:
        return 60.0


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

    if args.enable_telemetry:
        _init_telemetry()

    if args.enable_prometheus:
        port = get_metrics_port_from_env(default=9090)
        start_metrics_server(port)
        print(f"Prometheus metrics at http://localhost:{port}/metrics")

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

    def cleanup():
        print(f"\n{client.tag} Shutting down...")
        client.stop_mqtt()
        if args.enable_telemetry:
            shutdown_telemetry()
        if args.enable_prometheus:
            push_metrics_to_gateway()

    def handle_signal(_signum, _frame):
        cleanup()
        raise SystemExit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        client.run(rounds=args.rounds)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
