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
import random
import signal
from pathlib import Path

import numpy as np
import torch

from flower_basic.clients.federated_base import (
    ClientDataLoaders,
    FederatedMQTTClientBase,
)
from flower_basic.datasets.federated_common import (
    build_client_data,
    resolve_split_paths,
)
from flower_basic.datasets.swell_federated import load_node_split
from flower_basic.logging_utils import enable_timestamped_print
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
from flower_basic.swell_model import SwellMLP
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


def _set_determinism(seed: int) -> torch.Generator:
    """Configure deterministic behavior and return a seeded generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_client_data(
    node_dir: Path,
    *,
    batch_size: int,
    generator: torch.Generator | None,
) -> ClientDataLoaders:
    train_file, val_file, test_file = resolve_split_paths(node_dir)
    return build_client_data(
        train_file,
        val_file,
        test_file,
        load_split=load_node_split,
        batch_size=batch_size,
        generator=generator,
    )


class SwellFLClientMQTT(FederatedMQTTClientBase):
    def __init__(
        self,
        node_dir: str,
        region: str,
        client_id: str | None = None,
        seed: int | None = None,
        local_epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 64,
        mqtt_broker: str = MQTT_BROKER,
        mqtt_port: int = MQTT_PORT,
        topic_updates: str = TOPIC_UPDATES,
        topic_global: str = TOPIC_GLOBAL_MODEL,
    ):
        self.node_dir = Path(node_dir)
        self.seed = int(seed) if seed is not None else None
        self.metrics_path = self.node_dir / "val_metrics.jsonl"

        generator = _set_determinism(self.seed) if self.seed is not None else None
        data = _build_client_data(self.node_dir, batch_size=batch_size, generator=generator)
        model = SwellMLP(input_dim=data.input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        resolved_client_id = client_id or f"{region}_client_{os.getpid() % 10000}"

        super().__init__(
            tag=f"[CLIENT {region}]",
            region=region,
            client_id=resolved_client_id,
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
            global_source_service="server-swell",
            update_target_service="fog-broker",
        )

        print(
            f"{self.tag} Initialized with {self.num_samples} train / "
            f"{self.num_val_samples} val / {self.num_test_samples} test samples"
        )

    def on_global_model_buffered(self, envelope: GlobalModelEnvelope) -> None:
        record_metric(COUNTER_GLOBAL_MODELS_RECEIVED, 1, {"region": self.region})

    def on_train_round_completed(
        self, result: TrainRoundResult, duration: float
    ) -> None:
        record_metric(COUNTER_TRAINING_ROUNDS, 1, {"region": self.region})
        record_metric(
            COUNTER_BATCHES_PROCESSED, result.batch_count, {"region": self.region}
        )
        if HIST_TRAINING_DURATION:
            HIST_TRAINING_DURATION.record(duration, {"region": self.region})
        if HIST_TRAINING_LOSS:
            HIST_TRAINING_LOSS.record(result.avg_loss, {"region": self.region})
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
        record_metric(
            COUNTER_UPDATES_PUBLISHED,
            1,
            {"region": self.region, "client_id": self.client_id},
        )

    def on_dataset_metrics_registered(self) -> None:
        record_metric(GAUGE_TRAIN_SAMPLES, self.num_samples, {"region": self.region})
        record_metric(GAUGE_VAL_SAMPLES, self.num_val_samples, {"region": self.region})
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

    def persist_round_metrics(
        self, round_num: int, avg_loss: float, val_metrics: dict[str, float]
    ) -> None:
        try:
            record = {
                "round": round_num,
                "region": self.region,
                "train_loss": float(avg_loss),
            }
            record.update({key: float(value) for key, value in val_metrics.items()})
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            pass


def main():
    enable_timestamped_print()

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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mqtt-broker", default=MQTT_BROKER)
    ap.add_argument("--mqtt-port", type=int, default=MQTT_PORT)
    ap.add_argument("--topic-updates", default=TOPIC_UPDATES)
    ap.add_argument("--topic-global", default=TOPIC_GLOBAL_MODEL)
    ap.add_argument(
        "--client-index",
        type=int,
        default=-1,
        help="Client index for metrics port (0-99). If -1, uses PID-based port.",
    )
    ap.add_argument(
        "--client-id",
        default=None,
        help="Stable client identifier (defaults to region+pid).",
    )
    args = ap.parse_args()

    _init_telemetry()

    base_port = get_metrics_port_from_env(default=8100, component="CLIENT")
    if args.client_index >= 0:
        client_port = base_port + args.client_index
    else:
        client_port = base_port + (os.getpid() % 100)
    start_metrics_server(port=client_port)

    def cleanup(*_args):
        print(f"[CLIENT {args.region}] Pushing metrics before shutdown...")
        push_metrics_to_gateway(
            job="flower-client",
            grouping_key={
                "region": args.region,
                "client_index": str(args.client_index),
            },
        )
        shutdown_telemetry()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda s, f: (cleanup(), exit(0)))
    signal.signal(signal.SIGINT, lambda s, f: (cleanup(), exit(0)))

    client = SwellFLClientMQTT(
        args.node_dir,
        args.region,
        client_id=args.client_id,
        seed=args.seed,
        local_epochs=args.local_epochs,
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
