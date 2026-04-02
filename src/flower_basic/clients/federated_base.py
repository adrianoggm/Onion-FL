from __future__ import annotations

"""Common workflow for MQTT-based federated clients."""

import json
import threading
import time
from typing import Any

import torch

from flower_basic.clients.baseclient import BaseMQTTComponent
from flower_basic.datasets.federated_common import ClientDataLoaders
from flower_basic.runtime_protocol import (
    GlobalModelEnvelope,
    build_client_update_payload,
    decode_global_model_message,
)
from flower_basic.telemetry import (
    start_linked_consumer_span,
    start_linked_producer_span,
    start_span,
)
from flower_basic.training.local import (
    EvalResult,
    TrainRoundResult,
    evaluate_classifier,
    train_classifier_round,
)


class FederatedMQTTClientBase(BaseMQTTComponent):
    """Shared MQTT + round workflow for local federated clients."""

    def __init__(
        self,
        *,
        tag: str,
        region: str,
        client_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        data: ClientDataLoaders,
        local_epochs: int,
        mqtt_broker: str,
        mqtt_port: int,
        topic_updates: str,
        topic_global: str,
        tracer: Any = None,
        global_source_service: str = "server",
        update_target_service: str = "fog-broker",
    ) -> None:
        self.tag = tag
        self.region = region
        self.client_id = client_id
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = data.train_loader
        self.val_loader = data.val_loader
        self.num_samples = data.num_train_samples
        self.num_val_samples = data.num_val_samples
        self.num_test_samples = data.num_test_samples
        self.local_epochs = max(1, int(local_epochs))
        self.topic_updates = topic_updates
        self.topic_global = topic_global
        self.tracer = tracer
        self.global_source_service = global_source_service
        self.update_target_service = update_target_service
        self.current_round = 0
        self._got_global = False
        self._lock = threading.Lock()
        self._pending_global_state: dict[str, torch.Tensor] | None = None

        super().__init__(
            tag=self.tag,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            subscriptions=[self.topic_global],
        )

    def on_message(self, client, userdata, msg) -> None:
        if msg.topic != self.topic_global:
            return

        try:
            envelope = decode_global_model_message(
                msg.payload, self.model.state_dict().keys()
            )
            if envelope is None:
                return

            with start_linked_consumer_span(
                self.tracer,
                "client.receive_global_model",
                envelope.trace_context,
                source_service=self.global_source_service,
                attributes={
                    "region": self.region,
                    "round": envelope.round_num or "?",
                },
            ):
                state = {
                    name: torch.as_tensor(value)
                    for name, value in envelope.weights.items()
                }
                with self._lock:
                    self._pending_global_state = state
                    self._got_global = True
                self.on_global_model_buffered(envelope)
                print(
                    f"{self.tag} Global model available (round={envelope.round_num or '?'})"
                )
        except Exception as exc:
            print(f"{self.tag} Error processing global model: {exc}")

    def on_global_model_buffered(self, envelope: GlobalModelEnvelope) -> None:
        """Hook for telemetry or metrics after a model is buffered."""

    def on_train_round_completed(
        self, result: TrainRoundResult, duration: float
    ) -> None:
        """Hook for telemetry or metrics after training."""

    def on_validation_completed(self, result: EvalResult) -> None:
        """Hook for telemetry or metrics after validation."""

    def on_update_published(self, payload: dict[str, Any]) -> None:
        """Hook for telemetry or metrics after publishing a local update."""

    def on_dataset_metrics_registered(self) -> None:
        """Hook for one-time dataset metrics registration."""

    def persist_round_metrics(
        self, round_num: int, avg_loss: float, val_metrics: dict[str, float]
    ) -> None:
        """Hook to persist round metrics."""

    def should_wait_for_global(self, round_num: int) -> bool:
        """Whether a round should block waiting for a global model."""
        return round_num > 1

    def global_wait_timeout_seconds(self, round_num: int) -> float:
        """Timeout used while waiting for a global model."""
        return 30.0

    def _serialize_model_weights(self) -> dict[str, Any]:
        with self._lock:
            state = self.model.state_dict()
            return {name: value.detach().cpu().numpy() for name, value in state.items()}

    def train_one_round(self) -> float:
        start_time = time.time()

        with start_span(self.tracer, "client.train_one_round", {"region": self.region}):
            with self._lock:
                result = train_classifier_round(
                    self.model,
                    self.train_loader,
                    self.optimizer,
                    self.criterion,
                    local_epochs=self.local_epochs,
                )

        duration = time.time() - start_time
        self.on_train_round_completed(result, duration)
        print(
            f"{self.tag} Train loss: {result.avg_loss:.4f} | samples: {result.num_samples} | "
            f"batches: {result.batch_count} | epochs: {self.local_epochs}"
        )
        return result.avg_loss

    def evaluate_val(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        with self._lock:
            result = evaluate_classifier(self.model, self.val_loader, self.criterion)

        if result.num_samples == 0:
            return {}

        self.on_validation_completed(result)
        print(
            f"{self.tag} Val loss: {result.loss:.4f} | Val acc: {result.accuracy:.3f}"
        )
        return {"val_loss": result.loss, "val_acc": result.accuracy}

    def build_update_payload(
        self,
        *,
        avg_loss: float,
        val_acc: float,
        round_num: int | None,
        trace_context: dict[str, Any],
    ) -> dict[str, Any]:
        payload_round = (
            round_num if round_num is not None else max(1, self.current_round)
        )
        return build_client_update_payload(
            client_id=self.client_id,
            region=self.region,
            round_num=int(payload_round),
            weights=self._serialize_model_weights(),
            num_samples=self.num_samples,
            avg_loss=avg_loss,
            val_acc=val_acc,
            sent_at=time.time(),
            trace_context=trace_context,
        )

    def publish_update(
        self, avg_loss: float, val_acc: float = 0.0, round_num: int | None = None
    ) -> None:
        with start_linked_producer_span(
            self.tracer,
            "client.publish_update",
            self.update_target_service,
            {"region": self.region},
        ) as (_span, trace_ctx):
            payload = self.build_update_payload(
                avg_loss=avg_loss,
                val_acc=val_acc,
                round_num=round_num,
                trace_context=trace_ctx,
            )
            self.mqtt.publish(self.topic_updates, json.dumps(payload))
            self.on_update_published(payload)

        print(
            f"{self.tag} Local update published ({self.num_samples} samples) to {self.topic_updates}"
        )

    def wait_for_global(self, timeout_s: float = 30.0) -> bool:
        waited = 0.0
        interval = 0.5
        with self._lock:
            if self._pending_global_state is not None:
                return True
            self._got_global = False

        while waited < timeout_s:
            with self._lock:
                if self._pending_global_state is not None or self._got_global:
                    return True
            time.sleep(interval)
            waited += interval

        print(f"{self.tag} Timeout waiting for global model. Proceeding.")
        return False

    def apply_pending_global_state(self) -> bool:
        with self._lock:
            if self._pending_global_state is None:
                return False
            self.model.load_state_dict(self._pending_global_state, strict=False)
            self._pending_global_state = None
            self._got_global = False

        print(f"{self.tag} Global model applied")
        return True

    def run(self, rounds: int = 3, delay: float = 2.0) -> None:
        print(f"{self.tag} Starting {rounds} federated rounds (region={self.region})")
        self.on_dataset_metrics_registered()

        for round_num in range(1, rounds + 1):
            if self.should_wait_for_global(round_num):
                self.wait_for_global(
                    timeout_s=self.global_wait_timeout_seconds(round_num)
                )
            self.apply_pending_global_state()

            print(f"\n=== Round {round_num}/{rounds} ===")
            self.current_round = round_num
            avg_loss = self.train_one_round()
            val_metrics = self.evaluate_val()
            val_acc = float(val_metrics.get("val_acc", 0.0))
            self.persist_round_metrics(round_num, avg_loss, val_metrics)
            self.publish_update(avg_loss, val_acc, round_num=round_num)

            if round_num < rounds:
                time.sleep(delay)

        self.stop_mqtt()
