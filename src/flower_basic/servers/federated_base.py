from __future__ import annotations

"""Common workflow for MQTT-enabled Flower server strategies."""

import json
import time
import traceback
from typing import Any

import flwr as fl
import numpy as np
import paho.mqtt.client as mqtt

from flower_basic.prometheus_metrics import (
    FL_ACCURACY,
    FL_ACTIVE_CLIENTS,
    FL_AGGREGATIONS,
    FL_LOSS,
    FL_ROUND_DURATION,
    FL_ROUNDS,
)
from flower_basic.runtime_protocol import (
    build_global_model_payload,
    extract_named_parameters,
)
from flower_basic.telemetry import (
    record_metric,
    start_linked_producer_span,
    start_server_span,
)


class FederatedMQTTStrategyBase(fl.server.strategy.FedAvg):
    """Shared aggregation/publish flow for dataset-specific strategies."""

    def __init__(
        self,
        *,
        model,
        mqtt_client: mqtt.Client | None,
        eval_data: Any = None,
        total_rounds: int = 1,
        tracer: Any = None,
        model_topic: str = "fl/global_model",
        publish_target_service: str = "client",
        server_label: str = "server",
        tag: str = "[SERVER]",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.global_model = model
        self.mqtt = mqtt_client
        self.param_names = list(model.state_dict().keys())
        self.eval_data = eval_data
        self.total_rounds = total_rounds
        self.tracer = tracer
        self.model_topic = model_topic
        self.publish_target_service = publish_target_service
        self.server_label = server_label
        self.tag = tag
        self.history = self.initialize_history()
        self.log_eval_data_status()

    def initialize_history(self) -> dict[str, list[Any]]:
        """Initial aggregation history structure."""
        return {"round": [], "loss": [], "accuracy": []}

    def describe_eval_data(self) -> str | None:
        """Human-readable description used when eval data is available."""
        if self.eval_data is None:
            return None
        features, labels = self.eval_data
        unique_labels = np.unique(labels)
        return f"{features.shape[0]} samples, {len(unique_labels)} classes"

    def log_eval_data_status(self) -> None:
        description = self.describe_eval_data()
        if description is not None:
            print(f"{self.tag} Evaluation data loaded: {description}")
        else:
            print(
                f"{self.tag} WARNING: No evaluation data - model quality will NOT be verified!"
            )

    def before_aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> None:
        """Hook executed before running the Flower aggregation."""

    def on_aggregated_state_ready(
        self,
        server_round: int,
        state_dict: dict[str, Any],
        results: list[tuple[Any, fl.common.FitRes]],
    ) -> None:
        """Hook executed after the aggregated state dict is available."""

    def record_aggregation_metrics(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        aggregation_time: float,
    ) -> None:
        """Record default per-round telemetry and Prometheus metrics."""
        self.record_standard_round_metrics(
            server_round,
            num_results=len(results),
            aggregation_time=aggregation_time,
        )

    def should_evaluate_round(self, server_round: int) -> bool:
        """Whether the current round should trigger centralized evaluation."""
        return server_round == self.total_rounds

    def evaluate_aggregated_state(
        self, server_round: int, state_dict: dict[str, Any]
    ) -> Any:
        """Run centralized evaluation for the aggregated state."""
        raise NotImplementedError

    def on_final_evaluation(self, server_round: int, evaluation: Any) -> None:
        """Persist and publish evaluation outputs."""
        raise NotImplementedError

    def on_missing_final_eval_data(self, server_round: int) -> None:
        print(
            f"{self.tag} WARNING: FINAL ROUND: No test data available for evaluation!"
        )

    def on_deferred_evaluation(self, server_round: int) -> None:
        print(
            f"{self.tag} Round {server_round}/{self.total_rounds} complete. "
            "Evaluation will run after final round."
        )

    def finalize_standard_evaluation(
        self,
        server_round: int,
        *,
        loss: float,
        accuracy: float,
        accuracy_gauge: Any = None,
        loss_gauge: Any = None,
    ) -> None:
        """Record common final-evaluation outputs shared by server strategies."""
        self.history["round"].append(server_round)
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)

        if accuracy_gauge is not None:
            record_metric(accuracy_gauge, accuracy, {"round": str(server_round)})
        if loss_gauge is not None:
            record_metric(loss_gauge, loss, {"round": str(server_round)})

        FL_ACCURACY.labels(server=self.server_label).set(accuracy)
        FL_LOSS.labels(server=self.server_label).set(loss)
        self.print_final_evaluation_summary(loss=loss, accuracy=accuracy)

    def record_active_clients_measurement(
        self,
        server_round: int,
        *,
        num_results: int,
        active_clients_gauge: Any = None,
    ) -> None:
        """Record the current number of active clients on the OTEL gauge."""
        record_metric(active_clients_gauge, num_results, {"round": str(server_round)})

    def record_standard_round_metrics(
        self,
        server_round: int,
        *,
        num_results: int,
        aggregation_time: float,
        rounds_counter: Any = None,
        aggregations_counter: Any = None,
        aggregation_histogram: Any = None,
    ) -> None:
        """Record the metrics shared by all MQTT server strategies per round."""
        record_metric(aggregations_counter, 1, {"round": str(server_round)})
        record_metric(rounds_counter, 1)
        record_metric(
            aggregation_histogram,
            aggregation_time,
            {"round": str(server_round)},
        )

        FL_ROUNDS.labels(server=self.server_label).inc()
        FL_AGGREGATIONS.labels(server=self.server_label).inc()
        FL_ACTIVE_CLIENTS.labels(server=self.server_label).set(num_results)
        FL_ROUND_DURATION.labels(server=self.server_label).observe(aggregation_time)

    def print_final_evaluation_summary(self, *, loss: float, accuracy: float) -> None:
        """Render the shared final-evaluation banner."""
        if self.eval_data is None:
            return

        features, labels = self.eval_data
        n_samples = features.shape[0]
        n_classes = len(np.unique(labels))

        print(
            f"\n{self.tag} ╔══════════════════════════════════════════════════════════════╗"
        )
        print(
            f"{self.tag} ║     FEDERATED LEARNING COMPLETE - FINAL MODEL EVALUATION     ║"
        )
        print(
            f"{self.tag} ╠══════════════════════════════════════════════════════════════╣"
        )
        print(
            f"{self.tag} ║                                                              ║"
        )
        print(
            f"{self.tag} ║  Test Dataset:                                               ║"
        )
        print(
            f"{self.tag} ║    - Samples: {n_samples:>10,}                                   ║"
        )
        print(
            f"{self.tag} ║    - Classes: {n_classes:>10}                                   ║"
        )
        print(
            f"{self.tag} ║                                                              ║"
        )
        print(
            f"{self.tag} ║  Model Performance:                                          ║"
        )
        print(
            f"{self.tag} ║    ┌────────────────────────────────────────────────┐        ║"
        )
        print(
            f"{self.tag} ║    │  Loss:     {loss:>10.4f}                        │        ║"
        )
        print(
            f"{self.tag} ║    │  Accuracy: {accuracy:>10.4f}  ({accuracy*100:>6.2f}%)            │        ║"
        )
        print(
            f"{self.tag} ║    └────────────────────────────────────────────────┘        ║"
        )
        print(
            f"{self.tag} ║                                                              ║"
        )

        bar_len = 40
        filled = int(accuracy * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"{self.tag} ║  Accuracy: [{bar}]  ║")
        print(
            f"{self.tag} ║             0%                                   100%        ║"
        )
        print(
            f"{self.tag} ║                                                              ║"
        )
        print(
            f"{self.tag} ╚══════════════════════════════════════════════════════════════╝\n"
        )

    def log_round_banner(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> None:
        print(f"\n{self.tag} ╔══════════════════════════════════════════╗")
        print(
            f"{self.tag} ║           ROUND {server_round}/{self.total_rounds}                        ║"
        )
        print(f"{self.tag} ╚══════════════════════════════════════════╝")
        print(
            f"{self.tag} Received results from {len(results)} fog nodes, {len(failures)} failures"
        )

    def publish_global_model(
        self, server_round: int, state_dict: dict[str, Any]
    ) -> None:
        if self.mqtt is None:
            return

        with start_linked_producer_span(
            self.tracer,
            "server.publish_global_model",
            target_service=self.publish_target_service,
            attributes={"round": server_round},
        ) as (_span, trace_ctx):
            payload = build_global_model_payload(
                round_num=server_round,
                weights=state_dict,
                trace_context=trace_ctx,
            )
            self.mqtt.publish(self.model_topic, json.dumps(payload))
        print(f"{self.tag} Published global model on {self.model_topic}")

    def _run_final_evaluation(
        self, server_round: int, state_dict: dict[str, Any]
    ) -> None:
        if not self.should_evaluate_round(server_round):
            self.on_deferred_evaluation(server_round)
            return

        if self.eval_data is None:
            self.on_missing_final_eval_data(server_round)
            return

        try:
            evaluation = self.evaluate_aggregated_state(server_round, state_dict)
            self.on_final_evaluation(server_round, evaluation)
        except Exception as exc:
            print(f"{self.tag} Final evaluation failed: {exc}")
            traceback.print_exc()

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, fl.common.FitRes]],
        failures,
    ) -> fl.common.Parameters | None:
        start_time = time.time()
        self.log_round_banner(server_round, results, failures)
        self.before_aggregate_fit(server_round, results, failures)

        with start_server_span(
            self.tracer,
            "server.aggregate_fit",
            attributes={"round": server_round, "num_results": len(results)},
        ):
            new_parameters = super().aggregate_fit(server_round, results, failures)

        if new_parameters is None:
            print(f"{self.tag} Aggregation returned None")
            return None

        try:
            state_dict = extract_named_parameters(
                new_parameters,
                self.param_names,
                bytes_to_ndarray=fl.common.bytes_to_ndarray,
                parameters_to_ndarrays=fl.common.parameters_to_ndarrays,
            )
            self.publish_global_model(server_round, state_dict)
            self.on_aggregated_state_ready(server_round, state_dict, results)
            self._run_final_evaluation(server_round, state_dict)
        except Exception as exc:
            print(f"{self.tag} MQTT publish failed: {exc}")

        aggregation_time = time.time() - start_time
        self.record_aggregation_metrics(server_round, results, aggregation_time)
        return new_parameters
