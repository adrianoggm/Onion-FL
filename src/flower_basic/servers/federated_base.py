from __future__ import annotations

"""Common workflow for MQTT-enabled Flower server strategies."""

import json
import time
import traceback
from typing import Any

import flwr as fl
import paho.mqtt.client as mqtt

from flower_basic.runtime_protocol import (
    build_global_model_payload,
    extract_named_parameters,
)
from flower_basic.telemetry import start_linked_producer_span, start_server_span


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
        self.tag = tag
        self.history = self.initialize_history()
        self.log_eval_data_status()

    def initialize_history(self) -> dict[str, list[Any]]:
        """Initial aggregation history structure."""
        return {"round": [], "loss": [], "accuracy": []}

    def describe_eval_data(self) -> str | None:
        """Human-readable description used when eval data is available."""
        return None

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
        """Hook for telemetry and metrics after a round completes."""

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
        print(f"{self.tag} WARNING: FINAL ROUND: No test data available for evaluation!")

    def on_deferred_evaluation(self, server_round: int) -> None:
        print(
            f"{self.tag} Round {server_round}/{self.total_rounds} complete. "
            "Evaluation will run after final round."
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
