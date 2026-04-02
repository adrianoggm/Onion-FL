from __future__ import annotations

"""Common workflow for MQTT partial-aggregate fog bridges."""

import time
import traceback
from typing import Any, Callable

import flwr as fl
import numpy as np
import torch

from flower_basic.clients.baseclient import BaseMQTTComponent
from flower_basic.runtime_protocol import (
    PartialAggregateEnvelope,
    decode_partial_aggregate_message,
)
from flower_basic.telemetry import (
    start_linked_client_span,
    start_linked_consumer_span,
)


class BaseFogBridgeClient(BaseMQTTComponent, fl.client.NumPyClient):
    """Shared MQTT-to-Flower bridge for fog partial aggregates."""

    def __init__(
        self,
        *,
        server_address: str,
        model: torch.nn.Module,
        get_parameters_fn: Callable[[torch.nn.Module], list[np.ndarray]],
        set_parameters_fn: Callable[[torch.nn.Module, list[np.ndarray]], None],
        region: str,
        tag: str,
        mqtt_broker: str,
        mqtt_port: int,
        partial_topic: str,
        tracer: Any = None,
        partial_source_service: str = "fog-broker",
        server_target_service: str = "server",
    ) -> None:
        self.server_address = server_address
        self.model = model
        self._get_parameters_fn = get_parameters_fn
        self._set_parameters_fn = set_parameters_fn
        self.param_names = list(self.model.state_dict().keys())
        self.partial_weights: dict[str, Any] | None = None
        self.partial_metadata: dict[str, Any] = {}
        self.partial_trace_context: dict[str, Any] | None = None
        self.partial_topic = partial_topic
        self.region = region
        self.tag = tag
        self.tracer = tracer
        self.partial_source_service = partial_source_service
        self.server_target_service = server_target_service

        super().__init__(
            tag=self.tag,
            mqtt_broker=mqtt_broker,
            mqtt_port=mqtt_port,
            subscriptions=[self.partial_topic],
        )
        print(f"{self.tag} Listening for partials on {self.partial_topic}")

    def build_partial_metadata(
        self, envelope: PartialAggregateEnvelope
    ) -> dict[str, Any]:
        """Build metadata forwarded to the central server."""
        return {}

    def build_timeout_metrics(self) -> dict[str, Any]:
        """Metrics returned when no partial arrives before timeout."""
        return {}

    def wait_timeout_seconds(self) -> float:
        """Timeout while waiting for a broker partial."""
        return 60.0

    def forwarded_num_samples(self) -> int:
        """Synthetic sample count used for forwarding aggregated partials."""
        return 1000

    def on_partial_received(self, envelope: PartialAggregateEnvelope) -> None:
        """Hook for logging/telemetry after a partial is buffered."""

    def on_wait_completed(self, wait_duration: float) -> None:
        """Hook for recording wait-duration metrics."""

    def on_timeout(self) -> None:
        """Hook for timeout telemetry."""

    def on_forwarded(self, num_samples: int) -> None:
        """Hook for forwarding telemetry."""

    def build_error_metrics(self, exc: Exception) -> dict[str, Any]:
        """Metrics returned when forwarding a partial fails unexpectedly."""
        return {
            "bridge_error": True,
            "bridge_error_type": type(exc).__name__,
            "region": self.region,
        }

    def on_message(self, client, userdata, msg) -> None:
        try:
            envelope = decode_partial_aggregate_message(msg.payload)
            if envelope is None:
                print(f"{self.tag} Ignoring malformed partial payload")
                return
            if envelope.region != self.region:
                return

            self.partial_weights = envelope.weights
            self.partial_metadata = self.build_partial_metadata(envelope)
            self.partial_trace_context = envelope.trace_context

            with start_linked_consumer_span(
                self.tracer,
                "bridge.receive_partial",
                self.partial_trace_context,
                source_service=self.partial_source_service,
                attributes={"region": envelope.region},
            ):
                self.on_partial_received(envelope)
                print(
                    f"{self.tag} Partial aggregate received for region={envelope.region}"
                )
        except Exception as exc:
            print(f"{self.tag} Error processing partial: {exc}")

    def get_parameters(self, config):
        return self._get_parameters_fn(self.model)

    def fit(self, parameters: list[np.ndarray], config):
        start_wait = time.time()
        span = None

        try:
            with start_linked_client_span(
                self.tracer,
                "bridge.forward_to_server",
                self.server_target_service,
                trace_context=self.partial_trace_context,
                attributes={"region": self.region},
            ) as active_span:
                span = active_span
                self._set_parameters_fn(self.model, parameters)
                timeout = self.wait_timeout_seconds()
                waited = 0.0
                while self.partial_weights is None and waited < timeout:
                    time.sleep(0.5)
                    waited += 0.5

                wait_duration = time.time() - start_wait
                self.on_wait_completed(wait_duration)

                if self.partial_weights is None:
                    print(f"{self.tag} Timeout waiting for partial")
                    if span:
                        span.set_attribute("status", "timeout")
                    self.on_timeout()
                    return (
                        self._get_parameters_fn(self.model),
                        1,
                        self.build_timeout_metrics(),
                    )

                partial_list = [
                    np.array(self.partial_weights[name], dtype=np.float32)
                    for name in self.param_names
                ]
                metrics = dict(self.partial_metadata)
                self.partial_weights = None
                self.partial_metadata = {}
                self.partial_trace_context = None
                num_samples = self.forwarded_num_samples()
                print(f"{self.tag} Forwarding partial to central server")
                if span:
                    span.set_attribute("status", "forwarded")
                    span.set_attribute("num_samples", num_samples)
                self.on_forwarded(num_samples)
                return partial_list, num_samples, metrics
        except Exception as exc:
            print(f"{self.tag} Error forwarding partial to central server: {exc}")
            traceback.print_exc()
            self.partial_weights = None
            self.partial_metadata = {}
            self.partial_trace_context = None
            if span:
                span.set_attribute("status", "error")
                span.set_attribute("error.type", type(exc).__name__)
            return (
                self._get_parameters_fn(self.model),
                1,
                self.build_error_metrics(exc),
            )

    def evaluate(self, parameters, config):
        return 0.0, 0, {}
