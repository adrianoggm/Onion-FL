from __future__ import annotations

"""Shared helpers for regional MQTT aggregation brokers."""

import json
import time
import traceback
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from flower_basic.runtime_protocol import (
    build_partial_aggregate_payload,
    decode_client_update_message,
    summarize_update_batch,
)
from flower_basic.telemetry import (
    record_metric,
    start_linked_consumer_span,
    start_linked_producer_span,
    start_server_span,
)


@dataclass(frozen=True)
class BrokerConfig:
    """Static configuration for a broker update-processing flow."""

    broker_tag: str
    source_service: str
    target_service: str
    partial_topic: str
    default_k: int
    k_map: Mapping[str, int]
    use_round_metadata: bool = False
    stale_update_policy: str = "accept"


@dataclass(frozen=True)
class BrokerTelemetryHandles:
    """Telemetry handles used by broker helpers."""

    tracer: Any = None
    counter_updates_received: Any = None
    counter_partials_published: Any = None
    hist_aggregation_time: Any = None
    gauge_buffer_size: Any = None
    gauge_client_contribution: Any = None
    counter_aggregations_total: Any = None
    gauge_clients_per_region: Any = None


@dataclass(frozen=True)
class BrokerCallbacks:
    """Callbacks for dataset-specific metrics and logging."""

    record_prometheus_update: Callable[[str, str, int, int, int], None]
    record_prometheus_buffer_cleared: Callable[[str], None]
    record_prometheus_aggregation: Callable[[str], None]
    record_region_model_metrics: Callable[[str, dict[str, float], int], None]


def weighted_average(
    updates: list[dict[str, Any]], weights: list[float] | None = None
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute a weighted average plus centroid statistics for model updates."""
    n_updates = len(updates)
    if weights is None:
        weights = [1.0 / n_updates] * n_updates

    averaged: dict[str, Any] = {}
    flattened_params = []
    for key in updates[0]:
        param_arrays = [np.array(update[key]) for update in updates]
        stacked = np.stack(param_arrays, axis=0)
        weights_array = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
        mean_param = (stacked * weights_array).sum(axis=0)
        averaged[key] = mean_param.tolist()
        flattened_params.append(mean_param.flatten())

    all_weights = np.concatenate(flattened_params)
    centroid_stats = {
        "norm": float(np.linalg.norm(all_weights)),
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "num_params": len(all_weights),
    }
    return averaged, centroid_stats


def expected_client_round(latest_global_round: int) -> int:
    """Calculate the round clients are expected to send next."""
    return max(1, int(latest_global_round) + 1)


def handle_global_model_round_update(
    payload_bytes: bytes | bytearray | str,
    *,
    latest_global_round: int,
    broker_tag: str,
) -> int:
    """Track the latest global round announced by the central server."""
    try:
        payload = json.loads(
            payload_bytes.decode()
            if hasattr(payload_bytes, "decode")
            else payload_bytes
        )
        if not isinstance(payload, dict):
            raise TypeError("global model payload must be a JSON object")
        global_round = int(payload.get("round", latest_global_round))
        updated_round = max(latest_global_round, global_round)
        print(
            f"{broker_tag} Global round updated to {updated_round}; "
            f"expected client round={expected_client_round(updated_round)}"
        )
        return updated_round
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"{broker_tag} Ignoring malformed global model message: {exc}")
        return latest_global_round


def parse_k_map(k_map_text: str | None, *, broker_tag: str) -> dict[str, int]:
    """Parse an optional JSON k-map from CLI/env."""
    if not k_map_text:
        return {}
    try:
        parsed = json.loads(k_map_text)
        return {str(key): max(1, int(value)) for key, value in parsed.items()}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{broker_tag} Ignorando k-map inválido: {exc}")
        return {}


def handle_client_update(
    *,
    client,
    msg,
    config: BrokerConfig,
    telemetry: BrokerTelemetryHandles,
    callbacks: BrokerCallbacks,
    buffers: MutableMapping[str, list[dict[str, Any]]],
    clients_per_region: MutableMapping[str, set[str]],
    weighted_average_fn: Callable[
        [list[dict[str, Any]], list[float] | None],
        tuple[dict[str, Any], dict[str, float]],
    ],
    latest_global_round: int = 0,
) -> None:
    """Handle one local client update and possibly emit a partial aggregate."""
    try:
        received_at = time.time()
        expected_round = expected_client_round(latest_global_round)
        envelope = decode_client_update_message(
            msg.payload,
            fallback_round=(expected_round if config.use_round_metadata else None),
            fallback_sent_at=(received_at if config.use_round_metadata else None),
        )
        if envelope is None:
            print(f"{config.broker_tag} Ignoring malformed update payload")
            return

        region = envelope.region
        weights = envelope.weights
        client_id = envelope.client_id
        num_samples = envelope.num_samples
        update_round = (
            envelope.round_num if envelope.round_num is not None else expected_round
        )
        sent_at = envelope.sent_at if envelope.sent_at is not None else received_at
        trace_context = envelope.trace_context

        if not weights:
            print(f"{config.broker_tag} Received empty weights from {client_id}")
            return

        round_status = "current"
        if config.use_round_metadata:
            if update_round < expected_round:
                round_status = "stale"
            elif update_round > expected_round:
                round_status = "future"

            if config.stale_update_policy == "strict" and round_status != "current":
                print(
                    f"{config.broker_tag} Dropping {round_status} update from "
                    f"client={client_id}, region={region}, update_round={update_round}, "
                    f"expected_round={expected_round}"
                )
                return

        consumer_attributes = {
            "region": region,
            "client_id": client_id,
            "num_samples": num_samples,
        }
        if config.use_round_metadata:
            consumer_attributes["update_round"] = update_round
            consumer_attributes["expected_round"] = expected_round

        with start_linked_consumer_span(
            telemetry.tracer,
            "broker.receive_update",
            trace_context,
            source_service=config.source_service,
            attributes=consumer_attributes,
        ):
            buffers[region].append(
                {
                    "weights": weights,
                    "num_samples": num_samples,
                    "client_id": client_id,
                    "round": update_round,
                    "expected_round": expected_round,
                    "sent_at": sent_at,
                    "received_at": received_at,
                    "trace_context": trace_context,
                }
            )
            region_k = config.k_map.get(region, config.default_k)
            clients_per_region[region].add(client_id)

            record_metric(
                telemetry.counter_updates_received,
                1,
                {"region": region, "client_id": client_id},
            )
            record_metric(
                telemetry.gauge_buffer_size,
                len(buffers[region]),
                {"region": region},
            )
            record_metric(
                telemetry.gauge_client_contribution,
                num_samples,
                {"region": region, "client_id": client_id},
            )
            record_metric(
                telemetry.gauge_clients_per_region,
                len(clients_per_region[region]),
                {"region": region},
            )

            callbacks.record_prometheus_update(
                region,
                client_id,
                num_samples,
                len(buffers[region]),
                len(clients_per_region[region]),
            )

            message = (
                f"{config.broker_tag} Update received from client={client_id}, "
                f"region={region}, samples={num_samples}. "
                f"Buffer: {len(buffers[region])}/{region_k}"
            )
            if config.use_round_metadata:
                message += f" | round={update_round} expected={expected_round} status={round_status}"
            print(message)

            if len(buffers[region]) < region_k:
                return

            agg_start = time.time()
            batch = list(buffers[region])
            weight_list = [item["weights"] for item in batch]
            batch_summary = summarize_update_batch(
                batch,
                expected_round=(expected_round if config.use_round_metadata else None),
            )
            total_samples = batch_summary.total_samples
            round_min = (
                batch_summary.round_min
                if batch_summary.round_min is not None
                else expected_round
            )
            round_max = (
                batch_summary.round_max
                if batch_summary.round_max is not None
                else expected_round
            )

            with start_server_span(
                telemetry.tracer,
                "broker.aggregate",
                attributes={
                    "region": region,
                    "num_clients": len(weight_list),
                    "total_samples": total_samples,
                },
            ):
                partial, centroid_stats = weighted_average_fn(
                    weight_list, batch_summary.sample_weights
                )
                agg_duration = time.time() - agg_start

            callbacks.record_region_model_metrics(region, centroid_stats, total_samples)
            print(
                f"{config.broker_tag} Centroid stats for {region}: "
                f"norm={centroid_stats['norm']:.4f}, "
                f"mean={centroid_stats['mean']:.6f}, std={centroid_stats['std']:.4f}, "
                f"params={centroid_stats['num_params']}"
            )
            if config.use_round_metadata:
                print(
                    f"{config.broker_tag} Round summary for {region}: expected={expected_round}, "
                    f"min={round_min}, max={round_max}, stale={batch_summary.stale_update_count}, "
                    f"future={batch_summary.future_update_count}, "
                    f"max_delay={batch_summary.max_delay_seconds:.2f}s"
                )

            for item in batch:
                contribution_pct = (
                    item["num_samples"] / total_samples * 100
                    if total_samples > 0
                    else 0
                )
                print(
                    f"{config.broker_tag} Client {item['client_id']} contributed "
                    f"{item['num_samples']} samples ({contribution_pct:.1f}%)"
                )

            buffers[region].clear()
            record_metric(telemetry.gauge_buffer_size, 0, {"region": region})
            callbacks.record_prometheus_buffer_cleared(region)

            payload_kwargs = {
                "region": region,
                "partial_weights": partial,
                "total_samples": total_samples,
                "timestamp": time.time(),
            }
            with start_linked_producer_span(
                telemetry.tracer,
                "broker.publish_partial",
                target_service=config.target_service,
                attributes={"region": region, "total_samples": total_samples},
            ) as (_span, trace_ctx):
                payload_kwargs["trace_context"] = trace_ctx
                if config.use_round_metadata:
                    payload_kwargs.update(
                        {
                            "expected_round": expected_round,
                            "round_min": round_min,
                            "round_max": round_max,
                            "stale_update_count": batch_summary.stale_update_count,
                            "future_update_count": batch_summary.future_update_count,
                            "max_delay_seconds": batch_summary.max_delay_seconds,
                            "mean_delay_seconds": batch_summary.mean_delay_seconds,
                            "stale_policy": config.stale_update_policy,
                        }
                    )
                msg_payload = build_partial_aggregate_payload(**payload_kwargs)
                client.publish(config.partial_topic, json.dumps(msg_payload))

            record_metric(telemetry.counter_partials_published, 1, {"region": region})
            record_metric(telemetry.counter_aggregations_total, 1, {"region": region})
            if telemetry.hist_aggregation_time:
                telemetry.hist_aggregation_time.record(agg_duration, {"region": region})

            callbacks.record_prometheus_aggregation(region)
            print(
                f"{config.broker_tag} Partial aggregate published for region={region} "
                f"(total samples: {total_samples})"
            )
    except Exception as exc:
        print(f"{config.broker_tag} ERROR: Error procesando actualización: {exc}")
        traceback.print_exc()
