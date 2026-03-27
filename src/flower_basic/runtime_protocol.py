from __future__ import annotations

"""Pure helpers for FL runtime payloads and round metadata."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GlobalModelEnvelope:
    """Decoded server-to-client global model message."""

    round_num: int | None
    weights: dict[str, Any]
    trace_context: dict[str, Any]


@dataclass(frozen=True)
class StalenessSummary:
    """Aggregate view of stale/future updates reported by fog nodes."""

    stale_updates: int = 0
    future_updates: int = 0
    max_delay_seconds: float = 0.0
    max_round_span: int = 0


@dataclass(frozen=True)
class ClientUpdateEnvelope:
    """Decoded client-to-broker local update message."""

    client_id: str
    region: str
    weights: dict[str, Any]
    num_samples: int
    loss: float | None
    val_acc: float | None
    round_num: int | None
    sent_at: float | None
    trace_context: dict[str, Any]


@dataclass(frozen=True)
class PartialAggregateEnvelope:
    """Decoded broker-to-bridge partial aggregate message."""

    region: str
    weights: dict[str, Any]
    total_samples: int
    trace_context: dict[str, Any]
    timestamp: float | None
    expected_round: int | None
    round_min: int | None
    round_max: int | None
    stale_update_count: int
    future_update_count: int
    max_delay_seconds: float
    mean_delay_seconds: float
    stale_policy: str | None


@dataclass(frozen=True)
class UpdateBatchSummary:
    """Batch-level sample, round, and delay summary for broker aggregation."""

    total_samples: int
    sample_weights: list[float] | None
    round_min: int | None
    round_max: int | None
    stale_update_count: int = 0
    future_update_count: int = 0
    max_delay_seconds: float = 0.0
    mean_delay_seconds: float = 0.0


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _coerce_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value: Any, fallback: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _payload_to_text(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode()
    if isinstance(payload, str):
        return payload
    decode = getattr(payload, "decode", None)
    if callable(decode):
        try:
            decoded = decode()
        except TypeError:
            decoded = decode("utf-8")
        if isinstance(decoded, (bytes, bytearray)):
            return decoded.decode()
        return str(decoded)
    return str(payload)


def serialize_named_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Detach model weights into JSON-serializable primitives."""
    return {str(name): _to_jsonable(value) for name, value in weights.items()}


def build_client_update_payload(
    *,
    client_id: str,
    region: str,
    weights: Mapping[str, Any],
    num_samples: int,
    avg_loss: float,
    val_acc: float = 0.0,
    round_num: int | None = None,
    sent_at: float | None = None,
    trace_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload sent by a local client to the fog broker."""
    payload = {
        "client_id": client_id,
        "region": region,
        "weights": serialize_named_weights(weights),
        "num_samples": int(num_samples),
        "loss": float(avg_loss),
        "val_acc": float(val_acc),
        "trace_context": _as_mapping(trace_context),
    }
    if round_num is not None:
        payload["round"] = int(round_num)
    if sent_at is not None:
        payload["sent_at"] = float(sent_at)
    return payload


def build_global_model_payload(
    *,
    round_num: int,
    weights: Mapping[str, Any],
    trace_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the payload sent by the server to local clients."""
    return {
        "round": int(round_num),
        "global_weights": serialize_named_weights(weights),
        "trace_context": _as_mapping(trace_context),
    }


def build_partial_aggregate_payload(
    *,
    region: str,
    partial_weights: Mapping[str, Any],
    total_samples: int,
    timestamp: float,
    trace_context: Mapping[str, Any] | None = None,
    expected_round: int | None = None,
    round_min: int | None = None,
    round_max: int | None = None,
    stale_update_count: int | None = None,
    future_update_count: int | None = None,
    max_delay_seconds: float | None = None,
    mean_delay_seconds: float | None = None,
    stale_policy: str | None = None,
) -> dict[str, Any]:
    """Build the payload sent by a fog broker to a Flower bridge."""
    payload = {
        "region": region,
        "partial_weights": serialize_named_weights(partial_weights),
        "total_samples": int(total_samples),
        "timestamp": float(timestamp),
        "trace_context": _as_mapping(trace_context),
    }
    if expected_round is not None:
        payload["expected_round"] = int(expected_round)
    if round_min is not None:
        payload["round_min"] = int(round_min)
    if round_max is not None:
        payload["round_max"] = int(round_max)
    if stale_update_count is not None:
        payload["stale_update_count"] = int(stale_update_count)
    if future_update_count is not None:
        payload["future_update_count"] = int(future_update_count)
    if max_delay_seconds is not None:
        payload["max_delay_seconds"] = float(max_delay_seconds)
    if mean_delay_seconds is not None:
        payload["mean_delay_seconds"] = float(mean_delay_seconds)
    if stale_policy is not None:
        payload["stale_policy"] = str(stale_policy)
    return payload


def decode_global_model_message(
    payload_bytes: bytes | bytearray | str,
    allowed_param_names: Iterable[str],
) -> GlobalModelEnvelope | None:
    """Decode and filter a server global-model message for a specific client."""
    raw_payload = _payload_to_text(payload_bytes)

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None

    raw_weights = parsed.get("global_weights")
    if not isinstance(raw_weights, Mapping):
        return None

    allowed = set(allowed_param_names)
    filtered_weights = {
        str(name): _to_jsonable(value)
        for name, value in raw_weights.items()
        if str(name) in allowed
    }
    if not filtered_weights:
        return None

    round_num = parsed.get("round")
    if round_num is not None:
        round_num = int(round_num)

    return GlobalModelEnvelope(
        round_num=round_num,
        weights=filtered_weights,
        trace_context=_as_mapping(parsed.get("trace_context")),
    )


def decode_client_update_message(
    payload_bytes: bytes | bytearray | str,
    *,
    fallback_round: int | None = None,
    fallback_sent_at: float | None = None,
    default_region: str = "default_region",
    default_client_id: str = "unknown",
) -> ClientUpdateEnvelope | None:
    """Decode a local client update sent to a fog broker."""
    raw_payload = _payload_to_text(payload_bytes)

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None

    raw_weights = parsed.get("weights", {})
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, Mapping):
        return None

    return ClientUpdateEnvelope(
        client_id=str(parsed.get("client_id", default_client_id)),
        region=str(parsed.get("region", default_region)),
        weights=serialize_named_weights(raw_weights),
        num_samples=_coerce_int(parsed.get("num_samples"), 0) or 0,
        loss=_coerce_float(parsed.get("loss")),
        val_acc=_coerce_float(parsed.get("val_acc")),
        round_num=_coerce_int(parsed.get("round"), fallback_round),
        sent_at=_coerce_float(parsed.get("sent_at"), fallback_sent_at),
        trace_context=_as_mapping(parsed.get("trace_context")),
    )


def decode_partial_aggregate_message(
    payload_bytes: bytes | bytearray | str,
    *,
    default_region: str = "unknown",
) -> PartialAggregateEnvelope | None:
    """Decode a fog partial aggregate sent to a Flower bridge."""
    raw_payload = _payload_to_text(payload_bytes)

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        return None

    raw_weights = parsed.get("partial_weights", {})
    if raw_weights is None:
        raw_weights = {}
    if not isinstance(raw_weights, Mapping):
        return None

    return PartialAggregateEnvelope(
        region=str(parsed.get("region", default_region)),
        weights=serialize_named_weights(raw_weights),
        total_samples=_coerce_int(parsed.get("total_samples"), 0) or 0,
        trace_context=_as_mapping(parsed.get("trace_context")),
        timestamp=_coerce_float(parsed.get("timestamp")),
        expected_round=_coerce_int(parsed.get("expected_round")),
        round_min=_coerce_int(parsed.get("round_min")),
        round_max=_coerce_int(parsed.get("round_max")),
        stale_update_count=_coerce_int(parsed.get("stale_update_count"), 0) or 0,
        future_update_count=_coerce_int(parsed.get("future_update_count"), 0) or 0,
        max_delay_seconds=_coerce_float(parsed.get("max_delay_seconds"), 0.0) or 0.0,
        mean_delay_seconds=_coerce_float(
            parsed.get("mean_delay_seconds"), 0.0
        )
        or 0.0,
        stale_policy=(
            None
            if parsed.get("stale_policy") is None
            else str(parsed.get("stale_policy"))
        ),
    )


def summarize_update_batch(
    updates: Iterable[Any], *, expected_round: int | None = None
) -> UpdateBatchSummary:
    """Summarize batch sample weights plus optional round/delay metadata."""
    sample_counts: list[int] = []
    round_values: list[int] = []
    delays: list[float] = []
    stale_update_count = 0
    future_update_count = 0

    for raw_item in updates:
        item = _as_mapping(raw_item)
        sample_count = _coerce_int(item.get("num_samples"), 0) or 0
        sample_counts.append(sample_count)

        round_num = _coerce_int(item.get("round"), expected_round)
        if round_num is not None:
            round_values.append(round_num)
            if expected_round is not None:
                if round_num < expected_round:
                    stale_update_count += 1
                elif round_num > expected_round:
                    future_update_count += 1

        sent_at = _coerce_float(item.get("sent_at"))
        received_at = _coerce_float(item.get("received_at"))
        if sent_at is not None and received_at is not None:
            delays.append(max(0.0, received_at - sent_at))

    total_samples = sum(sample_counts)
    sample_weights = None
    if total_samples > 0:
        sample_weights = [count / total_samples for count in sample_counts]

    round_min = min(round_values) if round_values else expected_round
    round_max = max(round_values) if round_values else expected_round
    max_delay_seconds = max(delays) if delays else 0.0
    mean_delay_seconds = sum(delays) / len(delays) if delays else 0.0

    return UpdateBatchSummary(
        total_samples=total_samples,
        sample_weights=sample_weights,
        round_min=round_min,
        round_max=round_max,
        stale_update_count=stale_update_count,
        future_update_count=future_update_count,
        max_delay_seconds=max_delay_seconds,
        mean_delay_seconds=mean_delay_seconds,
    )


def extract_named_parameters(
    parameters: Any,
    param_names: Iterable[str],
    *,
    bytes_to_ndarray,
    parameters_to_ndarrays,
) -> dict[str, Any]:
    """Normalize Flower parameters into a named weight mapping."""
    parameters_obj = parameters[0] if isinstance(parameters, tuple) else parameters

    if hasattr(parameters_obj, "tensors"):
        param_arrays = [bytes_to_ndarray(tensor) for tensor in parameters_obj.tensors]
    else:
        param_arrays = list(parameters_to_ndarrays(parameters_obj))

    names = list(param_names)
    return {
        name: param_arrays[idx]
        for idx, name in enumerate(names)
        if idx < len(param_arrays)
    }


def summarize_staleness_metrics(
    metrics_payloads: Iterable[Any], server_round: int
) -> StalenessSummary:
    """Aggregate staleness metadata from per-client metrics payloads."""
    stale_updates = 0
    future_updates = 0
    max_delay_seconds = 0.0
    max_round_span = 0

    for raw_metrics in metrics_payloads:
        metrics = _as_mapping(raw_metrics)
        stale_updates += int(metrics.get("stale_update_count", 0))
        future_updates += int(metrics.get("future_update_count", 0))
        max_delay_seconds = max(
            max_delay_seconds, float(metrics.get("max_delay_seconds", 0.0))
        )
        round_min = int(metrics.get("round_min", server_round))
        round_max = int(metrics.get("round_max", server_round))
        max_round_span = max(max_round_span, round_max - round_min)

    return StalenessSummary(
        stale_updates=stale_updates,
        future_updates=future_updates,
        max_delay_seconds=max_delay_seconds,
        max_round_span=max_round_span,
    )
