from __future__ import annotations

import json

from flower_basic.runtime_protocol import (
    build_client_update_payload,
    build_global_model_payload,
    build_partial_aggregate_payload,
    decode_client_update_message,
    decode_global_model_message,
    decode_partial_aggregate_message,
    extract_named_parameters,
    summarize_staleness_metrics,
    summarize_update_batch,
)


class _FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def test_build_client_update_payload_serializes_optional_fields() -> None:
    payload = build_client_update_payload(
        client_id="c1",
        region="fog_0",
        weights={"layer": _FakeArray([1.0, 2.0])},
        num_samples=4,
        avg_loss=0.25,
        val_acc=0.5,
        round_num=2,
        sent_at=123.4,
        trace_context={"trace_id": "abc"},
    )

    assert payload == {
        "client_id": "c1",
        "region": "fog_0",
        "weights": {"layer": [1.0, 2.0]},
        "num_samples": 4,
        "loss": 0.25,
        "val_acc": 0.5,
        "round": 2,
        "sent_at": 123.4,
        "trace_context": {"trace_id": "abc"},
    }


def test_build_client_update_payload_omits_optional_fields() -> None:
    payload = build_client_update_payload(
        client_id="c1",
        region="fog_0",
        weights={"bias": [1]},
        num_samples=1,
        avg_loss=1.0,
    )

    assert "round" not in payload
    assert "sent_at" not in payload
    assert payload["trace_context"] == {}


def test_build_global_model_payload_serializes_weights() -> None:
    payload = build_global_model_payload(
        round_num=3,
        weights={"layer": _FakeArray([3, 4])},
        trace_context={"span_id": "s1"},
    )

    assert payload == {
        "round": 3,
        "global_weights": {"layer": [3, 4]},
        "trace_context": {"span_id": "s1"},
    }


def test_build_partial_aggregate_payload_omits_optional_fields() -> None:
    payload = build_partial_aggregate_payload(
        region="fog_0",
        partial_weights={"layer": _FakeArray([3, 4])},
        total_samples=7,
        timestamp=99.5,
        trace_context={"trace_id": "t1"},
    )

    assert payload == {
        "region": "fog_0",
        "partial_weights": {"layer": [3, 4]},
        "total_samples": 7,
        "timestamp": 99.5,
        "trace_context": {"trace_id": "t1"},
    }


def test_decode_global_model_message_filters_unknown_weights() -> None:
    payload = {
        "round": 4,
        "global_weights": {"keep": [1, 2], "drop": [9]},
        "trace_context": {"trace_id": "abc"},
    }

    decoded = decode_global_model_message(
        json.dumps(payload).encode("utf-8"), allowed_param_names={"keep"}
    )

    assert decoded is not None
    assert decoded.round_num == 4
    assert decoded.weights == {"keep": [1, 2]}
    assert decoded.trace_context == {"trace_id": "abc"}


def test_decode_global_model_message_rejects_invalid_json() -> None:
    decoded = decode_global_model_message(b"not-json", allowed_param_names={"keep"})

    assert decoded is None


def test_decode_global_model_message_rejects_without_matching_weights() -> None:
    payload = {"round": 1, "global_weights": {"drop": [9]}}

    decoded = decode_global_model_message(
        json.dumps(payload), allowed_param_names={"keep"}
    )

    assert decoded is None


def test_decode_global_model_message_rejects_non_mapping_weights() -> None:
    payload = {"round": 1, "global_weights": [1, 2, 3]}

    decoded = decode_global_model_message(
        json.dumps(payload), allowed_param_names={"keep"}
    )

    assert decoded is None


def test_decode_client_update_message_applies_fallbacks() -> None:
    payload = {
        "client_id": "c1",
        "weights": {"layer": [1, 2]},
        "num_samples": "5",
        "loss": "0.25",
        "val_acc": "0.75",
        "trace_context": {"trace_id": "abc"},
    }

    decoded = decode_client_update_message(
        json.dumps(payload),
        fallback_round=3,
        fallback_sent_at=44.5,
    )

    assert decoded is not None
    assert decoded.client_id == "c1"
    assert decoded.region == "default_region"
    assert decoded.weights == {"layer": [1, 2]}
    assert decoded.num_samples == 5
    assert decoded.loss == 0.25
    assert decoded.val_acc == 0.75
    assert decoded.round_num == 3
    assert decoded.sent_at == 44.5
    assert decoded.trace_context == {"trace_id": "abc"}


def test_decode_client_update_message_rejects_non_mapping_weights() -> None:
    payload = {"weights": [1, 2, 3]}

    decoded = decode_client_update_message(json.dumps(payload))

    assert decoded is None


def test_decode_partial_aggregate_message_parses_optional_metadata() -> None:
    payload = {
        "region": "fog_0",
        "partial_weights": {"layer": [1, 2]},
        "total_samples": 9,
        "timestamp": 12.0,
        "trace_context": {"span_id": "s1"},
        "expected_round": 4,
        "round_min": 3,
        "round_max": 5,
        "stale_update_count": 2,
        "future_update_count": 1,
        "max_delay_seconds": 1.5,
        "mean_delay_seconds": 0.75,
        "stale_policy": "accept",
    }

    decoded = decode_partial_aggregate_message(json.dumps(payload))

    assert decoded is not None
    assert decoded.region == "fog_0"
    assert decoded.weights == {"layer": [1, 2]}
    assert decoded.total_samples == 9
    assert decoded.timestamp == 12.0
    assert decoded.trace_context == {"span_id": "s1"}
    assert decoded.expected_round == 4
    assert decoded.round_min == 3
    assert decoded.round_max == 5
    assert decoded.stale_update_count == 2
    assert decoded.future_update_count == 1
    assert decoded.max_delay_seconds == 1.5
    assert decoded.mean_delay_seconds == 0.75
    assert decoded.stale_policy == "accept"


def test_decode_partial_aggregate_message_rejects_non_mapping_weights() -> None:
    payload = {"partial_weights": [1, 2, 3]}

    decoded = decode_partial_aggregate_message(json.dumps(payload))

    assert decoded is None


def test_summarize_update_batch_aggregates_samples_rounds_and_delays() -> None:
    summary = summarize_update_batch(
        [
            {"num_samples": 3, "round": 1, "sent_at": 10.0, "received_at": 11.5},
            {"num_samples": 1, "round": 3, "sent_at": 10.0, "received_at": 10.5},
            {"num_samples": 2, "round": 2},
        ],
        expected_round=2,
    )

    assert summary.total_samples == 6
    assert summary.sample_weights == [0.5, 1 / 6, 1 / 3]
    assert summary.round_min == 1
    assert summary.round_max == 3
    assert summary.stale_update_count == 1
    assert summary.future_update_count == 1
    assert summary.max_delay_seconds == 1.5
    assert summary.mean_delay_seconds == 1.0


def test_summarize_update_batch_defaults_without_samples() -> None:
    summary = summarize_update_batch([{"num_samples": 0}], expected_round=5)

    assert summary.total_samples == 0
    assert summary.sample_weights is None
    assert summary.round_min == 5
    assert summary.round_max == 5
    assert summary.stale_update_count == 0
    assert summary.future_update_count == 0


def test_summarize_staleness_metrics_aggregates_and_defaults() -> None:
    summary = summarize_staleness_metrics(
        [
            {
                "stale_update_count": 2,
                "future_update_count": 1,
                "max_delay_seconds": 1.25,
                "round_min": 1,
                "round_max": 3,
            },
            {
                "stale_update_count": 1,
                "future_update_count": 0,
                "max_delay_seconds": 0.5,
                "round_min": 2,
                "round_max": 4,
            },
            None,
            "invalid",
        ],
        server_round=2,
    )

    assert summary.stale_updates == 3
    assert summary.future_updates == 1
    assert summary.max_delay_seconds == 1.25
    assert summary.max_round_span == 2


def test_extract_named_parameters_supports_tuple_payload() -> None:
    converted = []

    def fake_parameters_to_ndarrays(parameters):
        converted.append(parameters)
        return [[1, 2], [3, 4]]

    named = extract_named_parameters(
        ("legacy-params", {"ignored": True}),
        ["a", "b", "c"],
        bytes_to_ndarray=lambda value: value,
        parameters_to_ndarrays=fake_parameters_to_ndarrays,
    )

    assert converted == ["legacy-params"]
    assert named == {"a": [1, 2], "b": [3, 4]}


def test_extract_named_parameters_supports_tensor_payload() -> None:
    class _TensorParams:
        def __init__(self):
            self.tensors = [b"a", b"b"]

    named = extract_named_parameters(
        _TensorParams(),
        ["first", "second"],
        bytes_to_ndarray=lambda value: value.decode("utf-8"),
        parameters_to_ndarrays=lambda value: [],
    )

    assert named == {"first": "a", "second": "b"}
