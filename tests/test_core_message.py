"""Tests for the transport-agnostic Message (issue #78)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from onion_fl.core.message import KINDS, Message, MessageError, Payload


def _state() -> dict[str, np.ndarray]:
    return {
        "trunk.0.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "trunk.0.bias": np.arange(2, dtype=np.float32),
    }


def test_kinds_cover_the_round_protocol() -> None:
    assert set(KINDS) == {
        "hello",
        "global_model",
        "update",
        "eval_request",
        "eval_report",
        "control",
    }


def test_update_message_holds_state_weights_metrics_and_meta() -> None:
    msg = Message(
        kind="update",
        src="edge_1",
        dst="fog_0",
        round=3,
        payload=Payload(
            state=_state(),
            weights={"trunk.0.weight": 120, "trunk.0.bias": 120},
            metrics={"loss": 0.42},
        ),
        meta={"sent_at": 12.5, "trace_context": {"traceparent": "00-abc"}},
    )

    assert msg.payload.weights["trunk.0.weight"] == 120
    assert msg.payload.metrics == {"loss": 0.42}
    assert msg.meta["trace_context"] == {"traceparent": "00-abc"}


def test_defaults_give_an_empty_payload_and_no_round() -> None:
    msg = Message(kind="hello", src="edge_1", dst="fog_0")

    assert msg.round is None
    assert msg.payload == Payload()
    assert msg.meta == {}


def test_message_is_immutable() -> None:
    msg = Message(kind="hello", src="edge_1", dst="fog_0")

    with pytest.raises(dataclasses.FrozenInstanceError):
        msg.round = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"kind": "gossip"}, "kind"),
        ({"src": ""}, "src"),
        ({"dst": ""}, "dst"),
        ({"round": -1}, "round"),
        ({"round": 1.5}, "round"),
        ({"round": True}, "round"),
    ],
)
def test_invalid_header_fields_are_rejected(kwargs: dict, fragment: str) -> None:
    fields = {"kind": "update", "src": "edge_1", "dst": "fog_0"} | kwargs

    with pytest.raises(MessageError, match=fragment):
        Message(**fields)


def test_state_values_must_be_numeric_arrays() -> None:
    with pytest.raises(MessageError, match="state"):
        Payload(state={"w": [1.0, 2.0]})  # type: ignore[dict-item]
    with pytest.raises(MessageError, match="state"):
        Payload(state={"w": np.array(["a", "b"])})
    with pytest.raises(MessageError, match="state"):
        Payload(state={"w": np.array([object()], dtype=object)})


def test_weights_only_for_keys_present_in_state() -> None:
    with pytest.raises(MessageError, match="weights"):
        Payload(state=_state(), weights={"head.weight": 10})


def test_weights_must_be_non_negative_numbers() -> None:
    with pytest.raises(MessageError, match="weights"):
        Payload(state=_state(), weights={"trunk.0.bias": -1})
    with pytest.raises(MessageError, match="weights"):
        Payload(state=_state(), weights={"trunk.0.bias": "10"})  # type: ignore[dict-item]


def test_payload_and_meta_types_are_checked() -> None:
    with pytest.raises(MessageError, match="payload"):
        Message(kind="update", src="a", dst="b", payload={"state": {}})  # type: ignore[arg-type]
    with pytest.raises(MessageError, match="meta"):
        Message(kind="update", src="a", dst="b", meta=[("k", 1)])  # type: ignore[arg-type]


def test_payloads_differ_on_keys_values_or_dtype() -> None:
    base = Payload(state=_state())

    assert base != Payload(state={"trunk.0.bias": _state()["trunk.0.bias"]})
    assert base != Payload(
        state=_state() | {"trunk.0.bias": np.ones(2, dtype=np.float32)}
    )
    assert base != Payload(
        state=_state() | {"trunk.0.bias": np.arange(2, dtype=np.float64)}
    )
    assert base != "not a payload"


def test_metrics_must_be_numbers() -> None:
    with pytest.raises(MessageError, match="metrics"):
        Payload(metrics={"loss": "low"})  # type: ignore[dict-item]
