"""Tests for the json and npz message codecs (issue #78)."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from onion_fl.core.codec import CODECS, get_codec
from onion_fl.core.message import Message, MessageError, Payload

CODEC_NAMES = ["json", "npz"]


def _update() -> Message:
    # Deterministic model-sized arrays: these tests serialise numbers, they train nothing.
    weight = (np.arange(64 * 32, dtype=np.float32) / 7).reshape(64, 32)
    return Message(
        kind="update",
        src="edge_1",
        dst="fog_0",
        round=7,
        payload=Payload(
            state={
                "trunk.0.weight": weight,
                "trunk.0.bias": np.arange(64, dtype=np.float32) / 3,
                "head.stress_binary.steps": np.arange(4, dtype=np.int64),
            },
            weights={"trunk.0.weight": 120, "trunk.0.bias": 120.5},
            metrics={"loss": 0.4213, "val_acc": 0.81},
        ),
        meta={"sent_at": 12.25, "trace_context": {"traceparent": "00-abc-01"}},
    )


def test_both_codecs_are_registered() -> None:
    assert set(CODECS) == {"json", "npz"}


def test_unknown_codec_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="json"):
        get_codec("protobuf")


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_roundtrip_preserves_the_message(name: str) -> None:
    codec = get_codec(name)
    msg = _update()

    decoded = codec.decode(codec.encode(msg))

    assert decoded == msg
    for key, array in msg.payload.state.items():
        assert decoded.payload.state[key].dtype == array.dtype
        assert decoded.payload.state[key].shape == array.shape


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_roundtrip_is_bit_exact_for_float32(name: str) -> None:
    codec = get_codec(name)
    msg = _update()

    decoded = codec.decode(codec.encode(msg))

    original = msg.payload.state["trunk.0.weight"].view(np.uint32)
    assert np.array_equal(
        decoded.payload.state["trunk.0.weight"].view(np.uint32), original
    )


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_roundtrip_of_an_empty_message(name: str) -> None:
    codec = get_codec(name)
    msg = Message(kind="hello", src="edge_1", dst="fog_0")

    assert codec.decode(codec.encode(msg)) == msg


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_size_is_the_length_of_the_encoding(name: str) -> None:
    codec = get_codec(name)
    msg = _update()

    assert codec.size(msg) == len(codec.encode(msg)) > 0


def test_npz_is_more_compact_than_json_for_model_sized_state() -> None:
    msg = _update()

    assert get_codec("npz").size(msg) < get_codec("json").size(msg)


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_encode_rejects_meta_that_is_not_json(name: str) -> None:
    msg = Message(kind="control", src="cloud", dst="fog_0", meta={"when": object()})

    with pytest.raises(MessageError, match="meta"):
        get_codec(name).encode(msg)


@pytest.mark.parametrize("name", CODEC_NAMES)
def test_decode_rejects_garbage(name: str) -> None:
    with pytest.raises(MessageError):
        get_codec(name).decode(b"\x00not a message\xff")


def _json_doc(**overrides) -> bytes:
    doc = json.loads(get_codec("json").encode(_update()))
    doc.update(overrides)
    return json.dumps(doc).encode()


@pytest.mark.parametrize(
    "data",
    [
        b'{"v": 1}',
        b"[1, 2, 3]",
        _json_doc(v=2),
        _json_doc(kind="gossip"),
        _json_doc(state={"w": {"dtype": "float32", "shape": [2, 2], "data": [1.0]}}),
        _json_doc(state={"w": {"dtype": "object", "shape": [1], "data": [1]}}),
        _json_doc(state={"w": {"dtype": "no-such-dtype", "shape": [1], "data": [1]}}),
    ],
    ids=[
        "missing-fields",
        "not-an-object",
        "unknown-version",
        "invalid-kind",
        "bad-shape",
        "object-dtype",
        "bad-dtype",
    ],
)
def test_json_decode_rejects_malformed_documents(data: bytes) -> None:
    with pytest.raises(MessageError):
        get_codec("json").decode(data)


UNPICKLED: list[str] = []


def _record_unpickle() -> str:
    UNPICKLED.append("executed")
    return "payload"


class _PickleBomb:
    """Runs code when unpickled; decoding bytes from the network must never do that."""

    def __reduce__(self):
        return (_record_unpickle, ())


def test_npz_decode_never_unpickles() -> None:
    buf = io.BytesIO()
    header = json.dumps(
        {"v": 1, "kind": "update", "src": "a", "dst": "b", "keys": ["w"]}
    )
    bomb = np.empty(1, dtype=object)
    bomb[0] = _PickleBomb()
    np.savez(buf, header=np.frombuffer(header.encode(), dtype=np.uint8), a0=bomb)
    UNPICKLED.clear()

    with pytest.raises(MessageError):
        get_codec("npz").decode(buf.getvalue())
    assert UNPICKLED == []


def test_npz_decode_rejects_non_numeric_arrays() -> None:
    buf = io.BytesIO()
    header = json.dumps(
        {"v": 1, "kind": "update", "src": "a", "dst": "b", "keys": ["w"]}
    )
    np.savez(
        buf,
        header=np.frombuffer(header.encode(), dtype=np.uint8),
        a0=np.array(["x", "y"]),
    )

    with pytest.raises(MessageError, match="non-numeric"):
        get_codec("npz").decode(buf.getvalue())


def test_npz_decode_rejects_an_archive_without_header() -> None:
    buf = io.BytesIO()
    np.savez(buf, a0=np.arange(3))

    with pytest.raises(MessageError):
        get_codec("npz").decode(buf.getvalue())
