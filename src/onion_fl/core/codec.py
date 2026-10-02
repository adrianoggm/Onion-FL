from __future__ import annotations

"""Wire formats for ``Message`` (spec §5.1). Each link picks one by name.

The encoded size is what the network model and the communication metrics use,
so ``size()`` is the length of the real encoding, not an estimate.
"""

import io
import json
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from onion_fl.core.message import Message, MessageError, Payload
from onion_fl.core.registry import Registry

VERSION = 1


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _header(msg: Message) -> dict[str, Any]:
    return {
        "v": VERSION,
        "kind": msg.kind,
        "src": msg.src,
        "dst": msg.dst,
        "round": None if msg.round is None else int(msg.round),
        "weights": {k: _scalar(v) for k, v in msg.payload.weights.items()},
        "metrics": {k: _scalar(v) for k, v in msg.payload.metrics.items()},
        "meta": dict(msg.meta),
    }


def _dumps(doc: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise MessageError(
            f"message meta/metrics are not JSON-serialisable: {exc}"
        ) from exc


def _message(doc: Any, state: dict[str, np.ndarray]) -> Message:
    if not isinstance(doc, Mapping) or doc.get("v") != VERSION:
        raise MessageError(
            f"unsupported or missing encoding version (expected v={VERSION})"
        )
    try:
        return Message(
            kind=doc["kind"],
            src=doc["src"],
            dst=doc["dst"],
            round=doc.get("round"),
            payload=Payload(
                state=state,
                weights=doc.get("weights", {}),
                metrics=doc.get("metrics", {}),
            ),
            meta=doc.get("meta", {}),
        )
    except KeyError as exc:
        raise MessageError(f"encoded message is missing field {exc}") from exc


def _numeric_dtype(name: Any) -> np.dtype:
    try:
        dtype = np.dtype(name)
    except TypeError as exc:
        raise MessageError(f"unknown dtype {name!r}") from exc
    if dtype.kind not in "biuf":
        raise MessageError(f"non-numeric dtype {name!r}")
    return dtype


class Codec(ABC):
    """Turns a ``Message`` into bytes for a link and back."""

    name: str

    @abstractmethod
    def encode(self, msg: Message) -> bytes: ...

    @abstractmethod
    def decode(self, data: bytes) -> Message: ...

    def size(self, msg: Message) -> int:
        return len(self.encode(msg))


class JsonCodec(Codec):
    """Readable JSON; arrays keep dtype and shape, so round trips are bit-exact."""

    name = "json"

    def encode(self, msg: Message) -> bytes:
        doc = _header(msg)
        doc["state"] = {
            key: {
                "dtype": arr.dtype.str,
                "shape": list(arr.shape),
                "data": arr.ravel().tolist(),
            }
            for key, arr in msg.payload.state.items()
        }
        return _dumps(doc)

    def decode(self, data: bytes) -> Message:
        try:
            doc = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MessageError(f"not a JSON message: {exc}") from exc
        if not isinstance(doc, Mapping):
            raise MessageError("not a JSON message object")
        state: dict[str, np.ndarray] = {}
        for key, spec in dict(doc.get("state", {})).items():
            try:
                dtype = _numeric_dtype(spec["dtype"])
                state[key] = np.asarray(spec["data"], dtype=dtype).reshape(
                    spec["shape"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MessageError(f"state[{key!r}] is malformed: {exc}") from exc
        return _message(doc, state)


class NpzCodec(Codec):
    """Binary NumPy archive: a JSON header plus one ``.npy`` per array. Never unpickles."""

    name = "npz"

    def encode(self, msg: Message) -> bytes:
        doc = _header(msg)
        keys = list(msg.payload.state)
        doc["keys"] = keys
        arrays = {f"a{i}": msg.payload.state[key] for i, key in enumerate(keys)}
        buf = io.BytesIO()
        np.savez(buf, header=np.frombuffer(_dumps(doc), dtype=np.uint8), **arrays)
        return buf.getvalue()

    def decode(self, data: bytes) -> Message:
        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                doc = json.loads(archive["header"].tobytes())
                keys = doc.get("keys", []) if isinstance(doc, Mapping) else []
                state = {}
                for i, key in enumerate(keys):
                    array = archive[f"a{i}"]
                    _numeric_dtype(array.dtype)
                    state[key] = array
        except MessageError:
            raise
        except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile) as exc:
            raise MessageError(f"not an npz message: {exc}") from exc
        return _message(doc, state)


codecs = Registry("codec")
codecs.register(
    "json",
    title="JSON",
    description="Texto legible. Cada array conserva su dtype y su forma.",
    explain=(
        "Fácil de inspeccionar y depurar, pero ocupa más bytes por enlace: "
        "cada número viaja como texto."
    ),
)(JsonCodec)
codecs.register(
    "npz",
    title="NPZ (binario)",
    description="Archivo NumPy: una cabecera JSON y un .npy por array.",
    explain=(
        "Más compacto que JSON, así que reduce el tiempo de transmisión y los "
        "bytes por enlace. Nunca ejecuta pickle al leer."
    ),
)(NpzCodec)


def get_codec(name: str) -> Codec:
    """Build the codec registered as ``name`` (``json``, ``npz`` or a plugin path)."""
    return codecs.create(name)
