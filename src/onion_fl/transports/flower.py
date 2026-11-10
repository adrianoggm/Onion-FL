from __future__ import annotations

"""Flower transport: every frame travels as a Flower ``Message`` (issue #204).

An optional adapter to Flower (``flwr`` >= 1.21, the ``flower`` extra) that
never controls the round loop: the coordinator, the aggregators and the edges
are the same state machines as on any other transport, and Flower only
carries their bytes.

What Flower gives is its message API. A frame the runtime built goes in a
``Message`` whose content is a ``RecordDict`` with one ``ConfigRecord``
(``onionfl``, the frame under ``frame``), and whose ``Metadata`` routes it:
``group_id`` is the inbox, ``dst_node_id`` a stable 63-bit id of it, and
``message_type`` is ``query``. On the wire is Flower's protobuf
(``flwr.proto.message_pb2.Message``), so anything that reads Flower messages
can read the traffic, and the bytes are the same from Flower 1.21 on.

What carries the messages is the ``grpc`` transport's machinery, under its
own service (``/onionfl.FlowerTransport/Deliver``) and discovery folder
(``onionfl-flower``): one gRPC server per process, direct links, one queue
and one thread per inbox, retries with backoff and the flush on ``stop``.

Why not Flower's SuperLink and SuperNodes. Their Fleet API is pull and
reply: a SuperNode only pushes replies to an instruction the ServerApp sent
(the SuperLink stores what a node pushes with ``is_reply_message=True``), so
an edge's ``hello`` or a fog's update could not leave unless the server had
asked for it, and the server would decide when each node talks: the round
loop. One SuperLink is also one level, a ServerApp over its SuperNodes, so a
fog would have to be both; and Flower starts the apps itself, per run. The
counters (``stats``) count frame bytes, as every transport does; the envelope
adds about 80 bytes per message.
"""

import hashlib
import time
from typing import Any

from onion_fl.core.registry import PluginError
from onion_fl.transports import transports
from onion_fl.transports.grpc import GrpcParams, GrpcTransport

RECORD, FRAME = "onionfl", "frame"
TTL = 12 * 3600.0  # Flower's default message TTL (s)


def _require_flower() -> Any:
    try:
        from flwr.app import ConfigRecord, Message, Metadata, RecordDict
        from flwr.common.serde import message_from_proto, message_to_proto
        from flwr.proto.message_pb2 import Message as ProtoMessage
    except ImportError:
        raise PluginError(
            "the flower transport needs flwr >= 1.21: pip install 'onion-fl[flower]'"
        ) from None
    return (
        ConfigRecord,
        Message,
        Metadata,
        RecordDict,
        message_from_proto,
        message_to_proto,
        ProtoMessage,
    )


def node_id(address: str) -> int:
    """A stable Flower node id for an inbox (63 bits, as Flower's are)."""
    digest = hashlib.blake2b(address.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


@transports.register(
    "flower",
    title="Flower",
    description=(
        "Cada trama viaja como un Message de Flower (>= 1.21) por enlaces gRPC "
        "directos; Flower no controla el bucle de rondas."
    ),
    params=GrpcParams,
)
class FlowerTransport(GrpcTransport):
    SERVICE = "onionfl.FlowerTransport"
    DISCOVERY = "onionfl-flower"

    def __init__(self, **params: Any) -> None:
        (
            self._ConfigRecord,
            self._Message,
            self._Metadata,
            self._RecordDict,
            self._from_proto,
            self._to_proto,
            self._Proto,
        ) = _require_flower()
        super().__init__(**params)

    def _wrap(self, address: str, data: bytes) -> bytes:
        metadata = self._Metadata(
            run_id=0,
            message_id="",
            src_node_id=0,
            dst_node_id=node_id(address),
            reply_to_message_id="",
            group_id=address,
            created_at=time.time(),
            ttl=TTL,
            message_type="query",
        )
        content = self._RecordDict({RECORD: self._ConfigRecord({FRAME: data})})
        message = self._Message(content, metadata=metadata)
        return self._to_proto(message).SerializeToString()

    def _unwrap(self, address: str, request: bytes) -> bytes:
        try:
            message = self._from_proto(self._Proto.FromString(request))
        except Exception as error:  # protobuf's DecodeError is not a ValueError
            raise ValueError(f"not a Flower message: {error}") from None
        if message.metadata.group_id != address:
            raise ValueError(
                f"a message for {message.metadata.group_id!r} reached {address!r}"
            )
        if not message.has_content() or RECORD not in message.content:
            raise ValueError("a Flower message without an Onion-FL frame")
        frame = message.content[RECORD].get(FRAME)
        if not isinstance(frame, bytes):
            raise ValueError("a Flower message without an Onion-FL frame")
        return frame
