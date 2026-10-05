from __future__ import annotations

"""Wall-clock runtime over real transports (spec §9.2).

The same nodes and ``Context`` as ``SimRuntime``, but time is the wall clock
and messages travel over a transport (MQTT across processes, or the memory
bus). One loop thread runs every handler, so a node still handles one event
at a time; transport threads only enqueue deliveries.

A process hosts a subset of the nodes (``hosted``); the others are known so
links can be declared, but they run elsewhere. Every message carries
``meta.msg_id`` and ``meta.sent_at``: the receiver records the same id and the
end-to-end latency, which needs NTP-synchronised clocks across machines.
"""

import dataclasses
import heapq
import itertools
import json
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from onion_fl.core.codec import Codec, get_codec
from onion_fl.core.context import node_rng
from onion_fl.core.message import Message, MessageError
from onion_fl.core.node import Node
from onion_fl.roles.policies import create
from onion_fl.transports import transports

SEPARATOR = b"\x00"  # codec name, separator, encoded message


class RealRuntimeError(RuntimeError):
    """The real runtime is misconfigured."""


class _Context:
    def __init__(self, runtime: RealRuntime, node_id: str) -> None:
        self._runtime, self.node_id = runtime, node_id
        self.rng = runtime._rng(node_id)
        self.outbox: list[Message] = []

    def send(self, msg: Message) -> None:
        if msg.src != self.node_id:
            raise RealRuntimeError(
                f"{self.node_id} cannot send a message from {msg.src!r}"
            )
        if (msg.src, msg.dst) not in self._runtime._links:
            raise RealRuntimeError(f"no link {msg.src} -> {msg.dst}")
        self.outbox.append(msg)

    def set_timer(self, delay: float, name: str) -> None:
        self._runtime._arm(self.node_id, name, delay)

    def cancel_timer(self, name: str) -> None:
        self._runtime._timers.pop((self.node_id, name), None)

    def now(self) -> float:
        return self._runtime.now

    def emit(self, name: str, value: float | None = None, **tags: Any) -> None:
        self._runtime._record(self.node_id, name, value, tags)

    def compute(self, samples: float) -> None:
        """Real work takes real time: nothing to simulate."""


def _memory_rss() -> int | None:
    try:
        import psutil  # optional

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # noqa: BLE001 - psutil missing or not allowed
        return None


class RealRuntime:
    def __init__(
        self,
        run_id: str,
        seed: int = 0,
        hosted: Iterable[str] | None = None,
        listeners: list[Any] | None = None,
        heartbeat_s: float | None = 10.0,
        epoch: float | None = None,
    ) -> None:
        self.run_id, self.seed = run_id, seed
        self.hosted = None if hosted is None else set(hosted)
        self.listeners = list(listeners or [])
        self.heartbeat_s = heartbeat_s
        self.epoch = time.time() if epoch is None else epoch
        self.events: list[dict[str, Any]] = []
        self._nodes: dict[str, Node] = {}
        self._links: dict[tuple[str, str], tuple[Codec, Any]] = {}
        # The codecs this runtime's links use, by the name a message carries: a
        # received message names one of these, never a plugin to import.
        self._codecs: dict[str, Codec] = {}
        self._transports: dict[str, Any] = {}
        self._subscribed: set[tuple[int, str]] = set()
        self._timers: dict[tuple[str, str], int] = {}
        self._rngs: dict[str, Any] = {}
        self._queue: list[tuple[float, int, str, tuple]] = []
        self._cv = threading.Condition()
        self._seq = itertools.count(1)
        self._msg_ids = itertools.count(1)
        self._running = False
        self._thread: threading.Thread | None = None

    # --- setup -------------------------------------------------------------

    @property
    def now(self) -> float:
        return time.time() - self.epoch

    def inbox(self, node_id: str) -> str:
        return f"onionfl/{self.run_id}/{node_id}/inbox"

    def is_hosted(self, node_id: str) -> bool:
        return self.hosted is None or node_id in self.hosted

    def add_node(
        self, node: Node, *, compute: Any = None, availability: Any = None
    ) -> None:
        """Host ``node`` here, or just know it if it runs in another process."""
        if self.is_hosted(node.id):
            self._nodes[node.id] = node

    def _transport(self, spec: str | Mapping[str, Any]) -> Any:
        key = json.dumps(spec, sort_keys=True)
        if key not in self._transports:
            self._transports[key] = create(transports, spec)
        return self._transports[key]

    def add_link(
        self,
        child: str,
        parent: str,
        profile: Any = "lan",
        codec: str = "json",
        transport: str | Mapping[str, Any] = "mqtt",
    ) -> None:
        """Link two nodes both ways. ``profile`` is the simulator's; real links are real."""
        wire, carrier = get_codec(codec), self._transport(transport)
        self._codecs[wire.name] = wire
        for src, dst in ((child, parent), (parent, child)):
            self._links[(src, dst)] = (wire, carrier)
        for node_id in (child, parent):
            if (
                self.is_hosted(node_id)
                and (id(carrier), node_id) not in self._subscribed
            ):
                self._subscribed.add((id(carrier), node_id))
                carrier.on_receive(
                    self.inbox(node_id),
                    lambda data, n=node_id: self._post(0.0, "deliver", (n, data)),
                )

    # --- running -----------------------------------------------------------

    def _rng(self, node_id: str) -> Any:
        if node_id not in self._rngs:
            self._rngs[node_id] = node_rng(self.seed, node_id)
        return self._rngs[node_id]

    def _post(
        self, delay: float, kind: str, payload: tuple, seq: int | None = None
    ) -> None:
        with self._cv:
            heapq.heappush(
                self._queue, (self.now + delay, seq or next(self._seq), kind, payload)
            )
            self._cv.notify()

    def _arm(self, node_id: str, name: str, delay: float) -> None:
        token = next(self._seq)
        self._timers[(node_id, name)] = token
        self._post(delay, "timer", (node_id, name, token), seq=token)

    def start(self) -> None:
        """Start the transports and the loop in a background thread."""
        self._thread = threading.Thread(
            target=self.run, kwargs={"stop": lambda: False}, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _all_stopped(self) -> bool:
        return bool(self._nodes) and all(
            getattr(n, "stopped", False) for n in self._nodes.values()
        )

    def run(
        self, timeout: float | None = None, stop: Callable[[], bool] | None = None
    ) -> None:
        """Run until every hosted node has stopped (or ``stop()`` says so, or ``timeout``)."""
        stop = stop or self._all_stopped
        deadline = None if timeout is None else time.monotonic() + timeout
        for transport in self._transports.values():
            transport.start()
        self._running = True
        for node_id in self._nodes:
            self._post(0.0, "start", (node_id,))
            if self.heartbeat_s:
                self._post(self.heartbeat_s, "heartbeat", (node_id,))
        try:
            while self._running and not stop():
                if deadline is not None and time.monotonic() >= deadline:
                    self._record(
                        next(iter(self._nodes), "runtime"),
                        "runtime.timeout",
                        timeout,
                        {},
                    )
                    break
                with self._cv:
                    wait = (
                        0.1
                        if not self._queue
                        else max(0.0, self._queue[0][0] - self.now)
                    )
                    if wait > 0:
                        self._cv.wait(min(wait, 0.1))
                        continue
                    _, _, kind, payload = heapq.heappop(self._queue)
                getattr(self, f"_on_{kind}")(*payload)
        finally:
            self._running = False
            if self.heartbeat_s:  # a run shorter than the interval still reports
                for node_id in self._nodes:
                    self._on_heartbeat(node_id, final=True)
            for transport in self._transports.values():
                transport.stop()

    def _on_start(self, node_id: str) -> None:
        self._handle(node_id, "on_start", lambda node, ctx: node.on_start(ctx))

    def _on_deliver(self, node_id: str, data: bytes) -> None:
        name, _, body = data.partition(SEPARATOR)
        try:
            codec = self._codecs.get(name.decode())
            if codec is None:
                raise MessageError(f"no link here uses the codec {name[:40]!r}")
            msg = codec.decode(body)
        except Exception as exc:  # anything can arrive; the node goes on
            self._record(
                node_id, "message.rejected", None, {"reason": f"undecodable: {exc}"}
            )
            return
        sent_at = msg.meta.get("sent_at")
        tags = {
            "src": msg.src,
            "kind": msg.kind,
            "round": msg.round,
            "msg_id": msg.meta.get("msg_id"),
        }
        if isinstance(sent_at, int | float):
            tags["latency"] = max(0.0, time.time() - sent_at)
        self._record(node_id, "message.delivered", None, tags)
        self._handle(node_id, "on_message", lambda node, ctx: node.on_message(msg, ctx))

    def _on_timer(self, node_id: str, name: str, token: int) -> None:
        if self._timers.get((node_id, name)) != token:
            return
        del self._timers[(node_id, name)]
        self._handle(node_id, "on_timer", lambda node, ctx: node.on_timer(name, ctx))

    def _on_heartbeat(self, node_id: str, final: bool = False) -> None:
        node = self._nodes[node_id]
        tags = {
            "round": getattr(node, "round", None),
            "queue": len(self._queue),
            "cpu_s": time.process_time(),
            "final": final,
        }
        memory = _memory_rss()
        if memory is not None:
            tags["memory_bytes"] = memory
        self._record(node_id, "node.heartbeat", getattr(node, "round", None), tags)
        if self.heartbeat_s and not final:
            self._post(self.heartbeat_s, "heartbeat", (node_id,))

    def _handle(
        self, node_id: str, handler: str, call: Callable[[Node, _Context], None]
    ) -> None:
        ctx = _Context(self, node_id)
        try:
            call(self._nodes[node_id], ctx)
        except Exception as exc:  # a node never takes the process down
            self._record(
                node_id,
                "node.error",
                None,
                {"handler": handler, "error": f"{type(exc).__name__}: {exc}"},
            )
            return
        for msg in ctx.outbox:
            self._transmit(msg)

    def _transmit(self, msg: Message) -> None:
        codec, carrier = self._links[(msg.src, msg.dst)]
        msg_id = f"{msg.src}>{msg.dst}#{next(self._msg_ids)}@{self.run_id}"
        msg = dataclasses.replace(
            msg, meta={**msg.meta, "msg_id": msg_id, "sent_at": time.time()}
        )
        try:
            data = codec.name.encode() + SEPARATOR + codec.encode(msg)
        except MessageError as exc:
            self._record(
                msg.src,
                "node.error",
                None,
                {"handler": "send", "error": f"MessageError: {exc}"},
            )
            return
        tags = {
            "dst": msg.dst,
            "kind": msg.kind,
            "round": msg.round,
            "codec": codec.name,
            "msg_id": msg_id,
        }
        self._record(msg.src, "message.sent", len(data), tags)
        carrier.send(self.inbox(msg.dst), data)

    def _record(
        self, node_id: str, name: str, value: Any, tags: Mapping[str, Any]
    ) -> None:
        event = {
            "t": self.now,
            "node": node_id,
            "name": name,
            "value": value,
            "tags": dict(tags),
        }
        self.events.append(event)
        for listener in self.listeners:
            listener(event)
