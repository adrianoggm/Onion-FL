from __future__ import annotations

"""Virtual-clock, single-process runtime (spec §9.1).

Events are ordered by (virtual time, sequence), so a run is deterministic for
a given seed unless a node uses the ``measured`` compute model.

A node handles one event at a time. Work declared with ``ctx.compute`` keeps
it busy: what it sent in that handler leaves when the work ends
(``compute_done``), and what arrives meanwhile waits.
"""

import heapq
import time
from collections.abc import Mapping
from typing import Any

from onion_fl.core.codec import Codec, get_codec
from onion_fl.core.context import node_rng
from onion_fl.core.message import Message, MessageError
from onion_fl.core.node import Node
from onion_fl.runtime.network import LinkChannel, LinkProfile, resolve_profile


class SimulationError(RuntimeError):
    """The simulation itself is misconfigured or runs away."""


class _Link:
    def __init__(self, channel: LinkChannel, codec: Codec) -> None:
        self.channel = channel
        self.codec = codec


class _SimContext:
    """The ``Context`` a node gets inside the simulator."""

    def __init__(self, runtime: SimRuntime, node_id: str) -> None:
        self._runtime = runtime
        self.node_id = node_id
        self.rng = runtime._rng(node_id)
        self.outbox: list[Message] = []
        self.work = 0.0

    def send(self, msg: Message) -> None:
        if msg.src != self.node_id:
            raise SimulationError(
                f"{self.node_id} cannot send a message from {msg.src!r}"
            )
        if (msg.src, msg.dst) not in self._runtime._links:
            raise SimulationError(f"no link {msg.src} -> {msg.dst}")
        self.outbox.append(msg)

    def set_timer(self, delay: float, name: str) -> None:
        if delay < 0:
            raise SimulationError(f"timer {name!r}: negative delay {delay}")
        self._runtime._arm_timer(self.node_id, name, self._runtime.now + delay)

    def cancel_timer(self, name: str) -> None:
        self._runtime._timers.pop((self.node_id, name), None)

    def now(self) -> float:
        return self._runtime.now

    def emit(self, name: str, value: float | None = None, **tags: Any) -> None:
        self._runtime._record(self.node_id, name, value, tags)

    def compute(self, samples: float) -> None:
        if samples < 0:
            raise SimulationError(f"compute: negative work {samples}")
        self.work += samples


class SimRuntime:
    """Hosts nodes and links on a virtual clock."""

    def __init__(
        self,
        seed: int = 0,
        max_events: int = 1_000_000,
        listeners: list[Any] | None = None,
    ) -> None:
        self.seed = seed
        self.listeners = list(listeners or [])  # called with every recorded event
        self.max_events = max_events
        self.now = 0.0
        self.events: list[dict[str, Any]] = []
        self._nodes: dict[str, Node] = {}
        self._compute: dict[str, Any] = {}
        self._availability: dict[str, Any] = {}
        self._busy_until: dict[str, float] = {}
        self._links: dict[tuple[str, str], _Link] = {}
        self._timers: dict[tuple[str, str], int] = {}
        self._queue: list[tuple[float, int, str, tuple]] = []
        self._seq = 0
        self._processed = 0
        self._rngs: dict[str, Any] = {}
        self._started = False

    # --- setup -------------------------------------------------------------

    def add_node(
        self, node: Node, *, compute: Any = None, availability: Any = None
    ) -> None:
        if node.id in self._nodes:
            raise SimulationError(f"node {node.id!r} is already in the simulation")
        self._nodes[node.id] = node
        self._compute[node.id] = compute
        self._availability[node.id] = availability
        self._busy_until[node.id] = 0.0

    def add_link(
        self,
        child: str,
        parent: str,
        profile: str | Mapping[str, Any] | LinkProfile = "lan",
        codec: str = "json",
    ) -> None:
        """Link ``child`` and ``parent`` in both directions (up = child -> parent)."""
        for node_id in (child, parent):
            if node_id not in self._nodes:
                raise SimulationError(
                    f"unknown node {node_id!r}; add nodes before links"
                )
        if (child, parent) in self._links or (parent, child) in self._links:
            raise SimulationError(f"link {child} <-> {parent} already exists")
        resolved = resolve_profile(profile)
        wire = get_codec(codec)
        for src, dst, direction in ((child, parent, "up"), (parent, child, "down")):
            rng = node_rng(self.seed, f"link/{src}->{dst}")
            self._links[(src, dst)] = _Link(LinkChannel(resolved, direction, rng), wire)

    # --- running -----------------------------------------------------------

    def run(self, until: float | None = None) -> None:
        """Process events up to ``until`` (or until none are left). Can be called again."""
        if not self._started:
            self._started = True
            for node_id in self._nodes:
                self._push(0.0, "start", (node_id,))
        while self._queue:
            if until is not None and self._queue[0][0] > until:
                break
            t, _, kind, payload = heapq.heappop(self._queue)
            self._processed += 1
            if self._processed > self.max_events:
                raise SimulationError(
                    f"max_events={self.max_events} exceeded at t={t:.6f}"
                )
            self.now = t
            getattr(self, f"_on_{kind}")(*payload)
        if until is not None:
            self.now = max(self.now, until)

    def _on_start(self, node_id: str) -> None:
        if self._is_up(node_id, None):
            self._handle(node_id, "on_start", lambda node, ctx: node.on_start(ctx))

    def _on_deliver(self, src: str, dst: str, data: bytes) -> None:
        if self._defer_if_busy(dst, "deliver", (src, dst, data)):
            return
        msg = self._links[(src, dst)].codec.decode(data)
        if not self._is_up(dst, msg.round):
            self._record(
                dst,
                "message.dropped_offline",
                None,
                {"src": src, "kind": msg.kind, "round": msg.round},
            )
            return
        self._record(
            dst,
            "message.delivered",
            None,
            {"src": src, "kind": msg.kind, "round": msg.round},
        )
        self._handle(dst, "on_message", lambda node, ctx: node.on_message(msg, ctx))

    def _on_timer(self, node_id: str, name: str, token: int) -> None:
        if self._timers.get((node_id, name)) != token:
            return  # re-armed or cancelled
        if self._defer_if_busy(node_id, "timer", (node_id, name, token)):
            return
        del self._timers[(node_id, name)]
        if not self._is_up(node_id, None):
            self._record(node_id, "timer.dropped_offline", None, {"timer": name})
            return
        self._handle(node_id, "on_timer", lambda node, ctx: node.on_timer(name, ctx))

    def _on_compute_done(self, node_id: str, outbox: list[Message]) -> None:
        if not self._is_up(node_id, None):
            self._record(
                node_id, "message.dropped_offline", None, {"pending": len(outbox)}
            )
            return
        for msg in outbox:
            self._transmit(msg)

    # --- internals ---------------------------------------------------------

    def _rng(self, node_id: str) -> Any:
        """One stream per node for the whole run, not one per handler."""
        if node_id not in self._rngs:
            self._rngs[node_id] = node_rng(self.seed, node_id)
        return self._rngs[node_id]

    def _push(self, t: float, kind: str, payload: tuple) -> None:
        self._seq += 1
        heapq.heappush(self._queue, (t, self._seq, kind, payload))

    def _arm_timer(self, node_id: str, name: str, at: float) -> None:
        self._seq += 1
        token = self._seq
        self._timers[(node_id, name)] = token
        heapq.heappush(self._queue, (at, token, "timer", (node_id, name, token)))

    def _defer_if_busy(self, node_id: str, kind: str, payload: tuple) -> bool:
        busy_until = self._busy_until[node_id]
        if self.now < busy_until:
            self._push(busy_until, kind, payload)
            return True
        return False

    def _is_up(self, node_id: str, round: int | None) -> bool:
        model = self._availability[node_id]
        return model is None or model.is_up(
            self.now, round=round, node_id=node_id, seed=self.seed
        )

    def _handle(self, node_id: str, handler: str, call: Any) -> None:
        ctx = _SimContext(self, node_id)
        started = time.perf_counter()
        try:
            call(self._nodes[node_id], ctx)
        except Exception as exc:  # a node never takes the simulation down
            self._record(
                node_id,
                "node.error",
                None,
                {"handler": handler, "error": f"{type(exc).__name__}: {exc}"},
            )
            return
        wall_s = time.perf_counter() - started
        model = self._compute[node_id]
        busy = 0.0 if model is None else model.duration(samples=ctx.work, wall_s=wall_s)
        if busy > 0:
            self._busy_until[node_id] = self.now + busy
            self._push(self.now + busy, "compute_done", (node_id, ctx.outbox))
        else:
            for msg in ctx.outbox:
                self._transmit(msg)

    def _transmit(self, msg: Message) -> None:
        link = self._links[(msg.src, msg.dst)]
        try:
            data = link.codec.encode(msg)
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
            "codec": link.codec.name,
        }
        self._record(msg.src, "message.sent", len(data), tags)
        arrival = link.channel.schedule(self.now, len(data))
        if arrival is None:
            self._record(msg.src, "link.dropped", len(data), tags)
            return
        self._push(arrival, "deliver", (msg.src, msg.dst, data))

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
