from __future__ import annotations

"""Virtual-clock, single-process runtime (spec §9.1).

Events are ordered by (virtual time, sequence), so a run is deterministic for
a given seed unless a node uses the ``measured`` compute model. A compute
model whose ``duration`` takes ``rng`` draws its times from the node's compute
stream (``node_rng(seed, "<node>/compute")``), apart from the node's own.

A node handles one event at a time. Work declared with ``ctx.compute`` keeps
it busy: what it sent in that handler leaves when the work ends
(``compute_done``), and what arrives meanwhile waits.

Nodes placed on a host with ``cpus`` share it: at most ``cpus`` of their jobs
run at once, and the others wait in FIFO order (``host.queued``). Nodes on no
host, or on one without ``cpus``, never wait for a CPU.
"""

import heapq
import inspect
import itertools
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
        self._compute_rng: dict[str, Any] = {}  # for models that draw their times
        self._availability: dict[str, Any] = {}
        self._busy_until: dict[str, float] = {}
        self._hosts: set[str] = set()
        self._host_of: dict[str, str] = {}
        self._slots: dict[str, list[float]] = {}  # host -> when each CPU is free
        self._links: dict[tuple[str, str], _Link] = {}
        self._timers: dict[tuple[str, str], int] = {}
        self._queue: list[tuple[float, int, str, tuple]] = []
        self._seq = 0
        self._processed = 0
        self._rngs: dict[str, Any] = {}
        self._msg_ids = itertools.count(1)
        self._started = False

    # --- setup -------------------------------------------------------------

    def add_node(
        self, node: Node, *, compute: Any = None, availability: Any = None
    ) -> None:
        if node.id in self._nodes:
            raise SimulationError(f"node {node.id!r} is already in the simulation")
        self._nodes[node.id] = node
        self.set_compute(node.id, compute)
        self._availability[node.id] = availability
        self._busy_until[node.id] = 0.0

    def add_host(self, host: str, cpus: int | None = None) -> None:
        """A machine; with ``cpus``, the jobs of its nodes contend for them."""
        if host in self._hosts:
            raise SimulationError(f"host {host!r} is already in the simulation")
        self._hosts.add(host)
        if cpus is not None:
            self._slots[host] = [0.0] * cpus

    def place(self, node_id: str, host: str) -> None:
        """Run ``node_id`` on ``host`` (one added with ``add_host``)."""
        if node_id not in self._nodes:
            raise SimulationError(f"unknown node {node_id!r}; add nodes first")
        if host not in self._hosts:
            raise SimulationError(f"unknown host {host!r}; add it with add_host")
        self._host_of[node_id] = host

    def set_compute(self, node_id: str, compute: Any) -> None:
        """Replace the compute model of a node already added."""
        if node_id not in self._nodes:
            raise SimulationError(f"unknown node {node_id!r}; add nodes first")
        self._compute[node_id] = compute
        takes_rng = (
            compute is not None
            and "rng" in inspect.signature(compute.duration).parameters
        )
        self._compute_rng[node_id] = (
            node_rng(self.seed, f"{node_id}/compute") if takes_rng else None
        )

    def add_link(
        self,
        child: str,
        parent: str,
        profile: str | Mapping[str, Any] | LinkProfile = "lan",
        codec: str = "json",
        transport: Any = None,
        channel: Any = None,
    ) -> None:
        """Link ``child`` and ``parent`` in both directions (up = child -> parent).

        ``transport`` is for the real runtime: here the link layer is the transport.
        ``channel(direction, rng)``, when given, builds each direction instead of
        a ``LinkChannel`` over ``profile`` (a path over a physical network).
        """
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
            built = (
                LinkChannel(resolved, direction, rng)
                if channel is None
                else channel(direction, rng)
            )
            self._links[(src, dst)] = _Link(built, wire)

    def compute_of(self, node_id: str) -> Any:
        """The compute model of a node (None: its work takes no time)."""
        return self._compute[node_id]

    def channel_of(self, src: str, dst: str) -> Any:
        """The channel of the direction ``src -> dst`` of a link."""
        return self._links[(src, dst)].channel

    def codec_of(self, src: str, dst: str) -> Codec:
        """The codec of the direction ``src -> dst`` of a link."""
        return self._links[(src, dst)].codec

    def note(
        self, t: float, node: str, name: str, value: Any = None, **tags: Any
    ) -> None:
        """Record an event at virtual time ``t`` (e.g. a link going down)."""
        self._push(t, "note", (node, name, value, tags))

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
        elif (back := self._back_at(node_id)) is not None:
            self._push(back, "start", (node_id,))  # it starts when it is back

    def _on_deliver(self, src: str, dst: str, data: bytes, msg_id: str) -> None:
        if self._defer_if_busy(dst, "deliver", (src, dst, data, msg_id)):
            return
        msg = self._links[(src, dst)].codec.decode(data)
        if not self._is_up(dst, msg.round):
            self._record(
                dst,
                "message.dropped_offline",
                None,
                {"src": src, "kind": msg.kind, "round": msg.round, "msg_id": msg_id},
            )
            return
        self._record(
            dst,
            "message.delivered",
            None,
            {"src": src, "kind": msg.kind, "round": msg.round, "msg_id": msg_id},
        )
        self._handle(dst, "on_message", lambda node, ctx: node.on_message(msg, ctx))

    def _on_timer(self, node_id: str, name: str, token: int) -> None:
        if self._timers.get((node_id, name)) != token:
            return  # re-armed or cancelled
        if self._defer_if_busy(node_id, "timer", (node_id, name, token)):
            return
        if not self._is_up(node_id, None):
            back = self._back_at(node_id)
            if back is not None:  # a local timer waits for the node, as when busy
                self._push(back, "timer", (node_id, name, token))
                return
            del self._timers[(node_id, name)]
            self._record(node_id, "timer.dropped_offline", None, {"timer": name})
            return
        del self._timers[(node_id, name)]
        self._handle(node_id, "on_timer", lambda node, ctx: node.on_timer(name, ctx))

    def _on_note(self, node: str, name: str, value: Any, tags: dict) -> None:
        self._record(node, name, value, tags)

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

    def _back_at(self, node_id: str) -> float | None:
        """When an offline node is up again; None if it never is (a crash)."""
        up_again = getattr(self._availability[node_id], "up_again", None)
        return None if up_again is None else up_again(self.now)

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
        model, rng = self._compute[node_id], self._compute_rng[node_id]
        drawn = {} if rng is None else {"rng": rng}
        busy = (
            0.0
            if model is None
            else model.duration(samples=ctx.work, wall_s=wall_s, **drawn)
        )
        if busy > 0:
            end = self._run_on_host(node_id, busy)
            self._busy_until[node_id] = end
            self._push(end, "compute_done", (node_id, ctx.outbox))
        else:
            for msg in ctx.outbox:
                self._transmit(msg)

    def _run_on_host(self, node_id: str, busy: float) -> float:
        """When a job of ``busy`` seconds ends: it starts now, or when a CPU of
        the node's host frees. Jobs ask in time order, so taking the CPU that
        frees first is FIFO."""
        host = self._host_of.get(node_id)
        slots = self._slots.get(host)
        if slots is None:
            return self.now + busy
        i = min(range(len(slots)), key=slots.__getitem__)
        start = max(self.now, slots[i])
        if start > self.now:
            self._record(node_id, "host.queued", start - self.now, {"host": host})
        slots[i] = start + busy
        return slots[i]

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
            # the same id is seen on delivery: it links send and receive (spans, latency)
            "msg_id": f"{msg.src}>{msg.dst}#{next(self._msg_ids)}",
        }
        self._record(msg.src, "message.sent", len(data), tags)
        arrival = link.channel.schedule(self.now, len(data))
        if arrival is None:
            self._record(msg.src, "link.dropped", len(data), tags)
            return
        self._push(arrival, "deliver", (msg.src, msg.dst, data, tags["msg_id"]))

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
