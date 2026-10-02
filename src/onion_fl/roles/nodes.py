from __future__ import annotations

"""Coordinator, aggregator and edge state machines (spec §6).

Every role only talks through ``Context``; none of them knows which dataset
an edge holds. Round 1 carries the full initial state (``meta.bootstrap``) so
aggregators can seed the groups they keep without building a model; later
rounds send each link only the groups its sharing scope lets through.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from onion_fl.core.context import Context
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.model import load_arrays, state_arrays
from onion_fl.learning.sharing import SharingPolicy, keys_crossing, keys_held_at
from onion_fl.roles.policies import AllChildren, Drop, quorum_needed

State = dict[str, np.ndarray]


def _reject(ctx: Context, msg: Message, reason: str) -> None:
    ctx.emit(
        "message.rejected", src=msg.src, kind=msg.kind, round=msg.round, reason=reason
    )


def _subset(state: Mapping[str, np.ndarray], keys: Iterable[str]) -> State:
    return {k: state[k] for k in keys if k in state}


def _train_metrics(reports: Iterable[Mapping[str, float]]) -> dict[str, float]:
    reports = [r for r in reports if r.get("train_examples")]
    examples = sum(r["train_examples"] for r in reports)
    if not examples:
        return {}
    loss = (
        sum(r.get("train_loss", 0.0) * r["train_examples"] for r in reports) / examples
    )
    return {"train_loss": float(loss), "train_examples": float(examples)}


class _Collector(Node):
    """What the coordinator and the aggregators share: registration and round closing."""

    def __init__(
        self,
        node_id: str,
        children: Sequence[str],
        *,
        level: str,
        levels: Sequence[str],
        sharing: SharingPolicy,
        aggregator: Any,
        quorum: float = 1.0,
        deadline: float | None = None,
        participation: Any = None,
        staleness: Any = None,
        register_timeout: float | None = None,
    ) -> None:
        super().__init__(node_id)
        self.children = set(children)
        self.level, self.levels, self.sharing = level, list(levels), sharing
        self.aggregator = aggregator
        self.quorum, self.deadline = quorum, deadline
        self.participation = participation or AllChildren()
        self.staleness = staleness or Drop()
        self.register_timeout = register_timeout
        self.registered: dict[str, Mapping[str, Any]] = {}
        self.ready = False
        self.round, self.open, self.opened_at = 0, False, 0.0
        self.participants: list[str] = []
        self.responses: dict[str, tuple[Contribution, Mapping[str, float]]] = {}
        self.stale: dict[str, Contribution] = {}

    # --- registration ------------------------------------------------------------

    def on_start(self, ctx: Context) -> None:
        if self.register_timeout is not None:
            ctx.set_timer(self.register_timeout, "register")

    def _hello(self, msg: Message, ctx: Context) -> None:
        self.registered[msg.src] = dict(msg.meta)
        ctx.emit("node.registered", child=msg.src, role=msg.meta.get("role"))
        if not self.ready and self.children <= set(self.registered):
            self._finish_registration(ctx)

    def _finish_registration(self, ctx: Context) -> None:
        self.ready = True
        ctx.cancel_timer("register")
        self._registered(ctx)

    def _registered(self, ctx: Context) -> None:
        raise NotImplementedError

    def trainers(self) -> list[str]:
        """Registered children that train: edges, or aggregators with edges below."""
        return sorted(
            child
            for child, meta in self.registered.items()
            if meta.get("role") != "evaluator" and meta.get("edges", 1) > 0
        )

    def edges_below(self) -> int:
        return sum(
            int(meta.get("edges", 1))
            for meta in self.registered.values()
            if meta.get("role") != "evaluator"
        )

    # --- rounds ---------------------------------------------------------------------

    def on_message(self, msg: Message, ctx: Context) -> None:
        if msg.src in self.children and msg.kind == "hello":
            self._hello(msg, ctx)
        elif msg.src in self.children and msg.kind == "update":
            self._update(msg, ctx)
        else:
            self._other(msg, ctx)

    def _other(self, msg: Message, ctx: Context) -> None:
        _reject(ctx, msg, "unexpected sender or kind")

    def _open(self, round: int, state: State, ctx: Context, bootstrap: bool) -> None:
        self.round, self.open, self.opened_at = round, True, ctx.now()
        self.responses = {}
        self.participants = self.participation.select(self.trainers(), round, ctx.rng)
        ctx.emit(
            "round.participants", len(self.participants), children=self.participants
        )
        down = (
            state
            if bootstrap
            else _subset(
                state, keys_crossing(state, self.sharing, self.levels, self.level)
            )
        )
        payload = Payload(state=down)
        for child in self.participants:
            ctx.send(
                Message(
                    kind="global_model",
                    src=self.id,
                    dst=child,
                    round=round,
                    payload=payload,
                    meta={"bootstrap": bootstrap},
                )
            )
        if not self.participants:
            self._close(ctx)
        elif self.deadline is not None:
            ctx.set_timer(self.deadline, "deadline")

    def _update(self, msg: Message, ctx: Context) -> None:
        r = msg.round
        contribution = Contribution(
            msg.src, dict(msg.payload.state), dict(msg.payload.weights)
        )
        current = self.open and r == self.round
        if current and msg.src in self.participants and msg.src not in self.responses:
            self.responses[msg.src] = (contribution, dict(msg.payload.metrics))
            if len(self.responses) == len(self.participants):
                self._close(ctx)
            return
        if r is not None and (r < self.round or (r == self.round and not self.open)):
            target = self.round if self.open else self.round + 1
            factor = self.staleness.weight(target - r)
            action = "drop" if factor is None or not contribution.state else "buffered"
            ctx.emit("update.late", target - r, src=msg.src, round=r, action=action)
            if action == "buffered":
                weights = {k: w * factor for k, w in contribution.weights.items()}
                self.stale[msg.src] = Contribution(msg.src, contribution.state, weights)
            return
        _reject(ctx, msg, "update for a round that is not open")

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "register" and not self.ready:
            self._finish_registration(ctx)
        elif name == "deadline" and self.open:
            self._close(ctx)

    def _close(self, ctx: Context) -> None:
        ctx.cancel_timer("deadline")
        self.open = False
        fresh = {s: c for s, (c, _) in self.responses.items() if c.state}
        stale = [c for s, c in sorted(self.stale.items()) if s not in fresh]
        self.stale = {}
        needed = quorum_needed(self.quorum, len(self.participants))
        tags = {"responded": len(fresh), "participants": len(self.participants)}
        if not fresh or len(fresh) < needed:
            ctx.emit("round.quorum_failed", self.round, needed=needed, **tags)
            self._failed(ctx)
            return
        aggregated = self.aggregator.aggregate(
            [*fresh.values(), *stale], source=self.id
        )
        metrics = _train_metrics(
            m for s, (_, m) in self.responses.items() if s in fresh
        )
        ctx.emit("round.closed", ctx.now() - self.opened_at, stale=len(stale), **tags)
        self._closed(aggregated, metrics, ctx)

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        raise NotImplementedError

    def _failed(self, ctx: Context) -> None:
        raise NotImplementedError


class Coordinator(_Collector):
    """The root: owns the global model, runs the rounds and applies the server optimizer."""

    def __init__(
        self,
        node_id: str,
        children: Sequence[str],
        *,
        state: Mapping[str, np.ndarray],
        rounds: int,
        server_optimizer: Any,
        **kw: Any,
    ) -> None:
        super().__init__(node_id, children, **kw)
        self.state: State = dict(state)
        self.rounds = rounds
        self.server_optimizer = server_optimizer
        self.finished = False

    def _registered(self, ctx: Context) -> None:
        ctx.emit(
            "federation.registered",
            self.edges_below(),
            children=sorted(self.registered),
        )
        self._next(ctx)

    def _next(self, ctx: Context) -> None:
        if self.round >= self.rounds:
            self.finished = True
            ctx.emit("run.finished", self.round)
            return
        ctx.emit("round.started", self.round + 1)
        self._open(self.round + 1, self.state, ctx, bootstrap=self.round == 0)

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        held = keys_held_at(aggregated.state, self.sharing, self.levels, self.level)
        self.state = self.server_optimizer.apply(
            self.state, _subset(aggregated.state, held)
        )
        if metrics:
            ctx.emit(
                "round.train_loss",
                metrics["train_loss"],
                examples=metrics["train_examples"],
            )
        self._next(ctx)

    def _failed(self, ctx: Context) -> None:
        self._next(ctx)


class Aggregator(_Collector):
    """Any level below the root: forwards the model, aggregates its zone and reports up."""

    def __init__(
        self, node_id: str, children: Sequence[str], *, parent: str, **kw: Any
    ) -> None:
        super().__init__(node_id, children, **kw)
        self.parent = parent
        self.zone: State = {}
        self.parent_level = self.levels[self.levels.index(self.level) - 1]

    def _registered(self, ctx: Context) -> None:
        meta = {"role": "aggregator", "edges": self.edges_below()}
        ctx.send(Message(kind="hello", src=self.id, dst=self.parent, meta=meta))

    def _other(self, msg: Message, ctx: Context) -> None:
        if msg.src != self.parent or msg.kind != "global_model" or msg.round is None:
            _reject(ctx, msg, "unexpected sender or kind")
            return
        state = dict(msg.payload.state)
        bootstrap = bool(msg.meta.get("bootstrap"))
        if bootstrap:
            self.zone = _subset(
                state, keys_held_at(state, self.sharing, self.levels, self.level)
            )
        self._open(msg.round, {**state, **self.zone}, ctx, bootstrap)

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        held = keys_held_at(aggregated.state, self.sharing, self.levels, self.level)
        self.zone.update(_subset(aggregated.state, held))
        up = keys_crossing(
            aggregated.state, self.sharing, self.levels, self.parent_level
        )
        payload = Payload(
            state=_subset(aggregated.state, up),
            weights={k: aggregated.weights[k] for k in up},
            metrics=metrics,
        )
        ctx.send(
            Message(
                kind="update",
                src=self.id,
                dst=self.parent,
                round=self.round,
                payload=payload,
            )
        )

    def _failed(self, ctx: Context) -> None:
        ctx.send(Message(kind="update", src=self.id, dst=self.parent, round=self.round))


class Edge(Node):
    """Trains on its data when the model arrives; an evaluator (``train=False``) only evaluates."""

    def __init__(
        self,
        node_id: str,
        parent: str,
        *,
        model: Any,
        sharing: SharingPolicy,
        levels: Sequence[str],
        data: Any = None,
        trainer: Any = None,
        train: bool = True,
    ) -> None:
        super().__init__(node_id)
        self.parent, self.model, self.data = parent, model, data
        self.trainer, self.train = trainer, train
        self.sharing, self.levels = sharing, list(levels)
        self.parent_level = self.levels[-2]

    def on_start(self, ctx: Context) -> None:
        meta = {"role": "edge" if self.train else "evaluator", "edges": int(self.train)}
        ctx.send(Message(kind="hello", src=self.id, dst=self.parent, meta=meta))

    def on_message(self, msg: Message, ctx: Context) -> None:
        if msg.src != self.parent or msg.kind != "global_model" or msg.round is None:
            _reject(ctx, msg, "unexpected sender or kind")
            return
        if not self.train:
            return
        received = dict(msg.payload.state)
        load_arrays(self.model, received)
        try:
            result = self.trainer.train(
                self.model, self.data, received=received, ctx=ctx
            )
        except Exception as exc:  # the edge counts as absent; the run goes on
            ctx.emit(
                "edge.train_failed",
                round=msg.round,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        ctx.compute(result.samples)
        arrays = state_arrays(self.model)
        up = keys_crossing(arrays, self.sharing, self.levels, self.parent_level)
        ctx.emit("edge.trained", result.loss, round=msg.round, examples=result.examples)
        payload = Payload(
            state=_subset(arrays, up),
            weights=dict.fromkeys(up, float(result.examples)),
            metrics={
                "train_loss": float(result.loss),
                "train_examples": float(result.examples),
            },
        )
        ctx.send(
            Message(
                kind="update",
                src=self.id,
                dst=self.parent,
                round=msg.round,
                payload=payload,
            )
        )
