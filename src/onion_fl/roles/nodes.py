from __future__ import annotations

"""Coordinator, aggregator and edge state machines (spec §6, §10.3).

Every role only talks through ``Context``; none of them knows which dataset
an edge holds. Round 1 carries the full initial state (``meta.bootstrap``) so
aggregators can seed the groups they keep without building a model; later
rounds send each link only the groups its sharing scope lets through.

Evaluation happens at three levels:

- an edge scores the model it received and the one it trained on its
  ``local_val`` and sends the results up inside its update;
- every aggregator combines those reports by samples (``source=children``) and
  asks its evaluators (``val`` subjects) to score the zone model;
- the coordinator asks its evaluators (``test`` subjects) to score the global
  model, and reports per dataset tag.
"""

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np

from onion_fl.core.context import Context
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.metrics import reduce_reports
from onion_fl.learning.model import load_arrays, state_arrays
from onion_fl.learning.sharing import SharingPolicy, keys_crossing, keys_held_at
from onion_fl.observability.diagnostics import RoundView
from onion_fl.observability.diagnostics import diagnostics as diagnostic_plugins
from onion_fl.roles.policies import AllChildren, Drop, quorum_needed

State = dict[str, np.ndarray]
Evaluate = Callable[[Any, Any], tuple[Mapping[str, float], int]]


def _reject(ctx: Context, msg: Message, reason: str) -> None:
    ctx.emit(
        "message.rejected", src=msg.src, kind=msg.kind, round=msg.round, reason=reason
    )


def _subset(state: Mapping[str, np.ndarray], keys: Iterable[str]) -> State:
    return {k: state[k] for k in keys if k in state}


def _due(every: int | None, round: int) -> bool:
    return bool(every) and round % every == 0


def _train_metrics(reports: Iterable[Mapping[str, float]]) -> dict[str, float]:
    reports = [r for r in reports if r.get("train_examples")]
    examples = sum(r["train_examples"] for r in reports)
    if not examples:
        return {}
    loss = (
        sum(r.get("train_loss", 0.0) * r["train_examples"] for r in reports) / examples
    )
    return {"train_loss": float(loss), "train_examples": float(examples)}


def _eval_part(metrics: Mapping[str, float], model: str) -> tuple[dict, float]:
    """``eval.<model>.*`` of an update: the scores and their sample count."""
    prefix = f"eval.{model}."
    values = {k[len(prefix) :]: v for k, v in metrics.items() if k.startswith(prefix)}
    return values, values.pop("samples", 0)


def _emit_scores(ctx: Context, scores: Mapping[str, float], **tags: Any) -> None:
    for name, value in scores.items():
        ctx.emit(f"eval.{name}", value, **tags)


class _Collector(Node):
    """What the coordinator and the aggregators share: registration, rounds, evaluation."""

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
        eval_every: int | None = None,
        aggregate_children: bool = True,
        holdout: bool = True,
        diagnostics: Sequence[Any] | None = None,
    ) -> None:
        super().__init__(node_id)
        self.children = set(children)
        self.level, self.levels, self.sharing = level, list(levels), sharing
        self.aggregator = aggregator
        self.quorum, self.deadline = quorum, deadline
        self.participation = participation or AllChildren()
        self.staleness = staleness or Drop()
        self.register_timeout = register_timeout
        self.eval_every, self.aggregate_children = eval_every, aggregate_children
        self.holdout = holdout
        self.registered: dict[str, Mapping[str, Any]] = {}
        self.ready = False
        self.round, self.open, self.opened_at = 0, False, 0.0
        self.participants: list[str] = []
        self.responses: dict[str, tuple[Contribution, Mapping[str, float]]] = {}
        self.stale: dict[str, Contribution] = {}
        self.pending_eval: dict[int, dict[str, Any]] = {}
        self.diagnostics = (
            [diagnostic_plugins.create(n) for n in diagnostic_plugins.names()]
            if diagnostics is None
            else list(diagnostics)
        )
        self.sent: State = {}
        self.previous: State | None = None
        self.late, self.quorum_at = 0, None

    # --- registration ------------------------------------------------------------

    def on_start(self, ctx: Context) -> None:
        if self.register_timeout is not None:
            ctx.set_timer(self.register_timeout, "register")

    def _hello(self, msg: Message, ctx: Context) -> None:
        if msg.src not in self.registered:
            ctx.emit("node.registered", child=msg.src, role=msg.meta.get("role"))
        self.registered[msg.src] = dict(msg.meta)
        ack = Message(kind="control", src=self.id, dst=msg.src, meta={"ack": "hello"})
        ctx.send(ack)  # every hello, so a lost ack is answered by the next retry
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

    def evaluators(self) -> list[str]:
        return sorted(
            c for c, meta in self.registered.items() if meta.get("role") == "evaluator"
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
        elif msg.src in self.children and msg.kind == "eval_report":
            self._eval_report(msg, ctx)
        else:
            self._other(msg, ctx)

    def _other(self, msg: Message, ctx: Context) -> None:
        _reject(ctx, msg, "unexpected sender or kind")

    def _open(self, round: int, state: State, ctx: Context, bootstrap: bool) -> None:
        self.round, self.open, self.opened_at = round, True, ctx.now()
        self.responses, self.late, self.quorum_at = {}, 0, None
        self.participants = self.participation.select(self.trainers(), round, ctx.rng)
        ctx.emit(
            "round.participants",
            len(self.participants),
            children=self.participants,
            round=round,
        )
        down = (
            state
            if bootstrap
            else _subset(
                state, keys_crossing(state, self.sharing, self.levels, self.level)
            )
        )
        self.sent = down
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
            answered = sum(1 for c, _ in self.responses.values() if c.state)
            needed = quorum_needed(self.quorum, len(self.participants))
            if self.quorum_at is None and answered >= max(needed, 1):
                self.quorum_at = ctx.now() - self.opened_at
            if len(self.responses) == len(self.participants):
                self._close(ctx)
            return
        if r is not None and (r < self.round or (r == self.round and not self.open)):
            target = self.round if self.open else self.round + 1
            factor = self.staleness.weight(target - r)
            action = "drop" if factor is None or not contribution.state else "buffered"
            self.late += 1
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
        elif name.startswith("eval/"):
            round = int(name.split("/", 1)[1])
            if round in self.pending_eval:
                self._eval_done(round, ctx)

    def _close(self, ctx: Context) -> None:
        ctx.cancel_timer("deadline")
        self.open = False
        fresh = {s: c for s, (c, _) in self.responses.items() if c.state}
        stale = [c for s, c in sorted(self.stale.items()) if s not in fresh]
        self.stale = {}
        needed = quorum_needed(self.quorum, len(self.participants))
        tags = {
            "responded": len(fresh),
            "participants": len(self.participants),
            "round": self.round,
        }
        reports = {s: m for s, (_, m) in sorted(self.responses.items()) if s in fresh}
        if not fresh or len(fresh) < needed:
            ctx.emit("round.quorum_failed", self.round, needed=needed, **tags)
            self._diagnose(fresh, {}, reports, ctx, failed=True)
            if self.aggregate_children:
                # Scores that arrived are evaluation, not aggregation: keep them.
                self._children_scores(list(reports.values()), ctx)
            self._failed(ctx)
            return
        aggregated = self.aggregator.aggregate(
            [*fresh.values(), *stale], source=self.id
        )
        self._diagnose(fresh, aggregated.state, reports, ctx, failed=False)
        self.previous = dict(aggregated.state)
        reports = list(reports.values())
        metrics = _train_metrics(reports)
        if self.aggregate_children:
            metrics |= self._children_scores(reports, ctx)
        ctx.emit("round.closed", ctx.now() - self.opened_at, stale=len(stale), **tags)
        self._closed(aggregated, metrics, ctx)

    def _diagnose(
        self, fresh, aggregated: State, reports, ctx: Context, failed: bool
    ) -> None:
        if not self.diagnostics:
            return
        view = RoundView(
            round=self.round,
            sent=self.sent,
            contributions=fresh,
            aggregated=aggregated,
            previous=self.previous,
            received=getattr(self, "received", {}),
            datasets={
                c: meta["tags"]["dataset"]
                for c, meta in self.registered.items()
                if "dataset" in (meta.get("tags") or {})
            },
            reports=reports,
            participants=self.participants,
            late=self.late,
            quorum_failed=failed,
            time_to_quorum=self.quorum_at,
        )
        for plugin in self.diagnostics:
            for name, value, tags in plugin.compute(view):
                ctx.emit(f"diagnostic.{name}", value, round=self.round, **tags)

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        raise NotImplementedError

    def _failed(self, ctx: Context) -> None:
        raise NotImplementedError

    # --- evaluation -----------------------------------------------------------------

    def _children_scores(
        self, reports: Sequence[Mapping[str, float]], ctx: Context
    ) -> dict[str, float]:
        """Edge scores carried in the updates, combined by samples; they go up too."""
        up: dict[str, float] = {}
        models = sorted(
            {k.split(".")[1] for r in reports for k in r if k.startswith("eval.")}
        )
        for model in models:
            scores, samples = reduce_reports([_eval_part(r, model) for r in reports])
            if not samples:
                continue
            _emit_scores(
                ctx,
                scores,
                model=model,
                source="children",
                round=self.round,
                samples=samples,
            )
            up |= {f"eval.{model}.{k}": v for k, v in scores.items()}
            up[f"eval.{model}.samples"] = float(samples)
        return up

    def _request_eval(self, round: int, state: State, model: str, ctx: Context) -> bool:
        evaluators = self.evaluators()
        if not evaluators:
            return False
        self.pending_eval[round] = {
            "model": model,
            "waiting": set(evaluators),
            "reports": [],
        }
        payload = Payload(state=dict(state))
        for child in evaluators:
            ctx.send(
                Message(
                    kind="eval_request",
                    src=self.id,
                    dst=child,
                    round=round,
                    payload=payload,
                    meta={"model": model},
                )
            )
        if self.deadline is not None:
            ctx.set_timer(self.deadline, f"eval/{round}")
        return True

    def _eval_report(self, msg: Message, ctx: Context) -> None:
        entry = self.pending_eval.get(msg.round)
        if entry is None or msg.src not in entry["waiting"]:
            _reject(ctx, msg, "eval report nobody asked for")
            return
        metrics = dict(msg.payload.metrics)
        samples = metrics.pop("samples", 0)
        tags = dict(self.registered.get(msg.src, {}).get("tags") or {})
        entry["reports"].append((metrics, samples, tags.get("dataset")))
        entry["waiting"].discard(msg.src)
        if not entry["waiting"]:
            self._eval_done(msg.round, ctx)

    def _eval_done(self, round: int, ctx: Context) -> None:
        entry = self.pending_eval.pop(round)
        ctx.cancel_timer(f"eval/{round}")
        tags = {"model": entry["model"], "source": "evaluators", "round": round}
        reports = entry["reports"]
        scores, samples = reduce_reports([(m, n) for m, n, _ in reports])
        if samples:
            _emit_scores(ctx, scores, samples=samples, **tags)
        for dataset in sorted({d for _, _, d in reports if d is not None}):
            part, n = reduce_reports([(m, s) for m, s, d in reports if d == dataset])
            if n:
                _emit_scores(ctx, part, dataset=dataset, samples=n, **tags)
        self._eval_finished(round, ctx)

    def _eval_finished(self, round: int, ctx: Context) -> None:
        """Hook for the coordinator, which waits for the last evaluation to finish."""


def _stop_children(node: _Collector, ctx: Context) -> None:
    """The run is over: tell every registered child, so their processes can exit."""
    for child in sorted(node.registered):
        ctx.send(Message(kind="control", src=node.id, dst=child, meta={"stop": True}))


class _Greeter:
    """A child repeats its hello every ``hello_retry`` seconds until the parent acknowledges it.

    It also stops (``stopped``) when the parent says the run is over.
    """

    id: str
    parent: str
    hello_retry: float | None = 5.0
    stopped: bool = False

    def _stop_requested(self, msg: Message) -> bool:
        return (
            msg.src == self.parent
            and msg.kind == "control"
            and bool(msg.meta.get("stop"))
        )

    def _say_hello(self, meta: Mapping[str, Any], ctx: Context) -> None:
        self._hello_meta, self.acknowledged = dict(meta), False
        self._send_hello(ctx)

    def _send_hello(self, ctx: Context) -> None:
        ctx.send(
            Message(kind="hello", src=self.id, dst=self.parent, meta=self._hello_meta)
        )
        if self.hello_retry:
            ctx.set_timer(self.hello_retry, "hello")

    def _acknowledged(self, msg: Message, ctx: Context) -> bool:
        if (
            msg.src == self.parent
            and msg.kind == "control"
            and msg.meta.get("ack") == "hello"
        ):
            self.acknowledged = True
            ctx.cancel_timer("hello")
            return True
        return False


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
            if not self.pending_eval:
                self._finish(ctx)
            return
        ctx.emit("round.started", self.round + 1, round=self.round + 1)
        self._open(self.round + 1, self.state, ctx, bootstrap=self.round == 0)

    def _finish(self, ctx: Context) -> None:
        if not self.finished:
            self.finished = True
            ctx.emit("run.finished", self.round)
            _stop_children(self, ctx)

    @property
    def stopped(self) -> bool:
        return self.finished

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        held = keys_held_at(aggregated.state, self.sharing, self.levels, self.level)
        self.state = self.server_optimizer.apply(
            self.state, _subset(aggregated.state, held)
        )
        if "train_loss" in metrics:
            ctx.emit(
                "round.train_loss",
                metrics["train_loss"],
                examples=metrics["train_examples"],
                round=self.round,
            )
        last = self.round == self.rounds
        if self.eval_every and (_due(self.eval_every, self.round) or last):
            self._request_eval(self.round, self.state, "global", ctx)
        self._next(ctx)

    def _failed(self, ctx: Context) -> None:
        self._next(ctx)

    def _eval_finished(self, round: int, ctx: Context) -> None:
        last_round_closed = self.round >= self.rounds and not self.open
        if last_round_closed and not self.pending_eval:
            self._finish(ctx)


class Aggregator(_Greeter, _Collector):
    """Any level below the root: forwards the model, aggregates its zone and reports up."""

    def __init__(
        self,
        node_id: str,
        children: Sequence[str],
        *,
        parent: str,
        hello_retry: float | None = 5.0,
        **kw: Any,
    ) -> None:
        super().__init__(node_id, children, **kw)
        self.parent, self.hello_retry = parent, hello_retry
        self.zone: State = {}
        self.received: State = {}
        self.parent_level = self.levels[self.levels.index(self.level) - 1]

    def _registered(self, ctx: Context) -> None:
        self._say_hello({"role": "aggregator", "edges": self.edges_below()}, ctx)

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "hello":
            if not self.acknowledged:
                self._send_hello(ctx)
        else:
            super().on_timer(name, ctx)

    def _other(self, msg: Message, ctx: Context) -> None:
        if self._acknowledged(msg, ctx):
            return
        if self._stop_requested(msg):
            _stop_children(self, ctx)
            self.stopped = True
            ctx.cancel_timer("hello")
            return
        if msg.src != self.parent or msg.kind != "global_model" or msg.round is None:
            _reject(ctx, msg, "unexpected sender or kind")
            return
        self.received = dict(msg.payload.state)
        bootstrap = bool(msg.meta.get("bootstrap"))
        if bootstrap:
            held = keys_held_at(self.received, self.sharing, self.levels, self.level)
            self.zone = _subset(self.received, held)
        self._open(msg.round, {**self.received, **self.zone}, ctx, bootstrap)

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
        if self.holdout and _due(self.eval_every, self.round):
            zone_model = {**self.received, **self.zone, **aggregated.state}
            self._request_eval(self.round, zone_model, "zone", ctx)

    def _failed(self, ctx: Context) -> None:
        ctx.send(Message(kind="update", src=self.id, dst=self.parent, round=self.round))


class Edge(_Greeter, Node):
    """Trains when the model arrives; an evaluator (``train=False``) only answers eval requests."""

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
        val_data: Any = None,
        evaluate: Evaluate | None = None,
        eval_every: int | None = None,
        eval_models: Sequence[str] = ("received", "local"),
        tags: Mapping[str, Any] | None = None,
        hello_retry: float | None = 5.0,
        finetuner: Any = None,
    ) -> None:
        super().__init__(node_id)
        self.hello_retry = hello_retry
        self.parent, self.model, self.data = parent, model, data
        self.trainer, self.train = trainer, train
        self.finetuner = finetuner
        self.sharing, self.levels = sharing, list(levels)
        self.parent_level = self.levels[-2]
        self.val_data, self.evaluate = val_data, evaluate
        self.eval_every, self.eval_models = eval_every, tuple(eval_models)
        self.tags = dict(tags or {})

    def on_start(self, ctx: Context) -> None:
        meta = {
            "role": "edge" if self.train else "evaluator",
            "edges": int(self.train),
            "tags": self.tags,
        }
        self._say_hello(meta, ctx)

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "hello" and not self.acknowledged:
            self._send_hello(ctx)

    def on_message(self, msg: Message, ctx: Context) -> None:
        if self._acknowledged(msg, ctx):
            return
        if self._stop_requested(msg):
            self.stopped = True
            ctx.cancel_timer("hello")
            return
        if msg.src != self.parent or msg.round is None:
            _reject(ctx, msg, "unexpected sender")
        elif msg.kind == "eval_request" and not self.train:
            self._answer_eval(msg, ctx)
        elif msg.kind == "global_model" and self.train:
            self._train(msg, ctx)
        else:
            _reject(
                ctx,
                msg,
                f"a {'trainer' if self.train else 'evaluator'} does not take {msg.kind}",
            )

    def _score(
        self, model: str, round: int, ctx: Context, module: Any = None
    ) -> dict[str, float]:
        scores, samples = self.evaluate(
            self.model if module is None else module, self.val_data
        )
        _emit_scores(
            ctx,
            scores,
            model=model,
            source="edge",
            round=round,
            samples=samples,
            **self.tags,
        )
        return {f"eval.{model}.{k}": float(v) for k, v in scores.items()} | {
            f"eval.{model}.samples": float(samples)
        }

    def _finetuned(self, start: Any, received: State, ctx: Context) -> Any:
        """``start`` (the model as received) trained by the finetune trainer.

        It draws from a child stream of the node's generator, so scoring
        never shifts the draws of the edge's own training.
        """
        rng = np.random.default_rng(ctx.rng.bit_generator.seed_seq.spawn(1)[0])
        self.finetuner.train(
            start, self.data, received=received, ctx=SimpleNamespace(rng=rng)
        )
        return start

    def _train(self, msg: Message, ctx: Context) -> None:
        received = dict(msg.payload.state)
        load_arrays(self.model, received)
        scoring = (
            self.evaluate is not None
            and self.val_data is not None
            and _due(self.eval_every, msg.round)
        )
        finetuning = (
            scoring and "finetuned" in self.eval_models and self.finetuner is not None
        )
        start = copy.deepcopy(self.model) if finetuning else None
        metrics: dict[str, float] = {}
        if scoring and "received" in self.eval_models:
            metrics |= self._score("received", msg.round, ctx)
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
        if scoring and "local" in self.eval_models:
            metrics |= self._score("local", msg.round, ctx)
        try:
            if scoring and "personal" in self.eval_models:
                personal = getattr(self.trainer, "personal", lambda: None)()
                if personal is not None:
                    metrics |= self._score("personal", msg.round, ctx, personal)
            if finetuning:
                finetuned = self._finetuned(start, received, ctx)
                metrics |= self._score("finetuned", msg.round, ctx, finetuned)
        except Exception as exc:  # scoring never costs the edge its update
            ctx.emit(
                "edge.eval_failed",
                round=msg.round,
                error=f"{type(exc).__name__}: {exc}",
            )
        arrays = state_arrays(self.model)
        up = keys_crossing(arrays, self.sharing, self.levels, self.parent_level)
        ctx.emit("edge.trained", result.loss, round=msg.round, examples=result.examples)
        payload = Payload(
            state=_subset(arrays, up),
            weights=dict.fromkeys(up, float(result.examples)),
            metrics={
                "train_loss": float(result.loss),
                "train_examples": float(result.examples),
                **metrics,
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

    def _answer_eval(self, msg: Message, ctx: Context) -> None:
        metrics: dict[str, float] = {"samples": 0.0}
        if self.evaluate is not None and self.data is not None:
            load_arrays(self.model, dict(msg.payload.state))
            scores, samples = self.evaluate(self.model, self.data)
            metrics = {k: float(v) for k, v in scores.items()} | {
                "samples": float(samples)
            }
        ctx.send(
            Message(
                kind="eval_report",
                src=self.id,
                dst=self.parent,
                round=msg.round,
                payload=Payload(metrics=metrics),
                meta={"model": msg.meta.get("model")},
            )
        )
