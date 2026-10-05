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

from onion_fl.core.context import Context, child_rng
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.metrics import compute, predict, reduce_reports
from onion_fl.learning.model import group_of, is_aux, load_arrays, state_arrays
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


def _model_part(state: Mapping[str, Any]) -> State:
    """The model's own arrays, without auxiliary ones (control variates, …)."""
    return {k: v for k, v in state.items() if not is_aux(k)}


def _train_metrics(reports: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """``train_*`` of the children: examples and edges summed, the rest averaged by examples."""
    reports = [r for r in reports if r.get("train_examples")]
    examples = sum(r["train_examples"] for r in reports)
    if not examples:
        return {}
    out = {
        "train_examples": float(examples),
        "train_edges": float(sum(r.get("train_edges", 1.0) for r in reports)),
    }
    for name in sorted({k for r in reports for k in r if k.startswith("train_edges/")}):
        out[name] = float(sum(r.get(name, 0.0) for r in reports))  # per group
    names = sorted({k for r in reports for k in r if k.startswith("train_")} - set(out))
    for name in names:
        holders = [r for r in reports if name in r]
        weight = sum(r["train_examples"] for r in holders)
        out[name] = float(sum(r[name] * r["train_examples"] for r in holders) / weight)
    return out


def _eval_part(metrics: Mapping[str, float], model: str) -> tuple[dict, float]:
    """``eval.<model>.*`` of an update: the scores and their sample count."""
    prefix = f"eval.{model}."
    values = {k[len(prefix) :]: v for k, v in metrics.items() if k.startswith(prefix)}
    return values, values.pop("samples", 0)


def _send_idle(node: Any, round: int, scores: Mapping[str, float], ctx) -> None:
    """An update with nothing trained: the parent leaves it out of the quorum."""
    payload = Payload(metrics={"idle": 1.0, **scores})
    ctx.send(
        Message(
            kind="update", src=node.id, dst=node.parent, round=round, payload=payload
        )
    )


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
        close_at_quorum: bool = False,
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
        self.close_at_quorum = close_at_quorum  # FedBuff-style: close at K, not at all
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
        self.stale_age: dict[str, float] = {}  # rounds behind, plus what it carried
        self.stale_round: dict[str, int] = {}  # the round each buffered update is from
        self.sent_history: dict[int, State] = {}  # what was sent down, per round
        self.owed: dict[str, list[int]] = {}  # rounds each child has not answered
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

    def holders_below(self) -> dict[str, int]:
        """Training edges below that send each parameter group up."""
        holders: dict[str, int] = {}
        for meta in self.registered.values():
            for group, n in (meta.get("holders") or {}).items():
                holders[group] = holders.get(group, 0) + int(n)
        return dict(sorted(holders.items()))

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
        if self.open:  # the parent moved on before this round closed
            self._abandon(round, ctx)
        self.round, self.open, self.opened_at = round, True, ctx.now()
        self.responses, self.late, self.quorum_at = {}, 0, None
        selected = self.participation.select(self.trainers(), round, ctx.rng)
        if self.close_at_quorum:  # FedBuff-style: a busy child gets no newer model
            selected = [c for c in selected if c not in self.owed]
        self.participants = selected
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
            self.owed.setdefault(child, []).append(round)
        # Kept while an update trained on it may still be kept, to rebase it; a
        # policy that drops late updates needs none, and max_staleness bounds it.
        self.sent_history[round] = down
        alive = {round}
        if self.staleness.weight(1) is not None:
            alive |= {*self.stale_round.values()}
            alive |= {r for rounds in self.owed.values() for r in rounds}
            limit = getattr(self.staleness, "max_staleness", None)
            if limit is not None:
                alive = {r for r in alive if round - r <= limit}
        self.sent_history = {r: s for r, s in self.sent_history.items() if r in alive}
        if not self.participants:
            self._close(ctx)
        elif self.deadline is not None:
            ctx.set_timer(self.deadline, "deadline")

    def _abandon(self, new_round: int, ctx: Context) -> None:
        """Keep the updates of a round the parent overtook, as late ones."""
        ctx.cancel_timer("deadline")
        kept = 0
        for src, (contribution, metrics) in sorted(self.responses.items()):
            if contribution.state:
                kept += self._buffer(src, contribution, self.round, new_round, metrics)
        ctx.emit(
            "round.abandoned",
            float(kept),
            round=self.round,
            kept=kept,
            responded=len(self.responses),
        )
        self.open = False

    def _buffer(
        self,
        src: str,
        contribution: Contribution,
        round: int,
        target: int,
        metrics: Mapping[str, float],
    ) -> bool:
        """Hold a late update for round ``target``, weighted by its staleness."""
        factor = self.staleness.weight(target - round)
        if factor is None or not contribution.state:
            return False
        weights = {k: w * factor for k, w in contribution.weights.items()}
        self.stale[src] = Contribution(src, contribution.state, weights)
        carried = float((metrics or {}).get("staleness", 0.0))
        self.stale_age[src] = target - round + carried
        self.stale_round[src] = round
        return True

    def _reached(self) -> bool:
        """A buffering round (FedBuff-style) closes on K updates, late ones included."""
        if not self.close_at_quorum:
            return False
        fresh = {s for s, (c, _) in self.responses.items() if c.state}
        count = len(fresh) + sum(1 for s in self.stale if s not in fresh)
        return count >= max(quorum_needed(self.quorum, len(self.participants)), 1)

    def _rebased(self, source: str) -> Contribution | None:
        """A late update as its change from the model it trained on, applied to the
        model sent this round: an old model would pull the aggregate back. None
        when that model is gone; the update is then dropped, never used as is."""
        item = self.stale[source]
        base = self.sent_history.get(self.stale_round.get(source, -1))
        if base is None:
            return None
        state = dict(item.state)
        for key, value in item.state.items():
            if is_aux(key) or key not in base or key not in self.sent:
                continue
            delta = np.asarray(value, np.float64) - np.asarray(base[key], np.float64)
            moved = np.asarray(self.sent[key], np.float64) + delta
            state[key] = moved.astype(np.asarray(value).dtype)
        return Contribution(item.source, state, item.weights)

    def _update(self, msg: Message, ctx: Context) -> None:
        r = msg.round
        owed = [x for x in self.owed.pop(msg.src, []) if r is None or x > r]
        if owed:  # links are FIFO: an answer for r settles every round up to r
            self.owed[msg.src] = owed
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
            if self._reached() or len(self.responses) == len(self.participants):
                self._close(ctx)
            return
        if r is not None and (r < self.round or (r == self.round and not self.open)):
            target = self.round if self.open else self.round + 1
            buffered = self._buffer(
                msg.src, contribution, r, target, msg.payload.metrics
            )
            self.late += 1
            ctx.emit(
                "update.late",
                target - r,
                src=msg.src,
                round=r,
                action="buffered" if buffered else "drop",
            )
            if self.open and self._reached():
                self._close(ctx)
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
        # A child with nothing to train on (a stream's empty buffer) is idle.
        idle = {
            s for s, (c, m) in self.responses.items() if not c.state and "idle" in m
        }
        rebased = {s: self._rebased(s) for s in sorted(self.stale) if s not in fresh}
        late = [s for s, c in rebased.items() if c is not None]
        stale = [rebased[s] for s in late]
        lost = [s for s, c in rebased.items() if c is None]
        if lost:
            ctx.emit(
                "update.unrebased", float(len(lost)), sources=lost, round=self.round
            )
        # Staleness of what is combined: rounds behind here plus what each carried.
        age = {s: float(self.responses[s][1].get("staleness", 0.0)) for s in fresh}
        age |= {s: self.stale_age[s] for s in late}
        ages = [age[s] for s in sorted(age)]
        self.stale, self.stale_age, self.stale_round = {}, {}, {}
        needed = quorum_needed(self.quorum, len(self.participants) - len(idle))
        count = len(fresh) + (len(stale) if self.close_at_quorum else 0)
        tags = {
            "responded": len(fresh),
            "participants": len(self.participants),
            "round": self.round,
        }
        reports = {s: m for s, (_, m) in sorted(self.responses.items()) if s in fresh}
        scored = [m for s, (_, m) in sorted(self.responses.items()) if s in idle]
        if not count and not stale and idle and len(idle) == len(self.participants):
            ctx.emit("round.idle", self.round, idle=len(idle), round=self.round)
            scores = (
                self._children_scores(scored, ctx) if self.aggregate_children else {}
            )
            self._idle(scores, ctx)
            return
        if not count or count < needed:
            ctx.emit("round.quorum_failed", self.round, needed=needed, **tags)
            self._diagnose(fresh, {}, reports, ctx, failed=True)
            if self.aggregate_children:
                # Scores that arrived are evaluation, not aggregation: keep them.
                self._children_scores(list(reports.values()), ctx)
            self._failed(ctx)
            return
        given = [*fresh.values(), *stale]
        aggregated = self.aggregator.aggregate(
            given, source=self.id, reference=self.sent, rng=child_rng(ctx.rng)
        )
        self._report_aggregation({c.source for c in given}, ctx)
        self._diagnose(fresh, aggregated.state, reports, ctx, failed=False)
        self.previous = dict(aggregated.state)
        reports = list(reports.values())
        metrics = _train_metrics(reports)
        reports += scored  # idle children scored what arrived
        if any(ages):  # absent means 0, so synchronous runs send nothing new
            metrics["staleness"] = float(np.mean(ages))
        if self.aggregate_children:
            metrics |= self._children_scores(reports, ctx)
        ctx.emit("round.closed", ctx.now() - self.opened_at, stale=len(stale), **tags)
        self._closed(aggregated, metrics, ctx)

    def _report_aggregation(self, sources: set[str], ctx: Context) -> None:
        """Emit what the aggregator reports; a selection gets its malicious counts."""
        malicious = {
            child
            for child, meta in self.registered.items()
            if (meta.get("tags") or {}).get("malicious")
        }
        for name, value, tags in getattr(self.aggregator, "report", list)():
            tags = dict(tags)
            if "dropped" in tags:
                tags["malicious_dropped"] = len(set(tags["dropped"]) & malicious)
                tags["malicious"] = len(malicious & sources)
            if "excluded" in tags:
                tags["malicious_excluded"] = len(set(tags["excluded"]) & malicious)
            ctx.emit(name, value, round=self.round, **tags)

    def _diagnose(
        self, fresh, aggregated: State, reports, ctx: Context, failed: bool
    ) -> None:
        if not self.diagnostics:
            return
        view = RoundView(
            round=self.round,
            sent=_model_part(self.sent),
            contributions={
                child: Contribution(c.source, _model_part(c.state), c.weights)
                for child, c in fresh.items()
            },
            aggregated=_model_part(aggregated),
            previous=None if self.previous is None else _model_part(self.previous),
            received=_model_part(getattr(self, "received", {})),
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
        round_every: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(node_id, children, **kw)
        self.state: State = dict(state)
        self.rounds = rounds
        self.server_optimizer = server_optimizer
        self.finished = False
        # Virtual seconds between round starts (a stream's pace); None: at once.
        self.round_every, self.started_at = round_every, None

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "round":
            self._next(ctx)
        else:
            super().on_timer(name, ctx)

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
        if self.round_every is not None:
            if self.started_at is None:
                self.started_at = ctx.now()
            wait = self.started_at + self.round * self.round_every - ctx.now()
            if wait > 1e-9:
                ctx.set_timer(wait, "round")
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
        stats = dict(metrics) | {"edges_total": float(self.edges_below())}
        stats |= {f"edges_total/{g}": float(n) for g, n in self.holders_below().items()}
        self.state = self.server_optimizer.apply(
            self.state, _subset(aggregated.state, held), stats
        )
        if "train_loss" in metrics:
            ctx.emit(
                "round.train_loss",
                metrics["train_loss"],
                examples=metrics["train_examples"],
                round=self.round,
            )
        self._evaluate_and_go_on(ctx)

    def _evaluate_and_go_on(self, ctx: Context) -> None:
        last = self.round == self.rounds
        if self.eval_every and (_due(self.eval_every, self.round) or last):
            self._request_eval(self.round, self.state, "global", ctx)
        self._next(ctx)

    def _idle(self, scores: dict[str, float], ctx: Context) -> None:
        self._evaluate_and_go_on(ctx)  # nothing trained: the model stays

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
        meta = {"role": "aggregator", "edges": self.edges_below()}
        self._say_hello(meta | {"holders": self.holders_below()}, ctx)

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

    def _idle(self, scores: dict[str, float], ctx: Context) -> None:
        _send_idle(self, self.round, scores, ctx)


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
        attack: Any = None,
        privacy: Any = None,
        stream: Any = None,
        metrics: Sequence[str] = ("loss", "accuracy"),
    ) -> None:
        super().__init__(node_id)
        self.hello_retry = hello_retry
        # A stream (continuum C3): rows arrive over time, are predicted by the
        # model the edge serves when they come, and train once labelled.
        self.stream, self.metrics = stream, tuple(metrics)
        self._clock = -np.inf  # continuum time up to which arrivals are handled
        self._served: State | None = None  # the last model received, as served
        if stream is not None:
            shape = (len(stream.data.y), stream.data.n_classes)
            self._logits = np.full(shape, np.nan)
            self._predicted = np.zeros(len(stream.data.y), bool)
        self.parent, self.model, self.data = parent, model, data
        self.trainer, self.train = trainer, train
        self.finetuner = finetuner
        self.attack = attack
        self.privacy = privacy
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
        if self.train:
            meta["holders"] = dict.fromkeys(self._groups_up(self.model.state_dict()), 1)
        self._say_hello(meta, ctx)

    def _groups_up(self, keys: Iterable[str]) -> list[str]:
        """Parameter groups of ``keys`` that travel to the parent."""
        up = keys_crossing(keys, self.sharing, self.levels, self.parent_level)
        return sorted({group_of(k) for k in up if not is_aux(k)})

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
        self.finetuner.train(
            start,
            self.data,
            received=received,
            ctx=SimpleNamespace(rng=child_rng(ctx.rng)),
        )
        return start

    def _arrivals(self, round: int, ctx: Context) -> tuple[Any, dict[str, float]]:
        """What came since the last round, predicted by the model served when it
        came and scored; then the rows to train on now (None: nothing)."""
        s = self.stream
        lo, now = self._clock, s.clock(ctx.now())
        self._clock = now
        arrived = s.arrived(lo, now)
        with_label = arrived & (s.label_at <= s.available_at)
        late = s.labelled(lo, now) & (s.label_at > s.available_at)
        ctx.emit(
            "data.arrived",
            float(arrived.sum()),
            round=round,
            labelled=int(with_label.sum()),
            unlabelled=int((arrived & ~with_label).sum()),
            history=int((arrived & s.history).sum()),
            **self.tags,
        )
        ctx.emit("data.labelled", float(late.sum()), round=round, **self.tags)
        new = arrived & ~s.history
        if self._served is not None and new.any():
            served = copy.deepcopy(self.model)
            load_arrays(served, self._served)
            self._logits[new] = predict(served, s.data.X[new])
            self._predicted |= new
        metrics: dict[str, float] = {}
        # Every arrival against its truth (simulation only), and what a
        # deployment could score: the stored predictions whose label came.
        for model, rows in (("prequential", new), ("prequential_labelled", None)):
            rows = (s.labelled(lo, now) if rows is None else rows) & self._predicted
            if not rows.any():
                continue
            scores = compute(
                self._logits[rows], s.data.y[rows], s.data.n_classes, self.metrics
            )
            samples = int(rows.sum())
            _emit_scores(
                ctx,
                scores,
                model=model,
                source="edge",
                round=round,
                samples=samples,
                **self.tags,
            )
            metrics |= {f"eval.{model}.{k}": float(v) for k, v in scores.items()}
            metrics[f"eval.{model}.samples"] = float(samples)
        # What became trainable since this edge's last round: a late or skipped
        # round loses and repeats nothing. A window caps how old it may be.
        since = lo if s.window is None else max(lo, now - s.window)
        buffer = s.trainable(since, now)
        return (s.take(buffer) if buffer.any() else None), metrics

    def _train(self, msg: Message, ctx: Context) -> None:
        received = dict(msg.payload.state)
        own, metrics = self.data, {}
        if self.stream is not None:
            own, metrics = self._arrivals(msg.round, ctx)
        load_arrays(self.model, received)
        if self.stream is not None:
            self._served = state_arrays(self.model)
            if own is None:  # nothing to train on: idle, with what it scored
                _send_idle(self, msg.round, metrics, ctx)
                return
        scoring = (
            self.evaluate is not None
            and self.val_data is not None
            and _due(self.eval_every, msg.round)
        )
        finetuning = (
            scoring and "finetuned" in self.eval_models and self.finetuner is not None
        )
        start = copy.deepcopy(self.model) if finetuning else None
        attacking = self.attack is not None and msg.round >= self.attack.start_round
        data = self.attack.on_data(own) if attacking else own
        if scoring and "received" in self.eval_models:
            metrics |= self._score("received", msg.round, ctx)
        model_before = state_arrays(self.model)
        try:
            # What a diverged round rolls back to: the model, and the trainer's
            # memory if it can snapshot it (built-ins can; a plugin may opt in).
            saved = getattr(self.trainer, "snapshot", lambda: None)()
            result = self.trainer.train(self.model, data, received=received, ctx=ctx)
        except Exception as exc:  # the edge counts as absent; the run goes on
            ctx.emit(
                "edge.train_failed",
                round=msg.round,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        ctx.compute(result.samples)
        arrays = state_arrays(self.model) | dict(result.aux)
        broken = sorted(k for k, v in arrays.items() if not np.isfinite(v).all())
        if broken:  # a diverged edge rolls back and tells its parent at once
            if saved is not None:
                self.trainer.restore(saved)
            load_arrays(self.model, model_before)
            ctx.emit(
                "edge.train_failed",
                round=msg.round,
                error=f"non-finite weights after training: {broken[:3]}",
            )
            ctx.send(
                Message(kind="update", src=self.id, dst=self.parent, round=msg.round)
            )
            return
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
        up = keys_crossing(arrays, self.sharing, self.levels, self.parent_level)
        arrays = _subset(arrays, up)  # the hooks see only what is released
        if attacking:
            arrays = self.attack.on_update(arrays, received, child_rng(ctx.rng))
        if self.privacy is not None:
            arrays = self.privacy.on_update(arrays, received, child_rng(ctx.rng))
            ctx.emit(
                "diagnostic.privacy_epsilon",
                self.privacy.epsilon(),
                round=msg.round,
                mechanism="local",
            )
        ctx.emit("edge.trained", result.loss, round=msg.round, examples=result.examples)
        # One vote per edge for trainers whose papers average clients (SCAFFOLD).
        uniform = getattr(self.trainer, "uniform_weights", False)
        payload = Payload(
            state=arrays,
            weights=dict.fromkeys(up, 1.0 if uniform else float(result.examples)),
            metrics={
                "train_loss": float(result.loss),
                "train_examples": float(result.examples),
                "train_steps": float(result.batches),
                "train_edges": 1.0,
                **{f"train_edges/{g}": 1.0 for g in self._groups_up(up)},
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
        data = self.data
        if self.stream is not None:  # an evaluation stream: what came since last asked
            lo, self._clock = self._clock, self.stream.clock(ctx.now())
            data = self.stream.take(
                self.stream.arrived(lo, self._clock) & ~self.stream.history
            )
        if self.evaluate is not None and data is not None:
            load_arrays(self.model, dict(msg.payload.state))
            scores, samples = self.evaluate(self.model, data)
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
