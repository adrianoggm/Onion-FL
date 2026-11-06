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

from onion_fl.continuum.drift import Reference, detectors, window
from onion_fl.continuum.memory import Memory
from onion_fl.continuum.pace import View
from onion_fl.continuum.unlabelled import Objective
from onion_fl.continuum.versions import MODELS, PARTS, Version
from onion_fl.core.context import Context, child_rng
from onion_fl.core.message import Message, Payload
from onion_fl.core.node import Node
from onion_fl.learning.aggregators import Contribution
from onion_fl.learning.metrics import compute, predict, reduce_reports
from onion_fl.learning.model import group_of, is_aux, load_arrays, state_arrays
from onion_fl.learning.sharing import SharingPolicy, keys_crossing, keys_held_at
from onion_fl.learning.trainers import frozen_keys
from onion_fl.observability.diagnostics import RoundView
from onion_fl.observability.diagnostics import diagnostics as diagnostic_plugins
from onion_fl.roles.policies import AllChildren, Drop, create, quorum_needed

State = dict[str, np.ndarray]
Evaluate = Callable[[Any, Any], tuple[Mapping[str, float], int]]


UPDATES = ("warm", "frozen", "scratch")


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


def _counted(key: str) -> bool:
    """Status keys that count toward a trigger since the last round."""
    return key == "fresh" or key.endswith(".detected")


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
        continuum: Any = None,
    ) -> None:
        super().__init__(node_id)
        # A continuous federation (continuum C6): the children's statuses heard
        # since the last tick, and this level's drift detectors.
        self.continuum = continuum
        self._heard: dict[str, float] = {}
        drift = continuum.drift if continuum is not None else {}
        self._detectors = {k: create(detectors, d) for k, d in sorted(drift.items())}
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
        # Children that already got a whole model; the first one each gets is.
        self.bootstrapped: set[str] = set()
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

    stopped: bool = False
    _ticks_from_start = True  # the root ticks from its first round instead

    def on_start(self, ctx: Context) -> None:
        if self.register_timeout is not None:
            ctx.set_timer(self.register_timeout, "register")
        if self.continuum is not None and self._ticks_from_start:
            ctx.set_timer(self.continuum.status_every, "status")

    def _hello(self, msg: Message, ctx: Context) -> None:
        if msg.src not in self.registered:
            ctx.emit("node.registered", child=msg.src, role=msg.meta.get("role"))
        self.registered[msg.src] = dict(msg.meta)
        ack = Message(kind="control", src=self.id, dst=msg.src, meta={"ack": "hello"})
        ctx.send(ack)  # every hello, so a lost ack is answered by the next retry
        if not self.ready and self.children <= set(self.registered):
            self._finish_registration(ctx)
        elif self.ready:
            self._late_child(ctx)

    def _late_child(self, ctx: Context) -> None:
        """A child said hello after registration ended (``register_timeout``)."""

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
        elif msg.src in self.children and msg.kind == "status":
            self._heard_from(msg.payload.metrics)
        else:
            self._other(msg, ctx)

    def _other(self, msg: Message, ctx: Context) -> None:
        _reject(ctx, msg, "unexpected sender or kind")

    def _open(
        self,
        round: int,
        state: State,
        ctx: Context,
        bootstrap: bool,
        whole: State | None = None,
    ) -> None:
        if self.open:  # the parent moved on before this round closed
            self._abandon(round, ctx)
        # What was heard for the round before is not for this one (continuum C6).
        self._heard = {k: v for k, v in self._heard.items() if not _counted(k)}
        self.round, self.open, self.opened_at = round, True, ctx.now()
        self.responses, self.late, self.quorum_at = {}, 0, None
        # A round a child owes stops counting once its update would be dropped: a
        # lost message must not keep the child waiting for good.
        self.owed = {
            child: kept
            for child, rounds in self.owed.items()
            if (
                kept := [
                    r for r in rounds if self.staleness.weight(round - r) is not None
                ]
            )
        }
        trainers = self.trainers()
        if self.close_at_quorum:  # FedBuff-style: a busy child gets no newer model
            trainers = [c for c in trainers if c not in self.owed]
        self.participants = self.participation.select(trainers, round, ctx.rng)
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
        first = Payload(state=state if whole is None else whole)
        for child in self.participants:
            new = child not in self.bootstrapped  # a child drawn late, or lost
            ctx.send(
                Message(
                    kind="global_model",
                    src=self.id,
                    dst=child,
                    round=round,
                    payload=first if new else payload,
                    meta={"bootstrap": new},
                )
            )
            self.bootstrapped.add(child)
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
        # Every child busy: the round waits for their late updates, not closing empty.
        busy = self.close_at_quorum and bool(self.owed)
        if not self.participants and not busy:
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
        elif name == "status":
            self._tick(ctx)

    def _heard_from(self, metrics: Mapping[str, float]) -> None:
        """Pool a child's status. Its rows and detections count only if the child
        sent it after this node's current round reached it: before, they are
        what that round takes, or what opened it."""
        stale = metrics.get("round", 0.0) < self.round
        for key, value in metrics.items():
            if key in ("at", "round") or (stale and _counted(key)):
                continue
            self._heard[key] = self._heard.get(key, 0.0) + float(value)

    def _tick(self, ctx: Context) -> None:
        """Pool what the children reported since the last tick, run this level's
        detectors on it (zone drift at a fog, federation drift at the root) and
        pass it on; until the run is over (spec §10)."""
        if self.stopped:
            return
        now = ctx.now() * self.continuum.speed
        heard, self._heard = self._heard, {}
        for kind, detector in self._detectors.items():
            n = heard.get(f"drift.{kind}.n", 0.0)
            if n <= 0:
                continue
            value = heard[f"drift.{kind}.sum"] / n
            found = detector.update(value)
            if found is not None:
                key = f"drift.{kind}.detected"
                heard[key] = heard.get(key, 0.0) + 1.0
                ctx.emit("drift.detected", found, kind=kind, window=value, at=now)
        self._heard_all(heard, now, ctx)
        # Every node ticks until the last label, the tick after it included: a
        # stop lost on its way down must not keep the run alive.
        if ctx.now() < self.continuum.until:
            ctx.set_timer(self.continuum.status_every, "status")

    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        """What the children said since the last tick, pooled at this level."""
        raise NotImplementedError

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
                self._children_scores([*reports.values(), *scored], ctx)
            self._failed(ctx)
            return
        given = [*fresh.values(), *stale]
        aggregated = self.aggregator.aggregate(
            given, source=self.id, reference=self.sent, rng=child_rng(ctx.rng)
        )
        # Combining costs host time by the bytes of what is combined (I1 §7);
        # without an aggregate rate the node has no compute model: it is free.
        ctx.compute(
            float(sum(np.asarray(v).nbytes for c in given for v in c.state.values()))
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


def _server_state(node: Any) -> State:
    """A copy of the server optimizer's state, to go with a version."""
    state = getattr(node.server_optimizer, "state", dict)()
    return {k: np.array(v, copy=True) for k, v in state.items()}


def _load_server_state(node: Any, state: Mapping[str, np.ndarray]) -> None:
    load = getattr(node.server_optimizer, "load_state", None)
    if load is not None:
        load({k: np.array(v, copy=True) for k, v in state.items()})


def _part_scores(reports: Sequence[Mapping[str, float]], prefix: str) -> dict:
    """The validators' scores under ``prefix``, combined by samples."""
    parts = []
    for report in reports:
        values = {
            k[len(prefix) :]: v for k, v in report.items() if k.startswith(prefix)
        }
        parts.append((values, values.pop("samples", 0.0)))
    scores, samples = reduce_reports(parts)
    return scores | {"samples": float(samples)}


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
        promotion: Any = None,
        validators: Iterable[str] = (),
        operators: Iterable[str] = ("operator",),
        **kw: Any,
    ) -> None:
        super().__init__(node_id, children, **kw)
        # Who validates and who may give orders come from the federation's
        # build, never from what a child says of itself in its hello.
        self._validators = frozenset(validators)
        self.operators = frozenset(operators)
        stray = sorted(self._validators - self.children)
        if stray:
            raise ValueError(f"validators {stray} are not children of {node_id}")
        if self.operators & self.children:
            raise ValueError(
                f"operators {sorted(self.operators & self.children)} are children "
                f"of {node_id}; an operator gives orders from outside the tree"
            )
        self.state: State = dict(state)
        self.rounds = rounds
        self.server_optimizer = server_optimizer
        self.finished = False
        # Champion and challenger (continuum C7): ``state`` is always the
        # champion, the model the edges serve; a round's aggregate is a
        # challenger until ``promotion`` decides on it.
        self.promotion = promotion
        self.versions: dict[int, Version] = {}
        self.champion = 0
        self._deciding: Version | None = None
        self._validation: dict[str, Any] = {}
        self._orders: list[Message] = []  # operator orders that wait for a decision
        self._promoted_at = 0.0  # virtual time the champion was promoted
        # Virtual seconds between round starts (a stream's pace); None: at once.
        self.round_every, self.started_at = round_every, None
        # A continuous federation (continuum C6): what the statuses said since the
        # last round opened, and when it opened (data time).
        self._volume, self._drift, self._opened_at = 0.0, {}, 0.0
        self._horizon = False  # the final round opened: the streams are over

    _ticks_from_start = False  # from round 1, so a schedule keeps its grid

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "round":
            self._next(ctx)
        elif name == "validate":
            if self._deciding is not None:
                self._decide(ctx)  # with the reports that came in time
        else:
            super().on_timer(name, ctx)

    def _registered(self, ctx: Context) -> None:
        ctx.emit(
            "federation.registered",
            self.edges_below(),
            children=sorted(self.registered),
        )
        if self.promotion is not None:  # the model it starts from is the first
            self.versions[0] = Version(
                0, None, 0, dict(self.state), _server_state(self), status="active"
            )
        self._next(ctx)

    def _next(self, ctx: Context) -> None:
        if self.round >= self.rounds:
            if not self.pending_eval:
                self._finish(ctx)
            return
        if self.continuum is not None:  # rounds wait for a trigger (spec §10)
            if self.round == 0:
                self._start("start", ctx.now() * self.continuum.speed, ctx)
                ctx.set_timer(self.continuum.status_every, "status")
            else:
                self._maybe_open(ctx)
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

    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        self._volume += heard.get("fresh", 0.0)
        for key, value in heard.items():
            if key.startswith("drift.") and key.endswith(".detected") and value:
                kind = key.split(".")[1]
                self._drift[kind] = self._drift.get(kind, 0) + int(value)
        self._maybe_open(ctx)

    def _maybe_open(self, ctx: Context) -> None:
        """Open the next round if the trigger asks. At the first tick after the
        last label, one final round, after which the run finishes."""
        if self.open or self.finished or self.round == 0 or self.round >= self.rounds:
            return
        if self._deciding is not None:  # the next round serves what is decided
            return
        now = ctx.now() * self.continuum.speed
        if ctx.now() >= self.continuum.until - 1e-9:
            name, self.rounds = "horizon", self.round + 1  # the last one
            self._horizon = True
        else:
            view = View(
                now=now,
                since=self._opened_at,
                volume=self._volume,
                drift=dict(self._drift),
            )
            name = self.continuum.trigger.fired(view)
            if name is None:
                return
        self._start(name, now, ctx)

    def _start(self, name: str, now: float, ctx: Context) -> None:
        """Open the next round because ``name`` fired, at data time ``now``."""
        ctx.emit(
            "trigger.fired",
            self._volume,
            trigger=name,
            at=now,
            round=self.round + 1,
            drift=dict(self._drift),
        )
        self._volume, self._drift, self._opened_at = 0.0, {}, now
        ctx.emit("round.started", self.round + 1, round=self.round + 1, at=now)
        self._open(self.round + 1, self.state, ctx, bootstrap=self.round == 0)

    def _finish(self, ctx: Context) -> None:
        if not self.finished:
            self.finished = True
            # With triggers, whether the streams ended or the rounds cap cut it.
            why = {} if self.continuum is None else {"reason": "rounds"}
            if self._horizon:
                why["reason"] = "horizon"
            ctx.emit("run.finished", self.round, **why)
            _stop_children(self, ctx)

    @property
    def stopped(self) -> bool:
        return self.finished

    def _closed(self, aggregated: Contribution, metrics: dict, ctx: Context) -> None:
        held = keys_held_at(aggregated.state, self.sharing, self.levels, self.level)
        stats = dict(metrics) | {"edges_total": float(self.edges_below())}
        stats |= {f"edges_total/{g}": float(n) for g, n in self.holders_below().items()}
        state = self.server_optimizer.apply(
            self.state, _subset(aggregated.state, held), stats
        )
        if "train_loss" in metrics:
            ctx.emit(
                "round.train_loss",
                metrics["train_loss"],
                examples=metrics["train_examples"],
                round=self.round,
            )
        if self.promotion is None:
            self.state = state
            self._evaluate_and_go_on(ctx)
        else:
            self._propose(state, ctx)

    # --- versions (continuum C7) ----------------------------------------------------

    def _propose(self, state: State, ctx: Context) -> None:
        """The round's model is a challenger: the validators score it next to the
        champion, and the next round waits for the decision (spec §5)."""
        champion = self.versions[self.champion]
        version = Version(
            max(self.versions) + 1,
            champion.v,
            self.round,
            dict(state),
            _server_state(self),
        )
        self.versions[version.v] = version
        self._deciding = version
        ctx.emit(
            "version.candidate",
            version.v,
            parent=champion.v,
            round=self.round,
            digest=version.digest,
        )
        validators = self.validators()
        self._validation = {"waiting": set(validators), "reports": []}
        if not validators:
            self._decide(ctx)
            return
        both = {f"champion/{k}": v for k, v in champion.state.items()}
        both |= {f"challenger/{k}": v for k, v in version.state.items()}
        meta: dict[str, Any] = {"validate": version.v, "since": self._promoted_at}
        if self.promotion.window is not None:
            meta["window"] = float(self.promotion.window)
        for child in validators:
            ctx.send(
                Message(
                    kind="eval_request",
                    src=self.id,
                    dst=child,
                    round=self.round,
                    payload=Payload(state=both),
                    meta=meta,
                )
            )
        if self.deadline is not None:
            ctx.set_timer(self.deadline, "validate")

    def _eval_report(self, msg: Message, ctx: Context) -> None:
        if "validate" not in msg.meta:
            super()._eval_report(msg, ctx)
            return
        deciding, waiting = self._deciding, self._validation.get("waiting", set())
        if (
            deciding is None
            or msg.meta["validate"] != deciding.v
            or msg.src not in waiting
        ):
            _reject(ctx, msg, "validation report nobody asked for")
            return
        waiting.discard(msg.src)  # answered, even if what it says is left out
        metrics = msg.payload.metrics
        broken = [k for k, v in metrics.items() if not np.isfinite(v)]
        broken += [k for k, v in metrics.items() if k.endswith(".samples") and v < 0]
        if broken:  # left out, so the decision goes on without it
            _reject(ctx, msg, f"validation report with bad values: {sorted(broken)}")
        else:
            self._validation["reports"].append(dict(metrics))
        if not waiting:
            self._decide(ctx)

    def _decide(self, ctx: Context) -> None:
        ctx.cancel_timer("validate")
        version, self._deciding = self._deciding, None
        reports = self._validation["reports"]
        scores = {
            model: {part: _part_scores(reports, f"{model}.{part}.") for part in PARTS}
            for model in MODELS
        }
        decision = self.promotion.decide(scores)
        tags = dict(decision.measures)
        tags |= {"round": version.round, "reason": decision.reason}
        champion = self.champion
        if decision.promote:
            self._activate(version, ctx)
            ctx.emit("version.promoted", version.v, superseded=champion, **tags)
        else:
            version.status = "rejected"
            _load_server_state(self, self.versions[champion].server)
            ctx.emit("version.rejected", version.v, champion=champion, **tags)
        orders, self._orders = self._orders, []
        for msg in orders:
            self._order(msg, ctx)
        self._evaluate_and_go_on(ctx)

    def _activate(self, version: Version, ctx: Context) -> None:
        """Make ``version`` the champion: the edges get it from the next round."""
        current = self.versions[self.champion]
        if version.status != "superseded":  # a rollback keeps where it returns to
            version.replaced = current.v
        current.status = "superseded"
        version.status = "active"
        self.champion, self.state = version.v, dict(version.state)
        _load_server_state(self, version.server)
        self._promoted_at = ctx.now()

    def validators(self) -> list[str]:
        """The configured validators that registered (continuum C7)."""
        return sorted(self._validators & set(self.registered))

    def evaluators(self) -> list[str]:
        return [c for c in super().evaluators() if c not in self._validators]

    def _other(self, msg: Message, ctx: Context) -> None:
        """An operator's order: ``{"rollback": v}`` (``None``: to the champion the
        current one replaced) or ``{"promote": v}``, as a control message from a
        configured operator. Who that sender really is, the transport has to
        authenticate (#203); here the id is checked."""
        orders = ("rollback", "promote")
        if msg.kind != "control" or not any(k in msg.meta for k in orders):
            super()._other(msg, ctx)
        elif msg.src not in self.operators:
            _reject(ctx, msg, f"{msg.src} is not an operator")
        elif self.promotion is None:
            _reject(ctx, msg, "no versions: the run has no promotion policy")
        elif self._deciding is not None:
            self._orders.append(msg)  # after the decision, on what it decided
        else:
            self._order(msg, ctx)

    def _order(self, msg: Message, ctx: Context) -> None:
        current = self.versions[self.champion]
        if "rollback" in msg.meta:
            to = msg.meta["rollback"]
            to = current.replaced if to is None else to
            target = self.versions.get(to) if isinstance(to, int) else None
            if target is None or target.status != "superseded":
                _reject(ctx, msg, f"no superseded version {to} to roll back to")
                return
            self._activate(target, ctx)
            current.status = "rolled_back"
            ctx.emit(
                "version.rollback",
                target.v,
                previous=current.v,
                round=self.round,
                reason="operator",
            )
            return
        to = msg.meta["promote"]
        target = self.versions.get(to) if isinstance(to, int) else None
        if target is None or target.status not in ("rejected", "rolled_back"):
            _reject(ctx, msg, f"no rejected or rolled back version {to} to promote")
            return
        target.status = "candidate"  # it replaces the champion as a new one would
        self._activate(target, ctx)
        ctx.emit(
            "version.promoted",
            target.v,
            superseded=current.v,
            round=self.round,
            reason="operator",
        )

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
        if last_round_closed and not self.pending_eval and self._deciding is None:
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
        self.whole: State = {}  # every key received so far, for a child's first model
        self.parent_level = self.levels[self.levels.index(self.level) - 1]

    def _registered(self, ctx: Context) -> None:
        self._say_hello(self._meta(), ctx)

    def _heard_all(self, heard: dict[str, float], now: float, ctx: Context) -> None:
        payload = Payload(metrics={"at": now, "round": float(self.round), **heard})
        ctx.send(Message(kind="status", src=self.id, dst=self.parent, payload=payload))

    def _late_child(self, ctx: Context) -> None:
        if self._meta() != self._hello_meta:  # the parent counts what is below now
            self._say_hello(self._meta(), ctx)

    def _meta(self) -> dict[str, Any]:
        return {
            "role": "aggregator",
            "edges": self.edges_below(),
            "holders": self.holders_below(),
        }

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
        self.whole.update(self.received)
        bootstrap = bool(msg.meta.get("bootstrap"))
        if bootstrap:
            held = keys_held_at(self.received, self.sharing, self.levels, self.level)
            self.zone = _subset(self.received, held)
        state = {**self.received, **self.zone}
        self._open(msg.round, state, ctx, bootstrap, {**self.whole, **self.zone})

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
        replay: Any = None,
        replay_ratio: float = 0.25,
        unlabelled: Any = None,
        continuum: Any = None,
        metrics: Sequence[str] = ("loss", "accuracy"),
        validates: bool = False,
        update: str = "warm",
    ) -> None:
        super().__init__(node_id)
        if update not in UPDATES:
            raise ValueError(
                f"{node_id}: update must be one of {UPDATES}, not {update}"
            )
        if update != "warm":
            if stream is None:
                raise ValueError(f"{node_id}: update {update} needs a stream")
            if replay is not None or unlabelled is not None:
                raise ValueError(
                    f"{node_id}: update {update} takes no replay memory and no "
                    "unlabelled objective; they need warm"
                )
        # How a stream edge updates (the baselines of continuum C8): warm goes on
        # from the model received; frozen trains v0 on the history, with the
        # first model it gets, and is idle from then on; scratch starts every training from the initial weights,
        # on every trainable row it has.
        self.update = update
        self._initial: State | None = None  # the first model received, whole
        if validates and (train or stream is None):
            raise ValueError(f"{node_id}: a validator is an evaluator with a stream")
        # A validator scores the champion and a challenger on its stream (C7).
        self.validates = validates
        if replay is not None:  # refused here, not as an error inside a round
            if stream is None:
                raise ValueError(f"{node_id}: a replay memory needs a stream")
            if not isinstance(replay, Memory):
                raise TypeError(
                    f"{node_id}: {type(replay).__name__} is not a replay memory; "
                    "it needs capacity, rng, add, rows, sample, state and load_state"
                )
            if not 0 <= replay_ratio < 1:
                raise ValueError(
                    f"{node_id}: replay_ratio must be in [0, 1), not {replay_ratio}"
                )
        if unlabelled is not None:
            if stream is None:
                raise ValueError(f"{node_id}: an unlabelled objective needs a stream")
            if not isinstance(unlabelled, Objective):
                raise TypeError(
                    f"{node_id}: {type(unlabelled).__name__} is not an unlabelled "
                    "objective; it needs size, fit and term"
                )
            if not getattr(trainer, "takes_unlabelled", False):
                raise ValueError(
                    f"{node_id}: trainer {type(trainer).__name__} runs its own "
                    "loops and cannot train on unlabelled rows"
                )
        self.hello_retry = hello_retry
        # A stream (continuum C3): rows arrive over time, are predicted by the
        # model the edge serves when they come, and train once labelled.
        self.stream, self.metrics = stream, tuple(metrics)
        # A replay memory keeps rows already consumed, to train on them again.
        self.replay, self.replay_ratio = replay, replay_ratio
        self._replayed = 0  # rows the current training replays
        # An objective on the rows whose label has not arrived (continuum C5).
        self.unlabelled = unlabelled
        self._fit: dict[str, Any] | None = None  # the current training's pool
        self._clock = -np.inf  # continuum time up to which arrivals are handled
        self._served: State | None = None  # the last model received, as served
        if stream is not None:
            shape = (len(stream.data.y), stream.data.n_classes)
            self._logits = np.full(shape, np.nan)
            self._predicted = np.zeros(len(stream.data.y), bool)
            # The round whose training first used each row (-1: not yet). A row
            # counts as used only once that training finished with finite
            # weights, so a failed round trains the same rows next time.
            self._consumed_by = np.full(len(stream.data.y), -1)
            self._pending = np.zeros(len(stream.data.y), bool)
            # The round whose training first used each row unlabelled, and per
            # accepted pseudo-label its class, confidence and the round whose
            # received model generated it (the audit of spec §9).
            self._unlabelled_by = np.full(len(stream.data.y), -1)
            self._pool = np.zeros(len(stream.data.y), bool)
            self._pseudo_label = np.full(len(stream.data.y), -1)
            self._pseudo_confidence = np.full(len(stream.data.y), np.nan)
            self._generated_by = np.full(len(stream.data.y), -1)
        # A continuous federation (continuum C6): statuses, detectors and the
        # local trigger; only a training edge with a stream takes part.
        self.continuum = continuum if stream is not None and train else None
        self._ticked = -np.inf  # continuum time of the last status
        self._updated_at = (
            float(stream.available_at.min()) if stream is not None else 0.0
        )  # of its last successful training: what the local trigger counts from
        self._local_drift: dict[str, int] = {}
        self._seen_round = 0  # the last round whose model reached this edge
        drift = self.continuum.drift if self.continuum is not None else {}
        self._detectors = {k: create(detectors, d) for k, d in sorted(drift.items())}
        self._reference = Reference(stream) if self._detectors else None
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
        role = "validator" if self.validates else "evaluator"
        meta = {
            "role": "edge" if self.train else role,
            "edges": int(self.train),
            "tags": self.tags,
        }
        if self.train:
            meta["holders"] = dict.fromkeys(self._groups_up(self.model.state_dict()), 1)
        self._say_hello(meta, ctx)
        if self.continuum is not None:
            ctx.set_timer(self.continuum.status_every, "status")

    def _groups_up(self, keys: Iterable[str]) -> list[str]:
        """Parameter groups of ``keys`` that travel to the parent."""
        up = keys_crossing(keys, self.sharing, self.levels, self.parent_level)
        return sorted({group_of(k) for k in up if not is_aux(k)})

    def on_timer(self, name: str, ctx: Context) -> None:
        if name == "hello" and not self.acknowledged:
            self._send_hello(ctx)
        elif name == "status" and not self.stopped:
            self._status(ctx)
            if ctx.now() < self.continuum.until:  # see _Collector._tick
                ctx.set_timer(self.continuum.status_every, "status")

    def on_message(self, msg: Message, ctx: Context) -> None:
        if self._acknowledged(msg, ctx):
            return
        if self._stop_requested(msg):
            self.stopped = True
            ctx.cancel_timer("hello")
            ctx.cancel_timer("status")
            return
        if msg.src != self.parent or msg.round is None:
            _reject(ctx, msg, "unexpected sender")
        elif msg.kind == "eval_request" and self.validates:
            self._validate(msg, ctx)
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
        self._predict(new)
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
        # Every trainable row not used yet: a late, skipped or failed round loses
        # and repeats nothing. A window caps how old a row may be.
        since = -np.inf if s.window is None else now - s.window
        buffer = s.trainable(since, now) & (self._consumed_by < 0)
        self._pending = buffer
        if self.unlabelled is not None:
            # Arrived, label not here yet, never used unlabelled; a row whose
            # label comes later still trains as labelled.
            self._pool = (
                s.arrived(since, now) & (s.label_at > now) & (self._unlabelled_by < 0)
            )
        rows = buffer.copy()
        if self.update == "scratch":  # all it may train on, used or not
            rows = s.trainable(since, now)
        if self.replay is not None:
            # replay_ratio r of the training comes from memory: k = r·n / (1 − r);
            # with nothing new the edge stays idle, whatever it remembers
            n = int(buffer.sum())
            r = self.replay_ratio
            replayed = self.replay.sample(int(r * n / (1 - r) + 0.5))
            rows[replayed] = True
            self._replayed = len(replayed)  # reported if the training works
        return (s.take(rows) if buffer.any() else None), metrics

    def _report_memory(self, round: int, ctx: Context) -> None:
        """The memory after a training that worked, and what that training replayed."""
        s, kept = self.stream, self.replay.rows()
        age = float(np.mean(self._clock - s.observed_at[kept])) if len(kept) else 0.0
        ctx.emit(
            "diagnostic.memory",
            float(len(kept)),
            round=round,
            capacity=int(self.replay.capacity),
            age=age,  # since each kept row was observed, in data seconds
            classes=np.bincount(s.data.y[kept], minlength=s.data.n_classes).tolist(),
            replayed=self._replayed,
            **self.tags,
        )

    def _report_unlabelled(self, round: int, ctx: Context) -> None:
        """What the pool of a training that worked gave, and the audit it leaves."""
        fit, rows = self._fit, np.flatnonzero(self._pool)
        self._unlabelled_by[rows] = round
        tags: dict[str, Any] = {
            "pseudo_labels_accepted": fit["accepted"],
            "acceptance_rate": fit["accepted"] / fit["seen"],
            "confidence": fit["confidence"],
        }
        if fit["labels"] is not None and fit["accepted"]:
            mask = fit["mask"]
            accepted = rows[mask]
            self._pseudo_label[accepted] = fit["labels"][mask]
            self._pseudo_confidence[accepted] = fit["row_confidence"][mask]
            self._generated_by[accepted] = round
            # Against the hidden label: streams run only in simulation.
            hits = fit["labels"][mask] == self.stream.data.y[accepted]
            tags["pseudo_label_accuracy"] = float(hits.mean())
        ctx.emit(
            "diagnostic.unlabelled",
            float(fit["seen"]),
            round=round,
            **tags,
            **self.tags,
        )

    def _predict(self, rows: np.ndarray) -> None:
        """Predict the rows not predicted yet with the model served now."""
        todo = rows & ~self._predicted
        if self._served is None or not todo.any():
            return
        served = copy.deepcopy(self.model)
        load_arrays(served, self._served)
        self._logits[todo] = predict(served, self.stream.data.X[todo])
        self._predicted |= todo

    def _status(self, ctx: Context) -> None:
        """What reached the edge since its last status, sent up without data
        (spec §10): the rows that became trainable, and per kind of drift the
        window's statistic, which also feeds the edge's own detector. New rows
        are predicted now, by the model served, as the next round would."""
        s = self.stream
        lo, now = self._ticked, s.clock(ctx.now())
        self._ticked = now
        arrived = s.arrived(lo, now)
        self._predict(arrived & ~s.history)
        fresh = s.trainable(lo, now) & (self._consumed_by < 0)
        metrics = {
            "at": now,
            "round": float(self._seen_round),
            "fresh": float(fresh.sum()),
        }
        labelled = s.labelled(lo, now)
        for kind, detector in self._detectors.items():
            value, n = window(
                kind,
                s,
                arrived,
                labelled,
                self._logits,
                self._predicted,
                self._reference,
            )
            found = detector.update(value) if n else None
            if found is not None:
                self._reference.moved(kind)  # compared from now on with this one
                self._local_drift[kind] = self._local_drift.get(kind, 0) + 1
                ctx.emit(
                    "drift.detected",
                    found,
                    kind=kind,
                    window=value,
                    at=now,
                    **self.tags,
                )
            metrics |= {
                f"drift.{kind}.sum": value * n,
                f"drift.{kind}.n": float(n),
                f"drift.{kind}.detected": float(found is not None),
            }
        payload = Payload(metrics=metrics)
        ctx.send(Message(kind="status", src=self.id, dst=self.parent, payload=payload))

    def _wants_update(self, round: int, ctx: Context, first: bool) -> bool:
        """The local trigger (spec §10): whether this edge has an update for the
        round; without one, it always has. It does not hold back the first
        model, which trains v0 on the history, nor an edge whose stream has
        ended, which trains what is left."""
        trigger = None if self.continuum is None else self.continuum.edge_trigger
        if trigger is None or first or self._clock >= self.stream.drain:
            return True
        view = View(
            now=self._clock,
            since=self._updated_at,
            volume=float(self._pending.sum()),
            drift=dict(self._local_drift),
        )
        name = trigger.fired(view)
        if name is not None:
            ctx.emit(
                "trigger.fired",
                view.volume,
                trigger=name,
                at=view.now,
                round=round,
                **self.tags,
            )
        return name is not None

    def _train(self, msg: Message, ctx: Context) -> None:
        self._seen_round = msg.round
        received = dict(msg.payload.state)
        # The trainer is anchored to the model it shares: the first model a child
        # gets is whole, and its local groups must not be pulled to their start.
        anchor = _subset(
            received,
            keys_crossing(received, self.sharing, self.levels, self.parent_level),
        )
        own, metrics = self.data, {}
        if self.stream is not None:
            own, metrics = self._arrivals(msg.round, ctx)
        load_arrays(self.model, received)
        if self._initial is None:  # the first model a child gets is whole
            self._initial = dict(received)
        if self.stream is not None:
            self._served = state_arrays(self.model)
            # Nothing to train on, or no update wanted yet: idle, with its scores.
            first = bool(msg.meta.get("bootstrap"))
            frozen = self.update == "frozen" and not first  # v0 only
            if own is None or frozen or not self._wants_update(msg.round, ctx, first):
                _send_idle(self, msg.round, metrics, ctx)
                return
            if self.update == "scratch":  # served as received, trained afresh
                load_arrays(self.model, self._initial)
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
        saved, extra, self._fit = None, {}, None
        try:
            if self.unlabelled is not None and self._pool.any():
                # Features only, seen by the model as received.
                pool = self.stream.data.X[self._pool]
                self._fit = self.unlabelled.fit(self.model, pool)
                extra = {"unlabelled": self.unlabelled}
            # What a diverged round rolls back to: the model, and the trainer's
            # memory if it can snapshot it (built-ins can; a plugin may opt in).
            saved = getattr(self.trainer, "snapshot", lambda: None)()
            result = self.trainer.train(
                self.model, data, received=anchor, ctx=ctx, **extra
            )
        except Exception as exc:  # the edge counts as absent; the run goes on
            # A trainer may have changed the model or its memory before failing:
            # roll both back, as for non-finite weights, so a retry starts clean.
            if saved is not None:
                self.trainer.restore(saved)
            load_arrays(self.model, model_before)
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
        if self.stream is not None:  # trained with finite weights: now used
            self._consumed_by[self._pending] = msg.round
            if self.continuum is not None:  # what the local trigger counts from
                self._updated_at, self._local_drift = self._clock, {}
            if self.replay is not None:  # and only now eligible for replay
                fresh = np.flatnonzero(self._pending)  # in time order
                self.replay.add(fresh, self.stream.data.y[fresh])
                self._report_memory(msg.round, ctx)
            if self._fit is not None:  # the pool is used only now
                self._report_unlabelled(msg.round, ctx)
        if scoring and "local" in self.eval_models:
            metrics |= self._score("local", msg.round, ctx)
        try:
            if scoring and "personal" in self.eval_models:
                personal = getattr(self.trainer, "personal", lambda: None)()
                if personal is not None:
                    metrics |= self._score("personal", msg.round, ctx, personal)
            if finetuning:
                finetuned = self._finetuned(start, anchor, ctx)
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
            # A frozen group crosses as it arrived: nothing to hide, and noise
            # on it would only make it drift.
            fixed = {k: arrays[k] for k in frozen_keys(self.trainer, arrays)}
            released = {k: v for k, v in arrays.items() if k not in fixed}
            arrays = self.privacy.on_update(released, received, child_rng(ctx.rng))
            arrays |= fixed
            ctx.emit(
                "diagnostic.privacy_epsilon",
                self.privacy.epsilon(),
                round=msg.round,
                mechanism="local",
            )
        examples = result.examples
        if self.replay is not None:  # replayed rows were weighted when they were new
            examples = int(self._pending.sum())
        ctx.emit("edge.trained", result.loss, round=msg.round, examples=examples)
        # One vote per edge for trainers whose papers average clients (SCAFFOLD).
        uniform = getattr(self.trainer, "uniform_weights", False)
        payload = Payload(
            state=arrays,
            weights=dict.fromkeys(up, 1.0 if uniform else float(examples)),
            metrics={
                "train_loss": float(result.loss),
                "train_examples": float(examples),
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

    def _validate(self, msg: Message, ctx: Context) -> None:
        """Score the champion and the challenger on the labels that came: the
        recent window, and the reference before it (spec §5.2)."""
        s = self.stream
        now = s.clock(ctx.now())
        window = msg.meta.get("window")
        lo = (
            s.clock(float(msg.meta.get("since", 0.0)))
            if window is None
            else now - window
        )
        known = np.isfinite(s.label_at) & (s.label_at <= now)
        recent = known & (s.label_at > lo)
        parts = {"recent": recent, "reference": known & ~recent}
        state = msg.payload.state
        metrics: dict[str, float] = {}
        for model in MODELS:
            head = f"{model}/"
            load_arrays(
                self.model,
                {k[len(head) :]: v for k, v in state.items() if k.startswith(head)},
            )
            for part, rows in parts.items():
                scores, samples = (
                    self.evaluate(self.model, s.take(rows)) if rows.any() else ({}, 0)
                )
                # a score that is not a number is left out: the decision then
                # misses it and fails closed, and no codec has to carry a NaN
                metrics |= {
                    f"{model}.{part}.{k}": float(v)
                    for k, v in scores.items()
                    if np.isfinite(v)
                }
                metrics[f"{model}.{part}.samples"] = float(samples)
        ctx.send(
            Message(
                kind="eval_report",
                src=self.id,
                dst=self.parent,
                round=msg.round,
                payload=Payload(metrics=metrics),
                meta={"validate": msg.meta.get("validate")},
            )
        )
