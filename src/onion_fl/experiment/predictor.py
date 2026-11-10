from __future__ import annotations

"""The factorised predictor: rounds to target × time per round (N7, issue #211).

The analytic model that Luo et al. (2021) and Valli et al. (CCGrid 2026) use to
choose a configuration before deploying it, as a decision backend of the
fidelity harness (I1 §12): a backend that does not decide better than it adds
nothing. Its two factors are estimated apart:

- **rounds to target**, measured on a *control* run of the same configuration
  in simulation, synchronous and without deadline or data–device coupling
  (what a data-only study gives): the first round whose mean root ``metric``
  reaches the target; the rounds run when none does (censored);
- **time per round**, computed from the compute and network models of the
  configuration itself, the ones its own run uses: over the tree of the round
  (the nodes the control sent the model to), a node's time is the longest of
  its children's ``down + own time + up``, plus its aggregation. An edge's own
  time is the mean duration of its work; a link takes ``bytes * 8 /
  bandwidth + mean latency``. The work and the bytes are those of the control
  run (the model and the data are the same). It ignores what a closed form
  cannot see: contention for a CPU or a shared segment, jitter and losses.

The prediction of a configuration is the product (the sum of the predicted
times of its first ``R`` rounds) for ``time_to_target``, capped at the
predicted duration of the run as the harness censors it, and the sum over all
the rounds for ``time``. The configuration that minimises it wins.

The **FLOSS-style variant** (arXiv 2507.23115) corrects for non-ignorable
participation, which the plain predictor ignores. With the deadlines of the
configuration, a child counts at its parent with the probability that it
arrives in time: for an edge, that its compute fits in the deadline minus its
links (analytic for ``lognormal_rate``, empirical for ``trace``, a yes or a no
for deterministic models); for an aggregator, whether its expected arrival
does. A round with a child that may miss lasts until the deadline in that
proportion, and the rounds to target grow by the inverse of the participation
weighted by the edges' examples, as inverse-probability weighting keeps the
update unbiased at the price of its effective size.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from onion_fl.experiment.ceiling import LOWER_IS_BETTER
from onion_fl.runtime.network import LinkChannel, LinkProfile

VARIANTS = ("factorised", "floss")
PREDICTED = ("time", "time_to_target")  # the metrics the predictor scores
TIMING = ("deadline", "quorum", "close_at_quorum", "staleness")


# --- pairing with the control --------------------------------------------------------


def _strip(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_strip(v, keys) for v in value]
    return value


def control_id(config: Mapping[str, Any]) -> str:
    """The ``design_id`` of a config without what its control drops.

    The control runs the same design synchronously, without the round timing
    of the topology (deadlines, quorum, staleness) or the assignment of edges
    to devices (the coupling).
    """
    from onion_fl.experiment.fidelity import design_id

    stripped = dict(config) | {"topology": _strip(config.get("topology"), TIMING)}
    if stripped.get("infrastructure"):
        stripped["infrastructure"] = {
            k: v for k, v in stripped["infrastructure"].items() if k != "assignment"
        }
    return design_id(stripped)


def has_deadline(config: Mapping[str, Any]) -> bool:
    """Does any node of the config's topology set a deadline?"""

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("deadline") is not None or any(
                walk(v) for v in value.values()
            )
        return isinstance(value, list) and any(walk(v) for v in value)

    return walk(config.get("topology"))


# --- the models ----------------------------------------------------------------------


def mean_duration(model: Any, samples: float) -> float | None:
    """Mean duration of ``samples`` of work; None when no closed form gives it.

    ``samples_per_second`` and ``lognormal_rate`` last ``overhead + samples /
    samples_per_second`` on average, ``trace`` the mean of its durations;
    ``measured`` depends on this machine's wall clock.
    """
    if model is None or samples <= 0:
        return 0.0
    if hasattr(model, "samples_per_second"):
        return getattr(model, "overhead_s", 0.0) + samples / model.samples_per_second
    if hasattr(model, "durations"):
        return float(np.mean(model.durations))
    return None


def within(model: Any, samples: float, budget: float) -> float:
    """Probability that ``samples`` of work end within ``budget`` seconds."""
    if budget < 0:
        return 0.0
    if model is None or samples <= 0:
        return 1.0
    if hasattr(model, "durations"):
        return float(np.mean(np.asarray(model.durations) <= budget))
    if hasattr(model, "samples_per_second"):
        work = budget - getattr(model, "overhead_s", 0.0)
        if work <= 0:
            return 0.0
        ratio = work * model.samples_per_second / samples  # the factor that fits
        sigma = getattr(model, "sigma", 0.0)
        if sigma == 0:
            return float(ratio >= 1.0)
        # duration factor exp(sigma Z - sigma² / 2) <= ratio, Z standard normal
        return float(stats.norm.cdf((math.log(ratio) + sigma**2 / 2) / sigma))
    return 1.0  # no closed form: assumed in time, as its mean counts as 0


def link_time(channel: Any, size: float) -> float:
    """Expected time of ``size`` bytes on a channel: transmission plus mean latency."""
    if isinstance(channel, LinkChannel):
        profile, direction = channel.profile, channel.direction
    elif hasattr(channel, "network"):  # a path: its own "up" is its direction
        fields = channel.network.profile(channel.hops, 0.0)
        if fields is None:
            return 0.0
        profile, direction = LinkProfile(**fields), "up"
    else:
        return 0.0
    return profile.transmission_s(int(size), direction) + profile.latency_s


@dataclass
class Plan:
    """What the predictor needs of a configuration's federation."""

    compute: dict[str, Any]
    channels: dict[tuple[str, str], Any]
    deadlines: dict[str, float | None]
    combine: dict[str, float]  # bytes of one child's state at each aggregation node
    unknown: list[str] = field(default_factory=list)  # no closed-form duration

    def compute_time(self, node: str, samples: float) -> float:
        return mean_duration(self.compute.get(node), samples) or 0.0

    def link(self, src: str, dst: str, size: float) -> float:
        channel = self.channels.get((src, dst))
        return 0.0 if channel is None else link_time(channel, size)


def plan_of(config: Mapping[str, Any], seed: int, scenario: str = "") -> Plan:
    """Build (without running) the federation of a run's config and read its models."""
    from onion_fl.core.ids import config_id
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import _scenario_data, build_scenario
    from onion_fl.experiment.sweep import Scenario
    from onion_fl.learning.sharing import keys_crossing

    parsed = parse_experiment(dict(config))
    built = Scenario(scenario, seed, parsed, config_id(dict(config)))
    topology, split, placement, _ = _scenario_data(built, None)
    federation = build_scenario(built, topology, split, placement, parent=None)
    runtime = federation.runtime
    aggregation = [federation.coordinator, *federation.aggregators.values()]
    nodes = [n.id for n in aggregation] + list(federation.edges)
    compute = {n: runtime.compute_of(n) for n in nodes}
    links = [(n.id, c) for n in aggregation for c in sorted(n.children)]
    state = federation.coordinator.state
    return Plan(
        compute=compute,
        channels={
            (s, d): runtime.channel_of(s, d)
            for parent, child in links
            for s, d in ((child, parent), (parent, child))
        },
        deadlines={n.id: n.deadline for n in aggregation},
        combine={
            n.id: float(
                sum(
                    np.asarray(state[k]).nbytes
                    for k in keys_crossing(state, n.sharing, n.levels, n.level)
                )
            )
            for n in aggregation
        },
        unknown=sorted(n for n, m in compute.items() if mean_duration(m, 1.0) is None),
    )


# --- the rounds of the control -------------------------------------------------------


@dataclass
class Round:
    children: dict[str, list[str]] = field(default_factory=dict)
    down: dict[tuple[str, str], float] = field(default_factory=dict)
    up: dict[tuple[str, str], float] = field(default_factory=dict)
    samples: dict[str, float] = field(default_factory=dict)
    examples: dict[str, float] = field(default_factory=dict)


def rounds_of(events: Sequence[Mapping[str, Any]]) -> tuple[str | None, dict]:
    """The root and, per round, its tree, bytes and work, from a run's events."""
    root = None
    rounds: dict[int, Round] = {}
    for e in events:
        tags = e.get("tags") or {}
        name, rnd = e["name"], tags.get("round")
        if name == "round.started" and e.get("role") == "coordinator":
            root = root or e["node"]
        if rnd is None:
            continue
        if name == "message.sent" and tags.get("kind") in ("global_model", "update"):
            r = rounds.setdefault(rnd, Round())
            link = (e["node"], tags.get("dst"))
            if tags["kind"] == "global_model" and link not in r.down:
                r.children.setdefault(link[0], []).append(link[1])
                r.down[link] = float(e["value"] or 0)
            elif tags["kind"] == "update":
                r.up.setdefault(link, float(e["value"] or 0))
        elif name == "edge.trained":
            r = rounds.setdefault(rnd, Round())
            r.examples[e["node"]] = float(tags.get("examples") or 0)
            r.samples[e["node"]] = float(tags.get("samples", tags.get("examples")) or 0)
    return root, dict(sorted(rounds.items()))


def rounds_to_target(
    events: Sequence[Mapping[str, Any]], metric: str, target: float
) -> tuple[int | None, bool]:
    """First round whose mean root ``metric`` reaches ``target``, and whether it is
    censored (never reached: the last round evaluated)."""
    lower = metric in LOWER_IS_BETTER
    values: dict[int, list[float]] = {}
    for e in events:
        if e["name"] == f"eval.{metric}" and e.get("role") == "coordinator":
            if e.get("round") is not None:
                values.setdefault(e["round"], []).append(float(e["value"]))
    for rnd in sorted(values):
        mean = float(np.mean(values[rnd]))
        if (mean <= target) if lower else (mean >= target):
            return rnd, False
    return (max(values) if values else None), True


# --- the prediction ------------------------------------------------------------------


def _node(
    node: str, rnd: Round, plan: Plan, floss: bool
) -> tuple[float, dict[str, float]]:
    """Expected time from a node's receiving the model to its sending the update,
    and the probability that each edge below it counts."""
    kids = [k for k in rnd.children.get(node, []) if (k, node) in rnd.up]
    if node not in rnd.children:  # an edge: its training
        return plan.compute_time(node, rnd.samples.get(node, 0.0)), {node: 1.0}
    deadline = plan.deadlines.get(node) if floss else None
    arrivals, everyone, reach = [], 1.0, {}
    for kid in kids:
        own, below = _node(kid, rnd, plan, floss)
        links = plan.link(node, kid, rnd.down[(node, kid)]) + plan.link(
            kid, node, rnd.up[(kid, node)]
        )
        arrival = links + own
        if deadline is None:
            made = 1.0
        elif kid not in rnd.children:  # an edge: its compute decides
            made = within(
                plan.compute.get(kid), rnd.samples.get(kid, 0.0), deadline - links
            )
        else:
            made = float(arrival <= deadline)
        arrivals.append(arrival)
        everyone *= made
        reach |= {edge: p * made for edge, p in below.items()}
    longest = max(arrivals, default=0.0)
    if deadline is not None:  # it waits for the deadline when one may miss
        longest = everyone * min(longest, deadline) + (1 - everyone) * deadline
    combine = plan.combine.get(node, 0.0) * len(kids)
    return longest + plan.compute_time(node, combine), reach


def predict(
    plan: Plan,
    events: Sequence[Mapping[str, Any]],
    target: tuple[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per variant, the predicted rounds, time per round, ``time`` and
    ``time_to_target`` of a configuration, from its plan and its control's events."""
    root, rounds = rounds_of(events)
    if root is None or not rounds:
        raise ValueError("the control run has no rounds")
    needed, censored = (
        (None, False) if target is None else rounds_to_target(events, *target)
    )
    out = {}
    for variant in VARIANTS:
        times, shares = [], []
        for rnd in rounds.values():
            time, reach = _node(root, rnd, plan, floss=variant == "floss")
            weights = {e: rnd.examples.get(e, 0.0) for e in reach}
            total = sum(weights.values())
            times.append(time)
            shares.append(
                sum(w * reach[e] for e, w in weights.items()) / total if total else 1.0
            )
        result = {
            "rounds": len(times),
            "round_times": times,
            "round_time": float(np.mean(times)),
            "time": float(np.sum(times)),
            "participation": float(np.mean(shares)),
        }
        if needed is not None:
            first = times[: min(needed, len(times))]
            share = float(np.mean(shares[: len(first)]))
            if variant == "factorised":
                to_target = float(needed)
            else:  # nothing counts: it never gets there
                to_target = needed / share if share > 0 else math.inf
            predicted = to_target * float(np.mean(first))
            result |= {
                "rounds_to_target": to_target,
                "censored": censored or predicted > result["time"],
                "time_to_target": min(predicted, result["time"]),
            }
        out[variant] = result
    return out


def predictions(
    cells: Mapping[Any, Any],
    control: Mapping[Any, Any],
    target: tuple[str, float] | None = None,
) -> tuple[dict[Any, dict[str, dict[str, Any]]], list[str]]:
    """The prediction of every run of ``cells`` (the fidelity harness's cells,
    by key), with the models of its own config and its control's events.

    A run of ``control`` is the control of a cell when it ran the same
    ``control_id`` with the same seed. A cell without one, or whose federation
    cannot be built here (its data missing, say), is left out with a warning.
    """
    from onion_fl.observability.events import read_events

    twins = {(control_id(c.meta["config"]), c.seed): c for c in control.values()}
    warnings = []
    timed = sorted(
        c.path.name for c in twins.values() if has_deadline(c.meta["config"])
    )
    if timed:
        warnings.append(
            f"predictor: {len(timed)} control run(s) have deadlines (e.g. "
            f"{timed[0]}); its rounds to target should come from synchronous runs "
            "without them (--control)"
        )
    others = sorted({c.backend for c in twins.values()} - {"sim"})
    if others:
        warnings.append(f"predictor: control runs on {others}, not sim")
    out = {}
    for key, cell in cells.items():
        twin = twins.get((control_id(cell.meta["config"]), cell.seed))
        if twin is None:
            warnings.append(f"predictor: {cell.path.name} has no control run")
            continue
        try:
            plan = plan_of(cell.meta["config"], cell.seed, cell.scenario)
            out[key] = predict(plan, read_events(twin.path / "events.jsonl"), target)
        except Exception as exc:  # the harness goes on without this prediction
            warnings.append(f"predictor: {cell.path.name}: {type(exc).__name__}: {exc}")
            continue
        if plan.unknown:
            warnings.append(
                f"predictor: {cell.path.name}: no closed-form compute time at "
                f"{plan.unknown}; it counts as 0"
            )
    return out, warnings
