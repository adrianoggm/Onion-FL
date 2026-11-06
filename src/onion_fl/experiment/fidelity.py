from __future__ import annotations

"""Fidelity of one backend against another (I1 §12, issue #191).

``onion_fl fidelity <runs A> <runs B>`` compares two folders of runs of the
same experiment: ``A`` is the reference (closer to the physical system), ``B``
the cheaper backend under test. A run of ``A`` and one of ``B`` are a pair
when they ran the same configuration with the same seed: the same
``design_id``, the ``config_id`` of the config without what only says where
and how it ran (``backend``, ``runtime.mode``, ``timeout``, ``heartbeat``,
``paths`` and the ``transport`` of the links of an inline topology). So
``A`` and ``B`` can also be one backend over two transports, MQTT and gRPC.

Three levels:

- **functional**, per pair: the final model (bit for bit, and within the
  tolerance of ``test_runtime_equivalence``), the sequence of protocol events
  of every node (rounds started, closed, failed, idle, abandoned; the end of
  the run) and the bytes per logical link and round. Traffic outside rounds
  (``hello``, acknowledgements, ``stop``) is reported, not judged: a real
  child repeats its ``hello`` until its parent answers.
- **performance**: the relative error ``(B - A) / A`` of the round
  durations at the root, of the compute time of every edge per round
  (from the model's arrival to its update's departure), of the latency of
  every link, of the run's duration and, with a target, of the time to it;
  the mean per pair, then over pairs with a 95 % bootstrap interval.
- **decision**, per metric, over the ``K`` configurations: Kendall's τ-b
  between the rankings of the mean over seeds, top-1 agreement and the
  selection regret (score in ``A`` of the winner of ``B`` minus the best score
  in ``A``, in the metric's units and relative to it), with 95 % intervals and
  the probability of top-1 agreement from a bootstrap paired by seed (the
  same resampled seeds in ``A`` and ``B`` for every configuration).

Every regret and top-1 is shown next to the **noise ceiling** (spec N6): the
same quantities between replicas of ``A``, from ``onion_fl ceiling`` output
when given, else between the seeds of ``A`` (paired replicas). A place is
kept for the **factorised predictor** (A7), which fills it later. When every
configuration ties on a metric (constant predictors, C3–C5), the decision on
it does not inform, and the result says so.
"""

import csv
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from onion_fl.core.ids import config_id
from onion_fl.experiment.backends import ALIASES
from onion_fl.experiment.ceiling import LOWER_IS_BETTER, ceiling, final_score
from onion_fl.observability.events import read_events
from onion_fl.observability.run import verify_run

PROTOCOL = (
    "round.started",
    "round.closed",
    "round.quorum_failed",
    "round.idle",
    "round.abandoned",
    "run.finished",
)
SYSTEM = ("time", "bytes", "time_to_target")  # from the run itself, lower is better
RUNTIME_ONLY = ("mode", "timeout", "heartbeat")
BOOTSTRAP = 2000
MIN_SEEDS = 5  # for decision fidelity (I1 §12, §17)
RTOL, ATOL = 1e-6, 1e-7  # as test_runtime_equivalence


def _without_transport(value: Any) -> Any:
    """A topology without the transport of its links (only ``LinkSpec`` has one)."""
    if isinstance(value, Mapping):
        return {k: _without_transport(v) for k, v in value.items() if k != "transport"}
    if isinstance(value, list):
        return [_without_transport(v) for v in value]
    return value


def design_id(config: Mapping[str, Any]) -> str:
    """The ``config_id`` of a run's config without where and how it ran."""
    design = {k: v for k, v in config.items() if k not in ("backend", "paths")}
    runtime = {
        k: v for k, v in (design.get("runtime") or {}).items() if k not in RUNTIME_ONLY
    }
    design |= {"runtime": runtime}
    if isinstance(design.get("topology"), Mapping):
        design["topology"] = _without_transport(design["topology"])
    return config_id(design)


@dataclass(frozen=True)
class Cell:
    design: str
    seed: int
    path: Path
    meta: dict[str, Any]

    @property
    def backend(self) -> str:
        name = self.meta.get("backend", "sim")
        return ALIASES.get(name, name)

    @property
    def scenario(self) -> str:
        return self.meta.get("scenario", "")


def load_cells(
    runs: str | Path, experiment: str | None = None
) -> tuple[dict[tuple[str, int], Cell], list[str]]:
    """The finished runs of a folder by ``(design_id, seed)``, the newest of each."""
    cells: dict[tuple[str, int], Cell] = {}
    warnings = []
    for run_json in sorted(Path(runs).glob("*/run.json")):
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        config = meta.get("config") or {}
        if experiment is not None and config.get("name") != experiment:
            continue
        if meta.get("status") != "finished":
            warnings.append(f"{run_json.parent.name}: {meta.get('status')}, left out")
            continue
        cell = Cell(design_id(config), int(meta["seed"]), run_json.parent, meta)
        key = (cell.design, cell.seed)
        if key in cells:
            older, newer = sorted(
                (cells[key], cell), key=lambda c: c.meta.get("started_at", "")
            )
            warnings.append(
                f"{older.path.name}: same configuration and seed as "
                f"{newer.path.name}; the newer one is compared"
            )
            cell = newer
        cells[key] = cell
    return cells, warnings


# --- decision ------------------------------------------------------------------------


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Kendall's τ-b between two score vectors; None when either is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(stats.kendalltau(a, b, variant="b").statistic)


def _winner(scores: np.ndarray, minimize: bool) -> int:
    """The best configuration; a tie goes to the first, as in the ceiling."""
    return int(np.argmin(scores) if minimize else np.argmax(scores))


def selection_regret(
    a: Sequence[float], b: Sequence[float], minimize: bool = False
) -> tuple[float, float | None]:
    """Score in ``a`` of the winner of ``b`` minus the best in ``a``; always >= 0.

    Also relative to the best score in ``a`` (None when it is 0).
    """
    a = np.asarray(a, float)
    best = a[_winner(a, minimize)]
    picked = a[_winner(np.asarray(b, float), minimize)]
    regret = float(picked - best if minimize else best - picked)
    return regret, (regret / abs(best) if best != 0 else None)


def ranking_agreement(
    a: Sequence[float], b: Sequence[float], minimize: bool = False
) -> dict[str, Any]:
    """τ-b, top-1 agreement and selection regret of two score vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    regret, relative = selection_regret(a, b, minimize)
    return {
        "tau_b": kendall_tau_b(a, b),
        "top1": bool(_winner(a, minimize) == _winner(b, minimize)),
        "regret": regret,
        "regret_relative": relative,
    }


def _ci(values: Sequence[float]) -> list[float] | None:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def decision(
    a: Any,
    b: Any,
    minimize: bool = False,
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """Decision fidelity of two tables of scores, ``seeds x configurations``.

    Row ``s`` of ``a`` and of ``b`` hold the same seed. The point estimate
    ranks the mean over seeds; the bootstrap resamples the seeds, the same
    ones in both tables, and gives a 95 % interval of each quantity and the
    probability of top-1 agreement.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 1:
        raise ValueError("decision: a and b must be seeds x configurations alike")
    out = ranking_agreement(a.mean(axis=0), b.mean(axis=0), minimize)
    n = a.shape[0]
    out |= {"seeds": n, "configurations": a.shape[1]}
    if n < 2:
        return out | {"ci": None, "p_top1": None}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        rows = rng.integers(0, n, size=n)
        draws.append(
            ranking_agreement(a[rows].mean(axis=0), b[rows].mean(axis=0), minimize)
        )
    return out | {
        "ci": {
            key: _ci([d[key] for d in draws])
            for key in ("tau_b", "regret", "regret_relative")
        },
        "p_top1": float(np.mean([d["top1"] for d in draws])),
    }


def ties(scores: Sequence[float]) -> bool:
    """Do all the configurations score the same (within rounding)?"""
    scores = np.asarray(scores, float)
    return len(scores) > 1 and bool(
        np.allclose(scores, scores[0], rtol=1e-9, atol=1e-12)
    )


# --- functional ----------------------------------------------------------------------


def _events(cell: Cell) -> list[dict[str, Any]]:
    return read_events(cell.path / "events.jsonl")


def _model(cell: Cell) -> dict[str, np.ndarray] | None:
    path = cell.path / "model.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def protocol(events: Sequence[Mapping[str, Any]]) -> dict[str, list[list[Any]]]:
    """Per node, its protocol events in order, as ``[name, round]``."""
    out: dict[str, list[list[Any]]] = {}
    for e in events:
        if e["name"] in PROTOCOL:
            out.setdefault(e["node"], []).append([e["name"], e.get("round")])
    return out


def link_bytes(events: Sequence[Mapping[str, Any]]) -> tuple[dict, dict]:
    """Bytes sent per ``(src, dst, round)`` inside rounds, and per link outside."""
    rounds: dict[tuple, int] = {}
    control: dict[tuple, int] = {}
    for e in events:
        if e["name"] != "message.sent":
            continue
        tags = e["tags"]
        size = int(e["value"] or 0)
        if tags.get("round") is None:
            key = (e["node"], tags.get("dst"))
            control[key] = control.get(key, 0) + size
        else:
            key = (e["node"], tags.get("dst"), tags["round"])
            rounds[key] = rounds.get(key, 0) + size
    return rounds, control


def functional(a: Cell, b: Cell) -> dict[str, Any]:
    """Model, protocol events per node and bytes per link of one pair."""
    events_a, events_b = _events(a), _events(b)
    model_a, model_b = _model(a), _model(b)
    if model_a is None or model_b is None:
        model = {"match": None, "bitwise": None, "max_abs_diff": None}
    elif model_a.keys() != model_b.keys():
        model = {"match": False, "bitwise": False, "max_abs_diff": None}
    else:
        diffs = [
            float(np.max(np.abs(model_a[k] - model_b[k]), initial=0.0)) for k in model_a
        ]
        model = {
            "match": all(
                np.allclose(model_a[k], model_b[k], rtol=RTOL, atol=ATOL)
                for k in model_a
            ),
            "bitwise": all(np.array_equal(model_a[k], model_b[k]) for k in model_a),
            "max_abs_diff": max(diffs, default=0.0),
        }
    proto_a, proto_b = protocol(events_a), protocol(events_b)
    differing = sorted(
        n for n in set(proto_a) | set(proto_b) if proto_a.get(n) != proto_b.get(n)
    )
    (bytes_a, control_a), (bytes_b, control_b) = (
        link_bytes(events_a),
        link_bytes(events_b),
    )
    links = sorted(
        {k[:2] for k in bytes_a.keys() ^ bytes_b.keys()}
        | {k[:2] for k in bytes_a.keys() & bytes_b.keys() if bytes_a[k] != bytes_b[k]}
    )
    return {
        "model": model,
        "protocol": {"match": not differing, "nodes_differing": differing},
        "bytes": {
            "match": bytes_a == bytes_b,
            "a": sum(bytes_a.values()),
            "b": sum(bytes_b.values()),
            "links_differing": [list(k) for k in links],
            "control": {"a": sum(control_a.values()), "b": sum(control_b.values())},
        },
        "complete": bool(model["match"]) and not differing and bytes_a == bytes_b,
    }


# --- performance ---------------------------------------------------------------------


def timings(
    events: Sequence[Mapping[str, Any]], target: tuple[str, float] | None = None
) -> dict[str, dict[Any, float]]:
    """Per quantity, its values by key, on the backend's clock (``t_virtual``)."""
    started, closed = {}, {}
    arrived: dict[tuple, float] = {}
    left: dict[tuple, float] = {}
    sent: dict[str, tuple[tuple, float]] = {}
    latency: dict[tuple, list[float]] = {}
    for e in events:
        name, tags, t = e["name"], e["tags"] or {}, e["t_virtual"]
        role, rnd = e.get("role"), tags.get("round")
        if role == "coordinator" and name == "round.started":
            started.setdefault(rnd, t)
        elif role == "coordinator" and name == "round.closed":
            closed.setdefault(rnd, t)
        elif name == "message.sent":
            if tags.get("msg_id") is not None:
                sent[tags["msg_id"]] = ((e["node"], tags.get("dst")), t)
            if role == "edge" and rnd is not None:
                left.setdefault((e["node"], rnd), t)
        elif name == "message.delivered":
            link_t = sent.get(tags.get("msg_id"))
            if link_t is not None:
                latency.setdefault(link_t[0], []).append(t - link_t[1])
            if role == "edge" and rnd is not None:
                arrived.setdefault((e["node"], rnd), t)
    out = {
        "round_time": {r: closed[r] - started[r] for r in closed if r in started},
        "compute_time": {
            k: left[k] - arrived[k]
            for k in left
            if k in arrived and left[k] >= arrived[k]
        },
        "link_latency": {k: float(np.mean(v)) for k, v in latency.items()},
        "duration": {"run": max((e["t_virtual"] or 0.0 for e in events), default=0.0)},
    }
    if target is not None:
        out["time_to_target"] = {"run": time_to_target(events, *target)}
    return out


def time_to_target(
    events: Sequence[Mapping[str, Any]], metric: str, target: float
) -> float:
    """First instant the root's mean ``metric`` of a round reaches ``target``.

    The run's duration when it never does (censored), as in ``analysis``.
    """
    lower = metric in LOWER_IS_BETTER
    rounds: dict[Any, list[tuple[float, float]]] = {}
    for e in events:
        if e["name"] == f"eval.{metric}" and e.get("role") == "coordinator":
            rounds.setdefault(e.get("round"), []).append((e["t_virtual"], e["value"]))
    for rnd in sorted(r for r in rounds if r is not None):
        value = float(np.mean([v for _, v in rounds[rnd]]))
        if (value <= target) if lower else (value >= target):
            return float(max(t for t, _ in rounds[rnd]))
    return float(max((e["t_virtual"] or 0.0 for e in events), default=0.0))


def _bootstrap_mean(
    values: Sequence[float], resamples: int, seed: int
) -> list[float] | None:
    values = np.asarray(values, float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values))).mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def performance(
    pairs: Sequence[tuple[dict, dict]], resamples: int = BOOTSTRAP, seed: int = 0
) -> dict[str, dict[str, Any]]:
    """Relative error ``(B - A) / A`` per quantity, mean per pair, then over pairs."""
    out = {}
    names = sorted({q for a, b in pairs for q in a.keys() & b.keys()})
    for quantity in names:
        signed, absolute, values = [], [], 0
        mean_a, mean_b = [], []
        for a, b in pairs:
            common = [k for k in a.get(quantity, {}) if k in b.get(quantity, {})]
            errors = [
                (b[quantity][k] - a[quantity][k]) / a[quantity][k]
                for k in common
                if a[quantity][k] > 0
            ]
            if not errors:
                continue
            values += len(errors)
            signed.append(float(np.mean(errors)))
            absolute.append(float(np.mean(np.abs(errors))))
            mean_a.append(float(np.mean([a[quantity][k] for k in common])))
            mean_b.append(float(np.mean([b[quantity][k] for k in common])))
        if not signed:
            continue
        out[quantity] = {
            "pairs": len(signed),
            "values": values,
            "a_mean": float(np.mean(mean_a)),
            "b_mean": float(np.mean(mean_b)),
            "relative_error": float(np.mean(signed)),
            "relative_error_ci": _bootstrap_mean(signed, resamples, seed),
            "absolute_relative_error": float(np.mean(absolute)),
            "absolute_relative_error_ci": _bootstrap_mean(absolute, resamples, seed),
        }
    return out


# --- the harness ---------------------------------------------------------------------


def score(
    cell: Cell,
    metric: str,
    series: str | None = None,
    target: tuple[str, float] | None = None,
) -> float:
    """What a configuration scores on a metric in one run (summary or events)."""
    if metric == "time_to_target":
        if target is None:
            raise ValueError("time_to_target needs a target, metric=value")
        return time_to_target(_events(cell), *target)
    if metric in ("time", "bytes"):
        summary = json.loads((cell.path / "summary.json").read_text(encoding="utf-8"))
        return float(
            summary["t_virtual_end"]
            if metric == "time"
            else summary["messages"]["bytes"]
        )
    return final_score(cell.path, metric, series)


def _ceiling_for(
    metric: str, table: np.ndarray, minimize: bool, given: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if given is not None and given.get("metric") == metric:
        return {
            "source": "ceiling",
            "top1": given["top1"],
            "top1_chance": given["top1_chance"],
            "regret": given["regret"]["mean"],
            "regret_relative": given["regret"].get("relative_mean"),
        }
    if table.shape[0] < 2:
        return None
    own = ceiling(table, minimize=minimize)
    return {
        "source": "seeds of A",
        "top1": own["top1"],
        "top1_chance": own["top1_chance"],
        "regret": own["regret"]["mean"],
        "regret_relative": own["regret"]["relative_mean"],
    }


def fidelity(
    a: str | Path,
    b: str | Path,
    metrics: Sequence[str] = ("accuracy", "time"),
    series: str | None = None,
    target: tuple[str, float] | None = None,
    experiment: str | None = None,
    noise_ceiling: Mapping[str, Any] | None = None,
    minimize: Sequence[str] = (),
    resamples: int = BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """Functional, performance and decision fidelity of the runs of ``b`` against ``a``."""
    cells_a, warn_a = load_cells(a, experiment)
    cells_b, warn_b = load_cells(b, experiment)
    warnings = [f"A {w}" for w in warn_a] + [f"B {w}" for w in warn_b]
    keys = sorted(cells_a.keys() & cells_b.keys())
    for side, cells in (("A", cells_a), ("B", cells_b)):
        alone = sorted(cells.keys() - set(keys))
        if alone:
            warnings.append(
                f"{len(alone)} run(s) of {side} have no pair in the other folder"
            )
    if not keys:
        raise ValueError(
            f"no run of {a} and {b} shares a configuration and seed; are they "
            "the same experiment?"
        )
    for cell in [cells_a[k] for k in keys] + [cells_b[k] for k in keys]:
        if not verify_run(cell.path):
            warnings.append(f"{cell.path.name}: its files do not match its run_hash")

    pairs = []
    times = []
    for key in keys:
        ca, cb = cells_a[key], cells_b[key]
        pairs.append(
            {
                "design_id": key[0],
                "seed": key[1],
                "scenario": ca.scenario,
                "a": _identity(ca),
                "b": _identity(cb),
                "functional": functional(ca, cb),
            }
        )
        times.append((timings(_events(ca), target), timings(_events(cb), target)))

    designs = sorted({k[0] for k in keys})
    paired = set(keys)
    seeds = sorted(  # the seeds that every configuration ran in both
        s for s in {k[1] for k in keys} if all((d, s) in paired for d in designs)
    )
    if len(seeds) < MIN_SEEDS:
        warnings.append(
            f"{len(seeds)} seed(s) complete for every configuration; decision "
            f"fidelity asks for at least {MIN_SEEDS} (I1 §12)"
        )
    if len(designs) < 2:
        warnings.append("one configuration only: there is no ranking to compare")
    decisions = {}
    for metric in metrics:
        lower = metric in minimize or metric in LOWER_IS_BETTER or metric in SYSTEM
        if not seeds:
            continue
        table_a = np.array(
            [
                [score(cells_a[(d, s)], metric, series, target) for d in designs]
                for s in seeds
            ]
        )
        table_b = np.array(
            [
                [score(cells_b[(d, s)], metric, series, target) for d in designs]
                for s in seeds
            ]
        )
        result = decision(
            table_a, table_b, minimize=lower, resamples=resamples, seed=seed
        )
        for side, table in (("A", table_a), ("B", table_b)):
            if ties(table.mean(axis=0)):
                warnings.append(
                    f"every configuration ties on {metric} in {side} (a constant "
                    "predictor?): the decision on it does not inform; decide on a "
                    "system metric too"
                )
        noise = _ceiling_for(metric, table_a, lower, noise_ceiling)
        decisions[metric] = result | {
            "minimize": lower,
            "scores": {
                "a": table_a.mean(axis=0).tolist(),
                "b": table_b.mean(axis=0).tolist(),
            },
            "ceiling": noise,
            "top1_below_ceiling": None
            if noise is None
            else bool(result["top1"] < noise["top1"]),
            "regret_above_ceiling": None
            if noise is None
            else bool(result["regret"] > noise["regret"]),
            "predictor": None,  # the factorised predictor (A7) goes here
        }
    return {
        "a": str(a),
        "b": str(b),
        "backends": {
            "a": sorted({cells_a[k].backend for k in keys}),
            "b": sorted({cells_b[k].backend for k in keys}),
        },
        "configurations": [
            {
                "design_id": d,
                "scenario": cells_a[next(k for k in keys if k[0] == d)].scenario,
            }
            for d in designs
        ],
        "seeds": seeds,
        "functional": {
            "pairs": len(pairs),
            "complete": all(p["functional"]["complete"] for p in pairs),
            "model": all(p["functional"]["model"]["match"] for p in pairs),
            "protocol": all(p["functional"]["protocol"]["match"] for p in pairs),
            "bytes": all(p["functional"]["bytes"]["match"] for p in pairs),
        },
        "performance": performance(times, resamples=resamples, seed=seed),
        "decision": decisions,
        "warnings": warnings,
        "pairs": pairs,
    }


def _identity(cell: Cell) -> dict[str, Any]:
    meta = cell.meta
    return {
        "run_id": meta.get("run_id"),
        "run_hash": meta.get("run_hash"),
        "backend": cell.backend,
        "topology_id": meta.get("topology_id"),
        "infrastructure_id": meta.get("infrastructure_id"),
        "config_id": meta.get("config_id"),
    }


# --- CSV and HTML --------------------------------------------------------------------

CSV_COLUMNS = ("level", "metric", "statistic", "value", "ci_low", "ci_high", "ceiling")


def rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The result as flat rows: one per level, metric and statistic."""
    out = []
    for name in ("model", "protocol", "bytes", "complete"):
        out.append(
            {"level": "functional", "metric": name, "statistic": "match"}
            | {"value": result["functional"][name]}
        )
    for quantity, perf in result["performance"].items():
        for stat in ("relative_error", "absolute_relative_error"):
            ci = perf[f"{stat}_ci"] or [None, None]
            out.append(
                {"level": "performance", "metric": quantity, "statistic": stat}
                | {"value": perf[stat], "ci_low": ci[0], "ci_high": ci[1]}
            )
    for metric, dec in result["decision"].items():
        noise = dec["ceiling"] or {}
        ci = dec.get("ci") or {}
        for stat, ceil in (
            ("tau_b", None),
            ("top1", noise.get("top1")),
            ("p_top1", noise.get("top1")),
            ("regret", noise.get("regret")),
            ("regret_relative", noise.get("regret_relative")),
        ):
            low, high = ci.get(stat) or [None, None]
            out.append(
                {"level": "decision", "metric": metric, "statistic": stat}
                | {"value": dec[stat], "ci_low": low, "ci_high": high, "ceiling": ceil}
            )
    return out


def write_csv(result: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows(result):
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
    return path


def _cell(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return html.escape(str(value))


def _table(head: Sequence[str], body: Sequence[Sequence[Any]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    tr = "".join(
        "<tr>" + "".join(f"<td>{_cell(v)}</td>" for v in row) + "</tr>" for row in body
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def write_html(result: Mapping[str, Any], path: str | Path) -> Path:
    """The fidelity report as one static HTML page, in the style of ``report``."""
    path = Path(path)
    func = result["functional"]
    functional_rows = [
        [
            p["scenario"],
            p["seed"],
            p["functional"]["model"]["match"],
            p["functional"]["model"]["max_abs_diff"],
            p["functional"]["protocol"]["match"],
            p["functional"]["bytes"]["match"],
            p["functional"]["bytes"]["control"]["a"],
            p["functional"]["bytes"]["control"]["b"],
        ]
        for p in result["pairs"]
    ]
    perf_rows = []
    for quantity, perf in result["performance"].items():
        ci = perf["absolute_relative_error_ci"] or [None, None]
        perf_rows.append(
            [
                quantity,
                perf["pairs"],
                perf["a_mean"],
                perf["b_mean"],
                perf["relative_error"],
                perf["absolute_relative_error"],
                ci[0],
                ci[1],
            ]
        )
    dec_rows = []
    for metric, dec in result["decision"].items():
        ci = dec.get("ci") or {}
        noise = dec["ceiling"] or {}
        regret_ci = ci.get("regret") or [None, None]
        dec_rows.append(
            [
                metric,
                dec["configurations"],
                dec["seeds"],
                dec["tau_b"],
                dec["top1"],
                dec["p_top1"],
                noise.get("top1"),
                dec["regret"],
                regret_ci[0],
                regret_ci[1],
                noise.get("regret"),
                noise.get("source"),
                dec["predictor"],
            ]
        )
    run_rows = [
        [side.upper(), p[side]["run_id"], p[side]["backend"], p[side]["config_id"]]
        + [p[side]["topology_id"], p[side]["infrastructure_id"], p[side]["run_hash"]]
        for p in result["pairs"]
        for side in ("a", "b")
    ]
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in result["warnings"])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Onion-FL fidelity</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111; }}
table {{ border-collapse: collapse; margin: .5rem 0 1.5rem; font-size: .9rem; }}
th, td {{ border: 1px solid #ddd; padding: .25rem .5rem; text-align: left; }}
section {{ margin-bottom: 2rem; }}
</style></head><body>
<h1>Onion-FL fidelity</h1>
<p>B ({html.escape(", ".join(result["backends"]["b"]))}, {html.escape(result["b"])})
against A ({html.escape(", ".join(result["backends"]["a"]))},
{html.escape(result["a"])}): {func["pairs"]} pairs, seeds {result["seeds"]}.</p>
{f"<h2>Warnings</h2><ul>{warnings}</ul>" if warnings else ""}
<section><h2>Functional</h2>
<p>Complete: {_cell(func["complete"])} (model {_cell(func["model"])}, protocol
{_cell(func["protocol"])}, bytes {_cell(func["bytes"])}).</p>
{
        _table(
            [
                "scenario",
                "seed",
                "model",
                "max |Δ|",
                "protocol",
                "bytes",
                "control A",
                "control B",
            ],
            functional_rows,
        )
    }</section>
<section><h2>Performance</h2>
<p>Relative error (B − A) / A, mean per pair, then over pairs; 95 % bootstrap
interval of the absolute one.</p>
{
        _table(
            [
                "quantity",
                "pairs",
                "A mean",
                "B mean",
                "error",
                "|error|",
                "ci low",
                "ci high",
            ],
            perf_rows,
        )
    }</section>
<section><h2>Decision</h2>
<p>Rankings of the mean over seeds; intervals from a bootstrap paired by seed.
Each top-1 and regret is shown next to the noise ceiling; the factorised
predictor (A7) is not computed yet.</p>
{
        _table(
            [
                "metric",
                "K",
                "seeds",
                "τ-b",
                "top-1",
                "P(top-1)",
                "ceiling top-1",
                "regret",
                "ci low",
                "ci high",
                "ceiling regret",
                "ceiling from",
                "predictor (A7)",
            ],
            dec_rows,
        )
    }</section>
<h2>Runs</h2>
{
        _table(
            [
                "side",
                "run_id",
                "backend",
                "config_id",
                "topology_id",
                "infrastructure_id",
                "run_hash",
            ],
            run_rows,
        )
    }
</body></html>
"""
    path.write_text(page, encoding="utf-8")
    return path
