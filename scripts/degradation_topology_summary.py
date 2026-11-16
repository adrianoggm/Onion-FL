from __future__ import annotations

"""Per-run metrics and paired contrasts of the degradation x topology study (#290).

    python scripts/degradation_topology_summary.py runs <runs folder> \\
        --experiment degradation_topology_wesad --target 0.7 --out runs.csv
    python scripts/degradation_topology_summary.py contrasts runs.csv --out contrasts.json

``runs`` writes one row per finished run: its identity, its arm (topology and
degradation level), the primary metrics of the pre-registration
(docs/preregistration/degradation_topology.md §4: final global macro-F1, time
to target) and the secondary ones (accuracy, the root's lost rounds, every
aggregator's failed quorums, WAN bytes, per-zone received macro-F1 and run
duration).

``contrasts`` pairs the runs by seed and gives, per degradation level and
metric, H1 (flat − two and three − two, Holm over the two) and H2 (the
difference-in-differences against nominal, Holm over the 16 of a metric), with
``onion_fl.observability.stats``. For H1 it adds ``seeds_needed`` for the
metric's margin from each contrast's ``sd_diff``: the pilot's seed rule (§6).
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from onion_fl.observability.events import read_events
from onion_fl.observability.stats import (
    holm,
    paired_power,
    paired_test,
    seeds_needed,
    verdict,
)

REFERENCE = "two"
TOPOLOGIES = ("flat", "two", "three")
LEVELS = (
    "nominal",
    "loss5",
    "loss20",
    "4g",
    "dropout",
    "dropout_k1",
    "crash",
    "partition",
    "partition_autonomy",
)
# metric -> (margin, log scale); the margins of the pre-registration, §4
MARGINS = {
    "macro_f1": (0.02, False),
    "accuracy": (0.02, False),
    "time_to_target_s": (math.log(1.1), True),
    "duration_s": (math.log(1.1), True),
    "wan_bytes": (math.log(1.1), True),
    "root_lost_rounds": (1.0, False),
    "zone_gap": (0.02, False),
}
PRIMARY = ("macro_f1", "time_to_target_s")


def arm(config: dict) -> tuple[str, str]:
    """(topology, degradation level) of a run's config."""
    topology = config["topology"]
    name = {"dt_flat": "flat", "dt_two_levels": "two", "dt_three_levels": "three"}
    infra = config["infrastructure"]["name"].removeprefix("degradation_lab")
    level = infra.removesuffix("_cifar10").strip("_") or "nominal"
    defaults = topology["fog"]["defaults"]
    if topology["edge"].get("availability"):
        level = "dropout_k1" if defaults.get("quorum") == 1.0 else "dropout"
    if level == "partition" and defaults.get("autonomy"):
        level = "partition_autonomy"
    return name[topology["name"]], level


def zones_of(config: dict) -> dict[str, str]:
    """Node -> 'dc', 'a' or 'b': the side of the WAN its host is on."""
    assignment = config["infrastructure"]["assignment"]
    hosts = {config["topology"]["root"]["id"]: assignment["root"]}
    hosts |= assignment["aggregators"] | assignment["edges"]["hosts"]
    out = {}
    for node, host in hosts.items():
        side = re.match(r"(?:field_|gw-|fog-)([ab])", host)
        out[node] = side.group(1) if side else "dc"
    return out


def wan_links(src: str, dst: str) -> int:
    """Zone-cloud links a message crosses: 0 inside a side, 1 to or from the
    cloud, 2 between the two zones (through the cloud's site)."""
    if src == dst:
        return 0
    return 1 if "dc" in (src, dst) else 2


def metrics(path: Path, target: float) -> dict:
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    events = read_events(path / "events.jsonl")
    config = meta["config"]
    topology, level = arm(config)
    zone = zones_of(config)
    root = config["topology"]["root"]["id"]
    for e in events:  # evaluators sit on their parent's host
        if e.get("role") == "evaluator" and e["name"] == "message.sent":
            zone.setdefault(e["node"], zone.get(e["tags"].get("dst"), "dc"))
    scores: dict[int, list[tuple[float, float]]] = defaultdict(list)
    edges: dict[str, float] = {}
    end, wan, lost, failed = None, 0.0, 0, 0
    last_edge_round = max(
        (e["round"] or 0 for e in events if e["name"] == "eval.macro_f1"), default=0
    )
    for e in events:
        name, tags = e["name"], e.get("tags") or {}
        if name == "message.sent" and tags.get("dst") is not None:
            src, dst = zone.get(e["node"]), zone.get(tags["dst"])
            if src is not None and dst is not None:
                wan += wan_links(src, dst) * (e["value"] or 0.0)
        elif name == "round.quorum_failed":
            failed += 1
            lost += e["node"] == root
        elif name == "run.finished" and e["node"] == root:
            end = e["t_virtual"]
        elif name == "eval.macro_f1" and tags.get("model") == "global":
            if "dataset" not in tags and e["node"] == root:
                scores[e["round"]].append((e["t_virtual"], e["value"]))
        if (
            name == "eval.macro_f1"
            and e.get("role") == "edge"
            and tags.get("model") == "received"
            and e["round"] == last_edge_round
        ):
            edges[e["node"]] = e["value"]
    end = float(summary["t_virtual_end"] if end is None else end)
    reached, hit = end, False
    for rnd in sorted(r for r in scores if r is not None):
        if np.mean([v for _, v in scores[rnd]]) >= target:
            reached, hit = max(t for t, _ in scores[rnd]), True
            break
    per_zone = {
        z: [v for node, v in edges.items() if zone.get(node) == z] for z in ("a", "b")
    }
    zone_f1 = {z: float(np.mean(v)) if v else None for z, v in per_zone.items()}
    final = summary["final"].get("global/*")
    rescored = final is None
    if rescored:  # no root round ever closed: the coordinator holds the init
        final = rescore(path, meta)
    gap = (
        abs(zone_f1["a"] - zone_f1["b"])
        if None not in (zone_f1["a"], zone_f1["b"])
        else None
    )
    code = meta.get("code_version") or {}
    return {
        "run_id": meta["run_id"],
        "topology": topology,
        "level": level,
        "seed": meta["seed"],
        "macro_f1": final["macro_f1"],
        "accuracy": final["accuracy"],
        "rescored": rescored,
        "time_to_target_s": reached,
        "target_reached": hit,
        "duration_s": end,
        "root_lost_rounds": lost,
        "quorum_failed_all": failed,
        "wan_bytes": wan,
        "zone_a_macro_f1": zone_f1["a"],
        "zone_b_macro_f1": zone_f1["b"],
        "zone_gap": gap,
        "reparented": sum(e["name"] == "edge.reparented" for e in events),
        "autonomous_rounds": sum(e["name"] == "fog.autonomous.round" for e in events),
        "wall_seconds": summary.get("wall_seconds"),
        "topology_id": meta["topology_id"],
        "infrastructure_id": meta.get("infrastructure_id"),
        "config_id": meta["config_id"],
        "data_id": meta.get("data_id"),
        "run_hash": meta["run_hash"],
        "commit": code.get("commit"),
        "dirty": code.get("dirty"),
    }


def rescore(path: Path, meta: dict) -> dict[str, float]:
    """The final global metrics of a run in which no root round closed (flat
    under ``dropout_k1``), so no global evaluation was ever emitted: the run's
    ``model.npz`` (the initial model, never replaced) scored on the test
    subjects the coordinator's evaluators hold, per subject and then weighted
    by rows, as ``global/*`` is. On runs that have ``global/*`` this gives the
    same values exactly."""
    from onion_fl.experiment.config import parse_experiment
    from onion_fl.experiment.runner import _scenario_data
    from onion_fl.experiment.sweep import Scenario
    from onion_fl.learning.metrics import compute, predict
    from onion_fl.learning.model import load_arrays, models
    from onion_fl.roles.policies import create

    config = parse_experiment(meta["config"])
    scenario = Scenario(meta["scenario"], meta["seed"], config, meta["config_id"])
    _, split, _, _ = _scenario_data(scenario, parent=None)
    with np.load(path / "model.npz") as npz:
        state = {k: npz[k] for k in npz.files}
    model = create(models, config.learning.model).build(
        [split.test[0].shape], seed=meta["seed"]
    )
    load_arrays(model, state)
    totals, rows = defaultdict(float), 0
    for subject in split.test:
        y = np.asarray(subject.y, np.int64)
        scores = compute(
            predict(model, subject.X), y, subject.n_classes, ["macro_f1", "accuracy"]
        )
        for name, value in scores.items():
            totals[name] += value * len(y)
        rows += len(y)
    return {name: value / rows for name, value in totals.items()}


def _value(row: dict, metric: str) -> float | None:
    raw = row.get(metric)
    if raw in (None, ""):
        return None
    value = float(raw)
    return math.log(value) if MARGINS[metric][1] and value > 0 else value


def _table(rows: list[dict], metric: str) -> dict[tuple[str, str], dict[int, float]]:
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = _value(row, metric)
        if value is not None:
            out[(row["topology"], row["level"])][int(row["seed"])] = value
    return out


def _paired(a: dict[int, float], b: dict[int, float]) -> np.ndarray:
    seeds = sorted(a.keys() & b.keys())
    return np.array([a[s] - b[s] for s in seeds])


def _test(d: np.ndarray, margin: float) -> dict:
    out = paired_test(d, margin)
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in out.items()
    }


def contrasts(rows: list[dict], metrics_: list[str]) -> dict:
    h1, h2 = [], []
    for metric in metrics_:
        margin, log = MARGINS[metric]
        table = _table(rows, metric)
        for level in LEVELS:
            family = []
            for topology in TOPOLOGIES:
                if topology == REFERENCE:
                    continue
                d = _paired(table[(topology, level)], table[(REFERENCE, level)])
                if len(d) < 2:
                    continue
                stats = _test(d, margin)
                sd = stats["sd_diff"]
                if sd is not None and sd < 1e-12:  # no spread but float noise
                    stats["seeds_needed"] = 2
                else:
                    stats["seeds_needed"] = (
                        None if sd is None else seeds_needed(margin, sd)
                    )
                family.append(
                    {
                        "metric": metric,
                        "log": log,
                        "margin": margin,
                        "level": level,
                        "contrast": f"{topology} - {REFERENCE}",
                        **stats,
                    }
                )
            _adjust(family)
            h1 += family
        family = []
        for level in LEVELS[1:]:
            for topology in TOPOLOGIES:
                if topology == REFERENCE:
                    continue
                seeds = sorted(
                    table[(topology, level)].keys()
                    & table[(topology, "nominal")].keys()
                    & table[(REFERENCE, level)].keys()
                    & table[(REFERENCE, "nominal")].keys()
                )
                if len(seeds) < 2:
                    continue
                d = np.array(
                    [
                        table[(topology, level)][s]
                        - table[(topology, "nominal")][s]
                        - table[(REFERENCE, level)][s]
                        + table[(REFERENCE, "nominal")][s]
                        for s in seeds
                    ]
                )
                family.append(
                    {
                        "metric": metric,
                        "log": log,
                        "margin": margin,
                        "level": level,
                        "contrast": f"DiD {topology} - {REFERENCE}",
                        **_test(d, margin),
                    }
                )
        _adjust(family)
        h2 += family
    return {"h1": h1, "h2": h2}


def _adjust(family: list[dict]) -> None:
    adjusted = holm([np.nan if f["p_perm"] is None else f["p_perm"] for f in family])
    for f, p in zip(family, adjusted, strict=True):
        f["p_adj"] = None if math.isnan(p) else p
        f["verdict"] = verdict(
            1.0 if f["p_adj"] is None else f["p_adj"],
            1.0 if f["p_tost"] is None else f["p_tost"],
        )


def seed_rule(h1: list[dict], low: int = 10, high: int = 30) -> dict:
    """§6: the largest seeds_needed over the primary H1 contrasts, clamped."""
    needed = [
        (c["seeds_needed"], c["metric"], c["level"], c["contrast"])
        for c in h1
        if c["metric"] in PRIMARY and c["seeds_needed"] is not None
    ]
    worst = max(needed) if needed else (None, None, None, None)
    n = min(max(worst[0] or low, low), high)
    under = [
        {
            "metric": c["metric"],
            "level": c["level"],
            "contrast": c["contrast"],
            "power": paired_power(n, c["margin"], c["sd_diff"]),
        }
        for c in h1
        if c["metric"] in PRIMARY
        and c["sd_diff"]
        and paired_power(n, c["margin"], c["sd_diff"]) < 0.8
    ]
    return {"largest": worst[0], "at": worst[1:], "n": n, "underpowered": under}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    runs = sub.add_parser("runs")
    runs.add_argument("folder")
    runs.add_argument("--experiment", required=True)
    runs.add_argument("--target", type=float, required=True)
    runs.add_argument("--out", required=True)
    pairs = sub.add_parser("contrasts")
    pairs.add_argument("csv")
    pairs.add_argument("--out", required=True)
    pairs.add_argument("--metric", action="append", default=None)
    args = parser.parse_args()
    if args.command == "runs":
        rows = []
        for run_json in sorted(Path(args.folder).glob("*/run.json")):
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            if meta["config"].get("name") != args.experiment:
                continue
            if meta.get("status") != "finished":
                print(f"{run_json.parent.name}: {meta.get('status')}, left out")
                continue
            rows.append(metrics(run_json.parent, args.target))
        rows.sort(
            key=lambda r: (
                LEVELS.index(r["level"]),
                TOPOLOGIES.index(r["topology"]),
                r["seed"],
            )
        )
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} runs -> {args.out}")
        return
    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = contrasts(rows, args.metric or list(MARGINS))
    out["seed_rule"] = seed_rule(out["h1"])
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["seed_rule"], indent=2))


if __name__ == "__main__":
    main()
