from __future__ import annotations

"""Per-run metrics of the architectures experiment (issue #192).

    python scripts/architectures_summary.py <runs folder> --out <run_metrics.csv>

Reads every finished run of ``infrastructure_architectures`` (or of the
experiment given with ``--experiment``) and writes one row per run with its
identity, its learning scores (the final global macro-F1 per dataset on the
test subjects, and the edges' at the last edge evaluation) and its system
metrics on the backend's clock: the mean and maximum time of a round at the
root, the time until the global macro-F1 first reaches ``--target`` (censored
at the run's end when it never does), and the bytes per level of the
federation. ``wan`` is every message to or from the cloud: on these
infrastructures the cloud is alone in its site, so each of them crosses a
fog-cloud link. It also prints the mean and 95 % interval per architecture.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from onion_fl.observability.events import read_events

ARCHITECTURES = {  # (topology, infrastructure, edge mix, failover) -> label
    ("cloud_swell_wesad", "architectures_lab", "homogeneous", "none"): "A",
    ("four_fogs_swell_wesad_physical", "architectures_lab", "homogeneous", "none"): "B",
    ("four_fogs_swell_wesad_physical", "architectures_lab", "mixed", "none"): "C",
    (
        "four_fogs_swell_wesad_physical",
        "architectures_lab_dynamic",
        "mixed",
        "nearest",
    ): "D",
}


def architecture(config: dict) -> str:
    topology = config["topology"]
    topology = topology if isinstance(topology, str) else topology.get("name", "")
    infra = config.get("infrastructure")
    if infra is None:  # the logical federation alone, for the local backend
        return {"cloud_swell_wesad": "A", "four_fogs_swell_wesad_physical": "B"}.get(
            topology, "?"
        ) + "-logical"
    mix = infra["assignment"]["edges"].get("mix", {})
    failover = infra.get("failover") or "none"
    name = failover if isinstance(failover, str) else failover.get("name", "")
    key = (
        topology,
        infra["name"],
        "homogeneous" if list(mix) == ["raspberry_pi"] else "mixed",
        "nearest" if name == "nearest_alive" else name,
    )
    return ARCHITECTURES.get(key, "?")


def level(src: str, dst: str, edges: set[str], cloud: str) -> str:
    if cloud in (src, dst):
        other = dst if src == cloud else src
        return "edge-cloud" if other in edges else "fog-cloud"
    return "edge-fog"


def metrics(path: Path, target: float) -> dict:
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    events = read_events(path / "events.jsonl")
    cloud = next(e["node"] for e in events if e.get("role") == "coordinator")
    edges = {e["node"] for e in events if e.get("role") == "edge"}
    started, closed, scores = {}, {}, defaultdict(list)
    sent: dict[str, float] = defaultdict(float)
    for e in events:
        name, tags = e["name"], e.get("tags") or {}
        if e.get("role") == "coordinator":
            if name == "round.started":
                started.setdefault(tags.get("round"), e["t_virtual"])
            elif name == "round.closed":
                closed.setdefault(tags.get("round"), e["t_virtual"])
            elif name == "eval.macro_f1" and tags.get("model", "global") == "global":
                scores[e.get("round")].append((e["t_virtual"], e["value"]))
        if name == "message.sent" and tags.get("dst") is not None:
            sent[level(e["node"], tags["dst"], edges, cloud)] += e["value"] or 0.0
    rounds = [closed[r] - started[r] for r in closed if r in started]
    end = float(summary["t_virtual_end"])
    reached, hit = end, False
    for rnd in sorted(r for r in scores if r is not None):
        if np.mean([v for _, v in scores[rnd]]) >= target:
            reached, hit = max(t for t, _ in scores[rnd]), True
            break
    final = summary["final"]
    return {
        "run_id": meta["run_id"],
        "architecture": architecture(meta["config"]),
        "backend": meta.get("backend", "sim"),
        "seed": meta["seed"],
        "global_swell_macro_f1": final.get("global/swell", {}).get("macro_f1"),
        "global_wesad_macro_f1": final.get("global/wesad", {}).get("macro_f1"),
        "edge_received_macro_f1": final.get("received/*", {}).get("macro_f1"),
        "edge_local_macro_f1": final.get("local/*", {}).get("macro_f1"),
        "quorum_failed": summary.get("quorum_failed", 0),
        "reparented": sum(e["name"] == "edge.reparented" for e in events),
        "round_time_mean_s": float(np.mean(rounds)) if rounds else None,
        "round_time_max_s": float(np.max(rounds)) if rounds else None,
        "time_to_target_s": reached,
        "target_reached": hit,
        "duration_s": end,
        "bytes_total": summary["messages"]["bytes"],
        **{
            f"bytes_{k}": sent.get(k, 0.0)
            for k in ("edge-fog", "fog-cloud", "edge-cloud")
        },
        "bytes_wan": sent.get("fog-cloud", 0.0) + sent.get("edge-cloud", 0.0),
        "topology_id": meta["topology_id"],
        "infrastructure_id": meta.get("infrastructure_id"),
        "config_id": meta["config_id"],
        "run_hash": meta["run_hash"],
        "commit": (meta.get("code") or {}).get("commit"),
        "dirty": (meta.get("code") or {}).get("dirty"),
    }


def ci(values: list[float]) -> tuple[float, float]:
    values = np.asarray([v for v in values if v is not None], float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), 0.0
    half = (
        stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return float(values.mean()), float(half)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--experiment", default="infrastructure_architectures")
    parser.add_argument("--target", type=float, default=0.6)
    args = parser.parse_args()
    rows = []
    for run_json in sorted(Path(args.runs).glob("*/run.json")):
        meta = json.loads(run_json.read_text(encoding="utf-8"))
        if meta["config"].get("name") != args.experiment:
            continue
        if meta.get("status") != "finished":
            print(f"{run_json.parent.name}: {meta.get('status')}, left out")
            continue
        rows.append(metrics(run_json.parent, args.target))
    rows.sort(key=lambda r: (r["architecture"], r["backend"], r["seed"]))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["backend"])].append(row)
    columns = [
        "global_swell_macro_f1",
        "global_wesad_macro_f1",
        "edge_received_macro_f1",
        "edge_local_macro_f1",
        "round_time_mean_s",
        "round_time_max_s",
        "time_to_target_s",
        "duration_s",
        "bytes_edge-fog",
        "bytes_fog-cloud",
        "bytes_edge-cloud",
        "bytes_wan",
        "quorum_failed",
        "reparented",
    ]
    for (arch, backend), group in sorted(groups.items()):
        reached = sum(r["target_reached"] for r in group)
        print(
            f"{arch} [{backend}] n={len(group)} reached target {reached}/{len(group)}"
        )
        for column in columns:
            mean, half = ci([r[column] for r in group])
            print(f"  {column}: {mean:.4g} ± {half:.2g}")


if __name__ == "__main__":
    main()
