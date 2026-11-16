from __future__ import annotations

"""Paired contrasts of the technique comparisons against FedAvg (#293).

    python scripts/paired_techniques.py <runs folder>... --out <folder>

For each experiment of ``EXPERIMENTS`` it reads the final scores of every
finished run (``summary.json``), names each run's arm, pairs every arm with the
reference arm on the runs that share a seed, and gives, per series (a column
of the README tables), the paired statistics of ``onion_fl.observability.stats``
with Holm over the arms of that series. It writes ``run_metrics.csv`` (one row
per run and series, with the run's identity) and ``contrasts.csv``.

A personal model exists only for Ditto and APFL: their ``personal/*`` is paired
with FedAvg's ``local/*``, the model each FedAvg edge trained, as the README
table compares them.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from onion_fl.observability.stats import holm, paired_test, verdict

MARGIN = 0.02  # macro-F1, the margin of the degradation study (#290)
METRIC = "macro_f1"


def drift(config: dict) -> str:
    return config["learning"]["trainer"]["name"]


def server(config: dict) -> str:
    name = config["learning"]["server_optimizer"]
    name = name if isinstance(name, str) else name["name"]
    topology = config["topology"]
    topology = topology if isinstance(topology, str) else topology["name"]
    buffered = topology.endswith("fedbuff")
    return f"{name}+fedbuff" if buffered else name


def personalisation(config: dict) -> str:
    sharing = config["learning"]["sharing"]
    sharing = sharing if isinstance(sharing, str) else sharing["name"]
    trainer = config["learning"]["trainer"]["name"]
    return sharing if trainer == "standard" else trainer


def placement(config: dict) -> str:
    return f"alpha={float(config['data']['placement']['alpha']):g}"


# experiment -> (arm namer, reference arm, series compared)
EXPERIMENTS = {
    "techniques_drift": (
        drift,
        "standard",
        ("global/swell", "global/wesad", "local/*"),
    ),
    "techniques_server": (server, "replace", ("global/swell", "global/wesad")),
    "techniques_personalisation": (
        personalisation,
        "fedavg",
        ("global/swell", "global/wesad", "local/*", "personal/*"),
    ),
    "mix_swell_wesad": (
        placement,
        "alpha=0.5",
        ("global/swell", "global/wesad", "global/*"),
    ),
}


def read(folders: list[Path]) -> list[dict]:
    rows = []
    for path in sorted(p for runs in folders for p in runs.glob("*/run.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        name = meta["config"]["name"]
        if name not in EXPERIMENTS:
            continue
        summary = json.loads((path.parent / "summary.json").read_text("utf-8"))
        namer = EXPERIMENTS[name][0]
        code = meta.get("code_version") or {}
        for series, scores in summary["final"].items():
            rows.append(
                {
                    "experiment": name,
                    "arm": namer(meta["config"]),
                    "seed": meta["seed"],
                    "series": series,
                    "round": scores.get("round"),
                    METRIC: scores.get(METRIC),
                    "run_id": meta["run_id"],
                    "run_hash": meta["run_hash"],
                    "config_id": meta["config_id"],
                    "topology_id": meta["topology_id"],
                    "data_id": meta.get("data_id"),
                    "commit": code.get("commit"),
                    "dirty": code.get("dirty"),
                }
            )
    return rows


def contrasts(rows: list[dict]) -> list[dict]:
    # experiment -> series -> arm -> seed -> score
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        table[r["experiment"]][r["series"]][r["arm"]][r["seed"]] = r[METRIC]
    out = []
    for name, (_, reference, series_list) in EXPERIMENTS.items():
        for series in series_list:
            arms = table[name].get(series, {})
            # personal models are compared with FedAvg's trained one
            base = table[name].get("local/*" if series == "personal/*" else series)
            ref = (base or {}).get(reference, {})
            if not ref:
                continue
            block = []
            for arm in sorted(arms):
                if arm == reference and series != "personal/*":
                    continue
                seeds = sorted(set(arms[arm]) & set(ref))
                d = np.array([arms[arm][s] - ref[s] for s in seeds], float)
                stats = paired_test(d, margin=MARGIN)
                means = (
                    float(np.mean([arms[arm][s] for s in seeds])),
                    float(np.mean([ref[s] for s in seeds])),
                )
                block.append(
                    {
                        "experiment": name,
                        "series": series,
                        "arm": arm,
                        "reference": reference,
                        "mean": means[0],
                        "reference_mean": means[1],
                    }
                    | stats
                )
            for row, p in zip(block, holm([b["p_perm"] for b in block]), strict=True):
                row["p_adj"] = p
                row["verdict"] = verdict(p, row["p_tost"])
            out.extend(block)
    return out


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired contrasts against FedAvg")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    write(rows, args.out / "run_metrics.csv")
    write(contrasts(rows), args.out / "contrasts.csv")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
