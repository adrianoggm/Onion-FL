"""Do the contribution scores rank the noisy edges below the honest ones? (#230)

Reads the run folders of ``experiments/contribution_noise.yaml``: the noisy
edges from the ``data.attack`` event, the scores from the fogs'
``diagnostic.contribution_*`` events. Per run and score, the AUC is the
probability that an honest edge scores above a noisy one, over the pairs in the
same fog and round (ties count 1/2); 0.5 is chance, 1 a perfect ranking. Each
edge's score in a round is its mean over the groups it sends. The runs are then
combined over the seeds with a 95% t interval.

    python scripts/contribution_auc.py runs/ --out results/contribution_noise
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCORES = ("contribution_shapley", "contribution_loo")


def auc(honest: list[float], noisy: list[float]) -> tuple[float, int]:
    """Sum of wins (ties 1/2) and number of pairs."""
    h, n = np.asarray(honest)[:, None], np.asarray(noisy)[None, :]
    return float((h > n).sum() + 0.5 * (h == n).sum()), h.size * n.size


def read_run(path: Path) -> dict | None:
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    noisy: set[str] = set()
    # (score, group) -> (fog, round, child) -> values
    scores: dict[tuple[str, str], dict] = defaultdict(lambda: defaultdict(list))
    with (path / "events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            name = event["name"]
            if name == "data.attack":
                noisy = set(event["tags"]["edges"])
            elif event["level"] == "fog" and name[len("diagnostic.") :] in SCORES:
                tags = event["tags"]
                key = (event["node"], event["round"], tags["child"])
                for group in (tags["group"], "*"):
                    scores[(name[len("diagnostic.") :], group)][key].append(
                        event["value"]
                    )
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    out = {
        "run_id": meta["run_id"],
        "run_hash": meta["run_hash"],
        "config_id": meta["config_id"],
        "commit": meta["code_version"]["commit"][:7],
        "dirty": meta["code_version"]["dirty"],
        "scenario": meta["scenario"],
        "seed": meta["seed"],
        "noisy_edges": len(noisy),
        "macro_f1": summary["final"]["global/*"]["macro_f1"],
    }
    if not noisy:
        return out
    for (score, group), by_key in scores.items():
        if group not in ("*", "trunk"):
            continue
        cells: dict[tuple, dict[bool, list[float]]] = defaultdict(
            lambda: {True: [], False: []}
        )
        for (fog, round_, child), values in by_key.items():
            cells[(fog, round_)][child in noisy].append(float(np.mean(values)))
        wins = pairs = 0.0
        means = {True: [], False: []}
        for cell in cells.values():
            means[True] += cell[True]
            means[False] += cell[False]
            if cell[True] and cell[False]:
                w, p = auc(cell[False], cell[True])
                wins, pairs = wins + w, pairs + p
        label = (
            f"{score.removeprefix('contribution_')}_{'all' if group == '*' else group}"
        )
        out[f"auc_{label}"] = wins / pairs if pairs else float("nan")
        out[f"honest_{label}"] = float(np.mean(means[False]))
        out[f"noisy_{label}"] = float(np.mean(means[True]))
    return out


def interval(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan")
    half = (
        stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, float(half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [
        r
        for path in sorted(args.runs.iterdir())
        if (path / "events.jsonl").exists() and (r := read_run(path))
    ]
    rows.sort(key=lambda r: (r["scenario"], r["seed"]))
    args.out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for scenario in sorted({r["scenario"] for r in rows}):
        group = [r for r in rows if r["scenario"] == scenario]
        for key in sorted({k for r in group for k in r}):
            if key.startswith(("auc_", "honest_", "noisy_", "macro_f1")):
                mean, half = interval([r[key] for r in group])
                summary.append(
                    {
                        "scenario": scenario,
                        "statistic": key,
                        "seeds": len(group),
                        "mean": round(mean, 4),
                        "ci95": round(half, 4),
                    }
                )
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
