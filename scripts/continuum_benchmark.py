"""The continual benchmark's metrics (issue #163, spec §12), from run folders.

    python scripts/continuum_benchmark.py metrics runs/<run_id> ... \
        --subjects test --curves curves.csv > run_metrics.csv
    python scripts/continuum_benchmark.py table run_metrics.csv --by scenario

``metrics`` writes one row per run and dataset:

- ``prequential_*``: every arrival scored by the model its edge was serving,
  pooled over the run from the edges' confusion counts; ``*_labelled`` keeps
  only the rows whose label arrived.
- ``final_*``: the final ``model.npz``, scored offline on every row of the
  held-out subjects (``--subjects``), as ``score_final_models.py`` does.
- ``heldout_pooled_*``: the root's per-round evaluations on the held-out
  subjects, pooled. With ``evaluation.global.subjects: val`` these are the
  validation streams, each round scoring what arrived since the last: the
  selection's criterion.
- ``auc``: the area under the root's per-round macro-F1 on the held-out
  subjects over data time, divided by the time it spans. Each round's model
  counts from when its round opened until the next one did.
- ``forgetting``: the best per-round macro-F1 of an earlier round minus the
  last round's, on the same subjects (never below 0 in practice: it is a drop).
- ``drift_round``, ``recovery_rounds``, ``recovery_minutes``: the first round
  whose arrivals' majority class (the hidden truth, simulation only) differs
  from the previous round's; the level is the pooled prequential accuracy of
  the rounds before it, and recovery is the rounds (and data minutes) until a
  round's prequential accuracy is back within ``--tolerance`` of that level.
  Empty when no round precedes the shift, or when it never recovers
  (``recovered`` says which).
- costs: ``rounds`` opened, ``trained_rounds`` (not idle at the root),
  ``bytes`` sent, ``trainings`` that worked and ``sample_epochs``: the rows
  each training went through (new, replayed or retrained) times its local
  epochs. The unlabelled pool's passes are not counted.
- process: ``promoted``, ``rejected`` and ``rollbacks`` of the challengers.

``table`` prints the mean ± 95% CI (Student t) over seeds of the chosen
columns, grouped by ``--by``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_final_models import score  # noqa: E402

DEFAULT_COLUMNS = [
    "prequential_macro_f1",
    "auc",
    "final_macro_f1",
    "forgetting",
    "recovery_rounds",
    "rounds",
    "bytes",
    "sample_epochs",
]


def macro_f1(confusion: np.ndarray) -> float:
    """As ``learning.metrics``: the mean F1 over classes, 0 where undefined."""
    tp = np.diag(confusion).astype(float)
    predicted, actual = confusion.sum(0), confusion.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(predicted + actual > 0, 2 * tp / (predicted + actual), 0.0)
    return float(f1.mean())


def _confusion(counts: dict[str, float], n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    for key, value in counts.items():
        i, j = map(int, key.split("."))
        matrix[i, j] += value
    return matrix


def _events(run: Path) -> list[dict]:
    with open(run / "events.jsonl", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


def measure(run: Path, subjects: str, tolerance: float) -> tuple[list, list]:
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    epochs = meta["config"]["learning"]["trainer"]
    epochs = epochs.get("local_epochs", 1) if isinstance(epochs, dict) else 1
    events = _events(run)
    # pooled and per-round confusion counts of the edges' stored predictions
    pooled: dict = defaultdict(lambda: defaultdict(float))
    per_round: dict = defaultdict(lambda: defaultdict(float))
    rounds_at: dict[int, float] = {}
    test: dict = defaultdict(dict)  # dataset -> round -> macro-F1 at the root
    heldout: dict = defaultdict(lambda: defaultdict(float))  # pooled over rounds
    trained: dict[tuple, int] = {}
    replayed: dict[tuple, int] = {}
    counts = defaultdict(int)
    idle_at_root = 0
    for e in events:
        name, tags, role = e["name"], e["tags"], e["role"]
        if role == "edge" and name.startswith("eval.confusion."):
            key = name[len("eval.confusion.") :]
            model, dataset = tags.get("model"), tags.get("dataset")
            if model in ("prequential", "prequential_labelled"):
                pooled[model, dataset][key] += e["value"]
                if model == "prequential":
                    per_round[dataset, tags["round"]][key] += e["value"]
        elif role == "coordinator" and name == "round.started":
            rounds_at[tags["round"]] = tags["at"]
        elif role == "coordinator" and tags.get("model") == "global":
            if "dataset" not in tags:
                continue
            if name == "eval.macro_f1":
                test[tags["dataset"]][tags["round"]] = e["value"]
            elif name.startswith("eval.confusion."):
                key = name[len("eval.confusion.") :]
                heldout[tags["dataset"]][key] += e["value"]
        elif role == "coordinator" and name == "round.idle":
            idle_at_root += 1
        elif name == "edge.trained":
            trained[e["node"], tags["round"]] = tags["examples"]
        elif name == "diagnostic.memory":
            replayed[e["node"], tags["round"]] = tags.get("replayed", 0)
        elif name in ("version.promoted", "version.rejected", "version.rollback"):
            counts[name] += 1
    finals = {r["dataset"]: r for r in score(run, subjects)}
    n_classes = {
        d: round(sum(k.startswith("confusion.") for k in r) ** 0.5)
        for d, r in finals.items()
    }
    common = {
        "run_id": meta["run_id"],
        "scenario": meta["scenario"],
        "seed": meta["seed"],
        "config_id": meta["config_id"],
        "topology_id": meta["topology_id"],
        "run_hash": meta.get("run_hash"),
        "rounds": summary["rounds"],
        "trained_rounds": summary["rounds"] - idle_at_root,
        "bytes": summary["messages"]["bytes"],
        "trainings": len(trained),
        "sample_epochs": epochs
        * sum(n + replayed.get(key, 0) for key, n in trained.items()),
        "promoted": counts["version.promoted"],
        "rejected": counts["version.rejected"],
        "rollbacks": counts["version.rollback"],
    }
    rows, curves = [], []
    for dataset in sorted(finals):
        n = n_classes[dataset]
        row = dict(common, dataset=dataset, subjects=subjects)
        for model in ("prequential", "prequential_labelled"):
            confusion = _confusion(pooled[model, dataset], n)
            row[f"{model}_macro_f1"] = round(macro_f1(confusion), 6)
            row[f"{model}_samples"] = int(confusion.sum())
        confusion = _confusion(heldout[dataset], n)
        row["heldout_pooled_macro_f1"] = round(macro_f1(confusion), 6)
        row["heldout_pooled_samples"] = int(confusion.sum())
        final = finals[dataset]
        row["final_macro_f1"] = final["macro_f1"]
        row["final_loss"] = final["loss"]
        row["final_confusion"] = json.dumps(
            _confusion(
                {k[len("confusion.") :]: v for k, v in final.items() if "." in k}, n
            )
            .astype(int)
            .tolist()
        )
        # the per-round curve on the held-out subjects, over data time
        series = test[dataset]
        ordered = sorted(series)
        spans = [
            rounds_at[b] - rounds_at[a]
            for a, b in zip(ordered, ordered[1:], strict=False)
        ]
        row["auc"] = (
            round(
                sum(series[r] * s for r, s in zip(ordered, spans, strict=False))
                / sum(spans),
                6,
            )
            if spans and sum(spans) > 0
            else None
        )
        row["forgetting"] = (
            round(max(series[r] for r in ordered[:-1]) - series[ordered[-1]], 6)
            if len(ordered) > 1
            else None
        )
        row["last_eval_round"] = ordered[-1] if ordered else None
        # recovery after the first change of the arrivals' majority class
        arrival_rounds = sorted(r for d, r in per_round if d == dataset)
        matrices = {r: _confusion(per_round[dataset, r], n) for r in arrival_rounds}
        majority = {r: int(np.argmax(m.sum(1))) for r, m in matrices.items()}
        accuracy = {r: float(np.trace(m) / m.sum()) for r, m in matrices.items()}
        drift = next(
            (
                b
                for a, b in zip(arrival_rounds, arrival_rounds[1:], strict=False)
                if majority[a] != majority[b]
            ),
            None,
        )
        row |= {"drift_round": drift, "recovered": None}
        row |= {"recovery_rounds": None, "recovery_minutes": None}
        if drift is not None:
            before = sum((matrices[r] for r in arrival_rounds if r < drift), 0)
            level = float(np.trace(before) / before.sum())
            back = next(
                (
                    r
                    for r in arrival_rounds
                    if r >= drift and accuracy[r] >= level - tolerance
                ),
                None,
            )
            row["recovered"] = back is not None
            if back is not None:
                row["recovery_rounds"] = back - drift
                minutes = (rounds_at[back] - rounds_at[drift]) / 60
                row["recovery_minutes"] = round(minutes, 3)
        rows.append(row)
        for r in sorted(set(arrival_rounds) | set(ordered)):
            curves.append(
                {
                    "run_id": meta["run_id"],
                    "scenario": meta["scenario"],
                    "seed": meta["seed"],
                    "dataset": dataset,
                    "round": r,
                    "at_minutes": round(rounds_at.get(r, np.nan) / 60, 3),
                    "arrivals": int(matrices[r].sum()) if r in matrices else 0,
                    "prequential_accuracy": (
                        round(accuracy[r], 6) if r in accuracy else None
                    ),
                    "heldout_macro_f1": (round(series[r], 6) if r in series else None),
                }
            )
    return rows, curves


def _write(rows: list[dict], out) -> None:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def table(path: Path, by: list[str], columns: list[str]) -> str:
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in by)].append(row)
    lines = [
        "| " + " | ".join(by + columns) + " |",
        "|" + "---|" * (len(by) + len(columns)),
    ]
    for key, members in groups.items():
        cells = list(key)
        for column in columns:
            values = np.array(
                [float(m[column]) for m in members if m[column] not in ("", "None")]
            )
            if not len(values):
                cells.append("—")
                continue
            mean, n = values.mean(), len(values)
            half = (
                stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
                if n > 1
                else 0.0
            )
            big = abs(mean) >= 1000
            text = f"{mean:.3g} ± {half:.2g}" if big else f"{mean:.3f} ± {half:.3f}"
            missing = len(members) - n
            cells.append(text + (f" ({n}/{len(members)})" if missing else ""))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("runs", nargs="+", type=Path)
    m.add_argument("--subjects", choices=["val", "test"], required=True)
    m.add_argument("--curves", type=Path)
    m.add_argument("--tolerance", type=float, default=0.1)
    t = sub.add_parser("table")
    t.add_argument("csv", type=Path)
    t.add_argument("--by", nargs="+", default=["scenario", "dataset"])
    t.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    args = parser.parse_args()
    if args.command == "table":
        print(table(args.csv, args.by, args.columns))
        return
    rows, curves = [], []
    for run in args.runs:
        got, curve = measure(run, args.subjects, args.tolerance)
        rows += got
        curves += curve
    _write(rows, sys.stdout)
    if args.curves is not None:
        with open(args.curves, "w", encoding="utf-8", newline="") as out:
            _write(curves, out)


if __name__ == "__main__":
    main()
