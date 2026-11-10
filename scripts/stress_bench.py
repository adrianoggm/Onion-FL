"""onion-bench-stress (issue #244): the classical references and the scoring of runs.

Every number of ``results/stress_e4/`` comes from here, on the test subjects of
the experiment's roles, which are the same in every scenario and seed:

- ``baselines``: majority class, logistic regression per dataset and pooled
  over every dataset, local-only logistic regression (one per training edge)
  and the leave-one-dataset-out regression (pooled over the other datasets);
- ``score``: the final global model of each run (``model.npz``) on the same
  test subjects, through the same scoring function as the baselines;
- ``tables``: mean ± 95% CI over seeds, and differences paired by seed.

The primary metric is the macro-F1 over every test window of a dataset (the
pooled F1 that ``onion_fl baseline`` reports). The per-subject F1 is each test
subject's macro-F1 over the classes present in its own windows. It is not the
federation's ``global/<dataset>`` score, which averages the evaluators'
macro-F1 weighted by samples. ::

    python scripts/stress_bench.py baselines benchmarks/stress.yaml --out base.json
    python scripts/stress_bench.py score runs/ --out runs.csv
    python scripts/stress_bench.py tables runs.csv --reference fedavg_hier_per_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from onion_fl.baselines import baseline_models, mean_ci
from onion_fl.data.contract import SubjectData, natural_key
from onion_fl.data.roles import DataSplit, split_subjects
from onion_fl.experiment.config import (
    ExperimentConfig,
    load_experiment,
    parse_experiment,
)
from onion_fl.experiment.runner import load_data
from onion_fl.learning.metrics import predict
from onion_fl.learning.model import load_arrays, models
from onion_fl.roles.policies import create

Proba = Callable[[np.ndarray], np.ndarray]  # rows -> P(class) per column


# --- scoring -------------------------------------------------------------------------


def subject_f1(y: np.ndarray, pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in this subject's windows."""
    present = np.unique(y)
    return float(f1_score(y, pred, labels=present, average="macro", zero_division=0))


def score_dataset(tests: Sequence[SubjectData], proba: Proba) -> dict[str, Any]:
    """Pooled scores over every test window, plus each test subject's F1."""
    ys, preds, probs, per_subject = [], [], [], {}
    for data in tests:
        p = proba(data.X)
        pred = p.argmax(axis=1)
        ys.append(data.y)
        preds.append(pred)
        probs.append(p)
        per_subject[data.subject] = subject_f1(data.y, pred)
    y, pred, p = np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)
    labels = list(range(tests[0].n_classes))
    auroc = float("nan")
    if len(np.unique(y)) == 2 and p.shape[1] == 2:
        auroc = float(roc_auc_score(y, p[:, 1]))
    f1s = np.array(list(per_subject.values()))
    return {
        "macro_f1": float(
            f1_score(y, pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "auroc": auroc,
        "subject_f1_median": float(np.median(f1s)),
        "subject_f1_p10": float(np.percentile(f1s, 10)),
        "subject_f1": per_subject,
        "test_windows": int(len(y)),
    }


def tests_by_dataset(split: DataSplit) -> dict[str, list[SubjectData]]:
    out: dict[str, list[SubjectData]] = defaultdict(list)
    for data in split.test:
        out[data.dataset].append(data)
    return {d: sorted(v, key=lambda s: natural_key(s.subject)) for d, v in out.items()}


def _split(config: ExperimentConfig) -> DataSplit:
    subjects, _ = load_data(config)
    return split_subjects(subjects, config.data.roles)


# --- classical references ------------------------------------------------------------


def _fit(rows: Sequence[SubjectData], n_classes: int, seed: int) -> Proba:
    """Logistic regression on ``rows``; a single training class predicts that class."""
    X = np.concatenate([r.X for r in rows])
    y = np.concatenate([r.y for r in rows])
    if len(np.unique(y)) < 2:
        only = int(y[0])
        return lambda Z: np.eye(n_classes)[np.full(len(Z), only)]
    model = baseline_models.create("lr").build(seed)
    model.fit(X, y)
    classes = model.classes_

    def proba(Z: np.ndarray) -> np.ndarray:
        out = np.zeros((len(Z), n_classes))
        out[:, classes] = model.predict_proba(Z)
        return out

    return proba


def baselines(config: ExperimentConfig, seed: int = 0) -> dict[str, Any]:
    split = _split(config)
    tests = tests_by_dataset(split)
    train = defaultdict(list)
    for client in split.clients:
        train[client.dataset].append(client)
    every = [c.train for c in split.clients]
    out: dict[str, Any] = {
        "roles": {d: split.roles[d]["test"] for d in sorted(split.roles)},
        "datasets": {},
    }
    for dataset in sorted(tests):
        n = tests[dataset][0].n_classes
        mine = [c.train for c in train[dataset]]
        y = np.concatenate([t.y for t in mine])
        majority = int(np.bincount(y, minlength=n).argmax())
        others = [c.train for c in split.clients if c.dataset != dataset]
        local = [score_dataset(tests[dataset], _fit([c], n, seed)) for c in mine]
        row = {
            "train_windows": int(len(y)),
            "majority": score_dataset(
                tests[dataset], lambda Z, k=majority, n=n: np.eye(n)[np.full(len(Z), k)]
            ),
            "lr_dataset": score_dataset(tests[dataset], _fit(mine, n, seed)),
            "local_only": {
                "edges": len(local),
                **{
                    f"{stat}_{m}": float(fn([s[m] for s in local]))
                    for m in ("macro_f1", "balanced_accuracy")
                    for stat, fn in (("mean", np.mean), ("median", np.median))
                },
            },
        }
        if len(tests) > 1:
            row["lr_pooled"] = score_dataset(tests[dataset], _fit(every, n, seed))
            row["lr_lodo"] = score_dataset(tests[dataset], _fit(others, n, seed))
        out["datasets"][dataset] = row
    return out


# --- runs ----------------------------------------------------------------------------


def _model_proba(config: ExperimentConfig, state, data: SubjectData, seed: int):
    family = create(models, config.learning.model)
    model = family.build([data.shape], seed=seed)  # an edge's model: its dataset only
    load_arrays(model, state)

    def proba(Z: np.ndarray) -> np.ndarray:
        logits = predict(model, Z)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    return proba


def score_runs(folders: Sequence[Path]) -> list[dict[str, Any]]:
    splits: dict[str, DataSplit] = {}
    rows = []
    for folder in folders:
        meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        config = parse_experiment(meta["config"])
        key = json.dumps(config.data.model_dump(mode="json"), sort_keys=True)
        if key not in splits:
            splits[key] = _split(config)
        tests = tests_by_dataset(splits[key])
        state = dict(np.load(folder / "model.npz"))
        for dataset, holders in sorted(tests.items()):
            proba = _model_proba(config, state, holders[0], meta["seed"])
            scored = score_dataset(holders, proba)
            rows.append(
                {
                    "experiment": config.name,
                    "scenario": meta["scenario"],
                    "seed": meta["seed"],
                    "dataset": dataset,
                    **{k: v for k, v in scored.items() if k != "subject_f1"},
                    "subject_f1": json.dumps(scored["subject_f1"], sort_keys=True),
                    "run_id": meta["run_id"],
                    "config_id": meta["config_id"],
                    "topology_id": meta["topology_id"],
                    "data_id": meta["data_id"],
                    "run_hash": meta["run_hash"],
                    "commit": meta["code_version"]["commit"][:7],
                    "dirty": meta["code_version"]["dirty"],
                    "status": meta["status"],
                    "bytes": summary["messages"]["bytes"],
                    "quorum_failed": summary["quorum_failed"],
                    "wall_seconds": round(summary["wall_seconds"], 1),
                }
            )
    return rows


# --- tables --------------------------------------------------------------------------


def _ci(values: Sequence[float]) -> str:
    stats = mean_ci(values)
    if stats["n"] < 2:
        return f"{stats['mean']:.3f}"
    return f"{stats['mean']:.3f} ± {stats['ci95']:.3f}"


def tables(rows: list[dict[str, str]], reference: str | None, label) -> str:
    by = defaultdict(dict)  # (scenario, dataset) -> seed -> row
    for row in rows:
        by[label(row), row["dataset"]][int(row["seed"])] = row
    datasets = sorted({d for _, d in by})
    scenarios = list(dict.fromkeys(s for s, _ in by))
    lines = []
    for metric in ("macro_f1", "subject_f1_median", "subject_f1_p10"):
        lines += [f"### {metric} (mean ± 95% CI over seeds)", ""]
        lines += ["| Scenario | " + " | ".join(datasets) + " |"]
        lines += ["|---|" + "---|" * len(datasets)]
        for s in scenarios:
            cells = [
                _ci([float(r[metric]) for r in by[s, d].values()])
                if (s, d) in by
                else "—"
                for d in datasets
            ]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    if reference:
        lines += [f"### macro_f1 minus {reference}, paired by seed", ""]
        lines += ["| Scenario | " + " | ".join(datasets) + " |"]
        lines += ["|---|" + "---|" * len(datasets)]
        for s in scenarios:
            if s == reference:
                continue
            cells = []
            for d in datasets:
                mine, ref = by.get((s, d), {}), by.get((reference, d), {})
                seeds = sorted(set(mine) & set(ref))
                diffs = [
                    float(mine[k]["macro_f1"]) - float(ref[k]["macro_f1"])
                    for k in seeds
                ]
                cells.append(_ci(diffs) if diffs else "—")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    b.add_argument("experiment")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+", help="run folders, or folders that hold them")
    s.add_argument("--out", required=True)
    t = sub.add_parser("tables")
    t.add_argument("csv", nargs="+")
    t.add_argument("--reference")
    t.add_argument(
        "--names", help="JSON file: scenario name -> short label (others kept)"
    )
    args = parser.parse_args(argv)

    if args.cmd == "baselines":
        result = baselines(load_experiment(args.experiment), args.seed)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.cmd == "score":
        folders = []
        for given in map(Path, args.runs):
            if (given / "run.json").exists():
                folders.append(given)
            else:
                folders += sorted(p.parent for p in given.glob("*/model.npz"))
        rows = score_runs(folders)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        names = json.loads(Path(args.names).read_text()) if args.names else {}
        rows = []
        for path in args.csv:
            with open(path, encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
        print(
            tables(
                rows,
                args.reference,
                lambda r: names.get(r["scenario"], r["scenario"]),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
