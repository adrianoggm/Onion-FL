"""Score each run's final global model (``model.npz``) offline on its held-out subjects.

The root's last in-run evaluation can be from an earlier round when the last
rounds fail, and a stream evaluator scores only what arrived; this scores the
final model on every row of the validation or test subjects, pooled per
dataset (docs/RULES.md: results cite the final model).

    python scripts/score_final_models.py runs/<run_id> ... --subjects val > out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.runner import _scenario_data
from onion_fl.experiment.sweep import Scenario
from onion_fl.learning.metrics import compute, predict
from onion_fl.learning.model import load_arrays
from onion_fl.learning.model import models as model_families
from onion_fl.roles.policies import create

METRICS = ["loss", "accuracy", "macro_f1", "confusion_matrix"]


def score(run: Path, subjects: str) -> list[dict]:
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = parse_experiment(meta["config"], signed=True)
    scenario = Scenario(meta["scenario"], meta["seed"], config, meta["config_id"])
    _, split, _, _ = _scenario_data(scenario)
    family = create(model_families, config.learning.model)
    with np.load(run / "model.npz") as saved:
        state = dict(saved)
    rows = []
    held = split.val if subjects == "val" else split.test
    for dataset in sorted({d.dataset for d in held}):
        data = [d for d in held if d.dataset == dataset]
        model = family.build([data[0].shape], seed=meta["seed"])
        load_arrays(model, state)
        X = np.concatenate([d.X for d in data])
        y = np.concatenate([d.y for d in data])
        scores = compute(predict(model, X), y, data[0].n_classes, METRICS)
        rows.append(
            {
                "run_id": meta["run_id"],
                "scenario": meta["scenario"],
                "seed": meta["seed"],
                "config_id": meta["config_id"],
                "subjects": subjects,
                "dataset": dataset,
                "samples": len(y),
                **{k: round(float(v), 6) for k, v in scores.items()},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--subjects", choices=["val", "test"], required=True)
    args = parser.parse_args()
    rows = [row for run in args.runs for row in score(run, args.subjects)]
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
