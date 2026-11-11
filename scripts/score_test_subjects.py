"""Score each run's final model on every test subject, one by one.

A run reports its global test metrics over all test subjects together
(``summary.json``, ``global/*``). In cross-silo (Fed-Heart-Disease, issue #251)
each test subject is a hospital's official test split, and the comparison
needs the score per hospital. This script rebuilds each run's split from its
own ``run.json`` (the same roles, preprocessing and data cache), loads its
``model.npz`` and scores it on each test subject. It checks that the
sample-weighted mean equals the run's ``global/*`` accuracy, and prints JSON::

    python scripts/score_test_subjects.py results/heart_disease_triangle/*/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from onion_fl.data.roles import split_subjects
from onion_fl.experiment.config import ExperimentConfig
from onion_fl.experiment.runner import _shapes, load_data
from onion_fl.learning.metrics import compute, predict
from onion_fl.learning.model import load_arrays, models


def score(run: Path) -> dict:
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(meta["config"])
    split = split_subjects(load_data(config)[0], config.data.roles)
    ref = config.learning.model
    name, params = (ref, {}) if isinstance(ref, str) else (ref["name"], ref)
    family = models.create(name, {k: v for k, v in params.items() if k != "name"})
    model = family.build(_shapes(split), seed=meta["seed"])
    with np.load(run / "model.npz") as arrays:
        load_arrays(model, dict(arrays))
    metrics = [m for m in config.evaluation.metrics if m != "loss"]
    per_subject = {}
    for data in split.test:
        y = np.asarray(data.y, np.int64)
        values = compute(predict(model, data.X), y, data.n_classes, metrics)
        per_subject[data.subject] = {"samples": len(y), **values}
    total = sum(s["samples"] for s in per_subject.values())
    pooled = sum(s["accuracy"] * s["samples"] for s in per_subject.values()) / total
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    reported = summary["final"]["global/*"]["accuracy"]
    if not math.isclose(pooled, reported, abs_tol=1e-9):
        raise SystemExit(f"{run}: rescored accuracy {pooled}, the run has {reported}")
    return {
        "run_id": meta["run_id"],
        "scenario": meta["scenario"],
        "seed": meta["seed"],
        "config_id": meta["config_id"],
        "topology_id": meta["topology_id"],
        "run_hash": meta["run_hash"],
        "pooled_accuracy": pooled,
        "test": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([score(run) for run in args.runs], indent=1))


if __name__ == "__main__":
    main()
