from __future__ import annotations

"""Explain a finished run's model: ``onion_fl explain <run folder>`` (issue #230).

The scenario is rebuilt from ``run.json`` (its config and seed), so the split
and the preprocessing are the run's own. The final ``model.npz`` is scored on
the ``val`` subjects of each dataset whose groups it holds; a dataset whose
groups stay local (not in the global model) is listed as skipped.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from onion_fl.experiment.config import parse_experiment
from onion_fl.experiment.sweep import Scenario
from onion_fl.learning.metrics import compute, predict
from onion_fl.learning.model import load_arrays, models
from onion_fl.observability.explain import explainers
from onion_fl.roles.policies import create

EXPLAIN = "explain.json"


def explain_run(
    path: str | Path, explainer: str | dict[str, Any] = "permutation"
) -> Path:
    """Write ``explain.json`` into the run folder ``path`` and return its path."""
    from onion_fl.experiment.runner import _scenario_data

    path = Path(path)
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    config = parse_experiment(meta["config"])
    scenario = Scenario(meta["scenario"], meta["seed"], config, meta["config_id"])
    _, split, _, _ = _scenario_data(scenario, parent=None)
    with np.load(path / "model.npz") as npz:
        state = {k: npz[k] for k in npz.files}
    plugin = create(explainers, explainer)
    family = create(models, config.learning.model)
    out: dict[str, Any] = {
        k: meta.get(k) for k in ("run_id", "run_hash", "config_id", "topology_id")
    }
    out |= {
        "code_version": meta.get("code_version"),
        "explainer": explainer,
        "datasets": {},
        "skipped": {},
    }
    for dataset in sorted({d.dataset for d in split.val}):
        val = [d for d in split.val if d.dataset == dataset]
        model = family.build([val[0].shape], seed=meta["seed"])
        missing = sorted(set(model.state_dict()) - set(state))
        if missing:
            out["skipped"][dataset] = f"not in the global model: {missing[:3]}"
            continue
        load_arrays(model, state)
        first = val[0]

        def score(X: np.ndarray, y: np.ndarray, model=model, n=first.n_classes):
            return compute(predict(model, X), y, n, [plugin.metric])[plugin.metric]

        X = np.concatenate([d.X for d in val])
        y = np.concatenate([np.asarray(d.y, np.int64) for d in val])
        subjects = np.concatenate([[d.subject] * len(d.y) for d in val])
        result = plugin.explain(
            dataset, score, X, y, subjects, list(first.feature_names)
        )
        out["datasets"][dataset] = result | {
            "subjects": [d.subject for d in val],
            "rows": int(len(y)),
        }
    target = path / EXPLAIN
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return target
