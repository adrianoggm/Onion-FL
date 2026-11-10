"""Tests for the explainers (issue #230).

The scorer is a hand-written rule over hand-picked rows: no model is trained or
evaluated (docs/RULES.md). The run comes from the CLI workspace, with the stub
trainer and no validation subjects.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from onion_fl.experiment.cli import main
from onion_fl.experiment.explain import EXPLAIN
from onion_fl.observability.explain import explainers
from onion_fl.observability.run import verify_run

from .test_cli import workspace  # noqa: F401  (the fixture)

# Feature "a" decides the label, "b" is a constant, "c" is irrelevant.
X = np.array([[1, 5, 3], [-1, 5, 1], [2, 5, 0], [-2, 5, 2], [3, 5, 1], [-3, 5, 0]])
Y = np.array([1, 0, 1, 0, 1, 0])
SUBJECTS = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])


def rule(X: np.ndarray, y: np.ndarray) -> float:
    return float(((X[:, 0] > 0).astype(int) == y).mean())


def explain(**params) -> dict:
    plugin = explainers.create("permutation", params)
    return plugin.explain("d", rule, X, Y, SUBJECTS, ["a", "b", "c"])


def test_the_feature_the_rule_reads_is_the_only_important_one() -> None:
    out = explain(repeats=20)

    by = {r["feature"]: r for r in out["features"]}
    assert out["baseline"] == 1.0 and out["metric"] == "macro_f1"
    assert [r["feature"] for r in out["features"]][0] == "a"
    assert by["a"]["importance"] > 0
    assert by["b"]["importance"] == by["c"]["importance"] == 0.0


def test_rows_are_shuffled_within_each_subject_only() -> None:
    seen: list[np.ndarray] = []

    def spy(X: np.ndarray, y: np.ndarray) -> float:
        seen.append(X.copy())
        return 0.0

    explainers.create("permutation").explain("d", spy, X, Y, SUBJECTS, ["a", "b", "c"])

    for shuffled in seen:
        for subject in ("s1", "s2"):
            rows = SUBJECTS == subject
            assert sorted(shuffled[rows, 0]) == sorted(X[rows, 0])


def test_the_permutations_are_seeded() -> None:
    assert explain(seed=3) == explain(seed=3)


def test_explain_writes_next_to_the_run_without_touching_its_hash(
    capsys,
    workspace: Path,  # noqa: F811
) -> None:
    experiment = yaml.safe_load((workspace / "exp.yaml").read_text(encoding="utf-8"))
    experiment["data"]["roles"] |= {"val": 0.0}
    (workspace / "exp.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    assert main(["run", str(workspace / "exp.yaml")]) == 0
    (path,) = [Path(line) for line in capsys.readouterr().out.split()]

    assert main(["explain", str(path), "--repeats", "2"]) == 0

    out = json.loads((path / EXPLAIN).read_text(encoding="utf-8"))
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert out["run_id"] == meta["run_id"] and out["run_hash"] == meta["run_hash"]
    assert out["explainer"] == {"name": "permutation", "repeats": 2}
    assert out["datasets"] == {}  # no val subjects: nothing is scored
    assert verify_run(path)
