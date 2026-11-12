"""Tests for load_runs, metrics, compare and the HTML report (issues #95, #288).

The run folders are hand-written fixtures with known values, to check the
arithmetic of the comparisons; they are not experiment results.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from onion_fl.observability.analysis import load_runs


def write_run(
    root: Path, run_id: str, topology: str, seed: int, events, experiment="mix"
) -> None:
    path = root / run_id
    path.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "topology_id": topology,
        "config_id": f"cfg-{topology}",
        "seed": seed,
        "scenario": "",
        "status": "finished",
        "config": {"experiment": experiment},
    }
    (path / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = []
    for node, level, round, value, tags in events:
        lines.append(
            json.dumps(
                {
                    "t_virtual": float(round),
                    "t_wall": 0.0,
                    "run_id": run_id,
                    "topology_id": topology,
                    "scenario": "",
                    "seed": seed,
                    "round": round,
                    "level": level,
                    "node": node,
                    "role": "aggregator",
                    "kind": "metric",
                    "name": "eval.accuracy",
                    "value": value,
                    "tags": {"round": round, "model": "local", **tags},
                }
            )
        )
    (path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path / "summary.json").write_text("{}", encoding="utf-8")


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    fog = "fog"
    write_run(
        tmp_path,
        "r1",
        "T1",
        0,
        [("fog_0", fog, 1, 0.5, {}), ("fog_1", fog, 1, 0.7, {})],
    )
    write_run(
        tmp_path,
        "r2",
        "T1",
        1,
        [("fog_0", fog, 1, 0.6, {}), ("fog_1", fog, 1, 0.8, {})],
    )
    write_run(
        tmp_path,
        "r3",
        "T2",
        0,
        [("fog_0", fog, 1, 0.9, {}), ("fog_1", fog, 1, 0.9, {})],
    )
    write_run(
        tmp_path,
        "r4",
        "T2",
        0,
        [
            ("cloud", "global", 1, 0.4, {"dataset": "a"}),
            ("cloud", "global", 1, 0.6, {"dataset": "b"}),
        ],
        experiment="other",
    )
    return tmp_path


def test_load_runs_reads_every_folder_and_filters_by_experiment(runs_dir: Path) -> None:
    assert len(load_runs(runs_dir)) == 4
    assert {r.meta["run_id"] for r in load_runs(runs_dir, experiment="mix")} == {
        "r1",
        "r2",
        "r3",
    }


def test_load_runs_skips_folders_without_a_run(runs_dir: Path) -> None:
    (runs_dir / "not_a_run").mkdir()

    assert len(load_runs(runs_dir)) == 4


def test_metrics_is_a_table_with_every_tag(runs_dir: Path) -> None:
    table = load_runs(runs_dir).metrics(level="global", name="accuracy")

    assert sorted(table["dataset"]) == ["a", "b"]
    assert set(table.columns) >= {
        "run_id",
        "topology_id",
        "seed",
        "round",
        "node",
        "value",
        "model",
    }


def test_metrics_filters_by_tag(runs_dir: Path) -> None:
    table = load_runs(runs_dir).metrics(level="global", name="accuracy", dataset="b")

    assert table["value"].tolist() == [0.6]


def test_compare_gives_the_mean_and_ci_over_seeds_and_the_spread_across_nodes(
    runs_dir: Path,
) -> None:
    out = load_runs(runs_dir, experiment="mix").compare(
        level="fog", metric="accuracy", by=["topology_id"]
    )

    t1 = out[out["topology_id"] == "T1"].iloc[0]
    assert t1["n"] == 2 and t1["round"] == 1
    assert t1["mean"] == pytest.approx(0.65)
    half = 12.7062047 * math.sqrt(0.005) / math.sqrt(2)  # t(0.975, 1) * sd / sqrt(n)
    assert t1["ci_low"] == pytest.approx(0.65 - half, rel=1e-5)
    assert t1["ci_high"] == pytest.approx(0.65 + half, rel=1e-5)
    assert t1["node_min"] == pytest.approx(0.55)
    assert t1["node_max"] == pytest.approx(0.75)
    assert t1["node_spread"] == pytest.approx(0.1)


def test_compare_with_one_seed_has_no_interval(runs_dir: Path) -> None:
    out = load_runs(runs_dir, experiment="mix").compare(
        level="fog", metric="accuracy", by=["topology_id"]
    )

    t2 = out[out["topology_id"] == "T2"].iloc[0]
    assert t2["n"] == 1 and t2["mean"] == pytest.approx(0.9)
    assert math.isnan(t2["ci_low"]) and t2["node_spread"] == 0.0


def test_compare_can_split_by_tags(runs_dir: Path) -> None:
    out = load_runs(runs_dir).compare(
        level="global", metric="accuracy", by=["topology_id", "dataset"]
    )

    assert sorted(out["dataset"]) == ["a", "b"]
    assert out.set_index("dataset")["mean"].to_dict() == pytest.approx(
        {"a": 0.4, "b": 0.6}
    )


def test_compare_of_nothing_is_empty(runs_dir: Path) -> None:
    out = load_runs(runs_dir).compare(level="fog", metric="loss", by=["topology_id"])

    assert out.empty


# --- HTML report ---------------------------------------------------------------------------


def test_the_report_compares_groups_per_level(runs_dir: Path, tmp_path: Path) -> None:
    path = load_runs(runs_dir).report(
        tmp_path / "report.html", metrics=["accuracy"], by=["topology_id"]
    )

    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    for text in ("fog", "global", "accuracy", "T1", "T2", "<svg", "0.650"):
        assert text in html
    assert "r1" in html  # the runs behind the numbers are listed


def test_the_report_escapes_what_it_shows(tmp_path: Path) -> None:
    write_run(tmp_path / "runs", "r1", "<b>T</b>", 0, [("fog_0", "fog", 1, 0.5, {})])

    html = (
        load_runs(tmp_path / "runs")
        .report(tmp_path / "r.html")
        .read_text(encoding="utf-8")
    )

    assert "<b>T</b>" not in html and "&lt;b&gt;T&lt;/b&gt;" in html


# --- on a real recorded run ------------------------------------------------------------------


def test_load_runs_reads_what_run_writes(tmp_path: Path) -> None:
    from onion_fl.core.topology import parse_topology
    from onion_fl.learning.model import (
        DataShape,
        ModularMLP,
        ModularMLPConfig,
        state_arrays,
    )
    from onion_fl.learning.trainers import trainers
    from onion_fl.observability.run import Run
    from onion_fl.roles import EdgeSpec, build_federation

    shape = DataShape(dataset="a", task="t", n_features=2, n_classes=2)
    config = ModularMLPConfig(adapter_width=2, trunk_hidden=[2], dropout=0.0)
    topology = parse_topology(
        {
            "name": "t",
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", "eval": {"every": 1}},
            "fog": {"nodes": [{"id": "fog_0"}]},
        }
    )
    edges = {
        "fog_0": [
            EdgeSpec("e1", ModularMLP(config, [shape]), trainer=trainers.create("stub"))
        ],
        "cloud": [
            EdgeSpec(
                "t1",
                ModularMLP(config, [shape]),
                data=0.5,
                train=False,
                tags={"dataset": "a"},
            )
        ],
    }
    fed = build_federation(
        topology,
        edges,
        initial_state=state_arrays(ModularMLP(config, [shape])),
        rounds=2,
        evaluate=lambda model, data: ({"accuracy": data}, 1),
    )
    run = Run(tmp_path, config={"experiment": "smoke"}, topology=topology, seed=0)
    run.attach(fed)
    fed.run()
    run.finish()

    table = load_runs(tmp_path, experiment="smoke").metrics(
        level="global", name="accuracy", model="global"
    )

    assert table["round"].tolist() == [1, 1, 2, 2]  # overall and dataset a, per round
    assert set(table["value"]) == {0.5}


# --- different series are never averaged (QA2, #174) -------------------------------


def test_compare_keeps_models_and_datasets_apart(tmp_path: Path) -> None:
    events = [
        ("cloud", "global", 1, 0.9, {"model": "global", "source": "evaluators"}),
        ("cloud", "global", 1, 0.5, {"model": "received", "source": "children"}),
        ("cloud", "global", 1, 0.7, {"model": "global", "dataset": "swell"}),
    ]
    write_run(tmp_path, "r1", "topo", 0, events)

    out = load_runs(tmp_path).compare(
        level="global", metric="accuracy", by=["topology_id"]
    )

    assert sorted(out["mean"]) == [0.5, 0.7, 0.9]  # three series, none averaged
    assert {"model", "dataset", "source"} <= set(out.columns)
    assert out.attrs["by"][0] == "topology_id"


# --- paired comparisons by seed (#288) ---------------------------------------------

# The per-seed accuracy of three arms; B - A and C - A are the differences
# the paired statistics are checked on.
ARMS = {
    "A": [0.70, 0.72, 0.68, 0.75, 0.71, 0.69, 0.74, 0.70],
    "B": [0.73, 0.74, 0.69, 0.79, 0.73, 0.72, 0.76, 0.74],
    "C": [0.701, 0.719, 0.682, 0.749, 0.711, 0.691, 0.741, 0.699],
}


@pytest.fixture
def paired_dir(tmp_path: Path) -> Path:
    for arm, values in ARMS.items():
        for seed, value in enumerate(values):
            # two nodes whose mean is the arm's value, so node averaging comes first
            events = [
                ("fog_0", "fog", 1, value - 0.01, {}),
                ("fog_1", "fog", 1, value + 0.01, {}),
            ]
            write_run(tmp_path, f"{arm}{seed}", arm, seed, events)
    return tmp_path


def _differences(arm: str) -> np.ndarray:
    return np.array(ARMS[arm]) - np.array(ARMS["A"])


def test_paired_compare_checks_each_arm_against_the_reference_with_scipy(
    paired_dir: Path,
) -> None:
    out = load_runs(paired_dir).compare(
        level="fog",
        metric="accuracy",
        paired=True,
        reference="A",
        margin=0.005,
    )
    rows = out.set_index("topology_id")
    b, c = _differences("B"), _differences("C")

    assert rows.loc["A", "verdict"] == "reference" and rows.loc["A", "n_pairs"] == 0
    assert rows.loc["B", "n_pairs"] == 8
    assert rows.loc["B", "delta"] == pytest.approx(b.mean())
    assert rows.loc["B", "delta_low"] < b.mean() < rows.loc["B", "delta_high"]
    assert rows.loc["B", "d_z"] == pytest.approx(b.mean() / b.std(ddof=1))
    assert rows.loc["B", "p_wilcoxon"] == pytest.approx(stats.wilcoxon(b).pvalue)
    exact = stats.permutation_test(
        (b,), np.mean, permutation_type="samples", n_resamples=np.inf
    )
    assert rows.loc["B", "p_perm"] == pytest.approx(
        exact.pvalue
    )  # 2 / 256: every difference is positive
    tost = max(
        stats.ttest_1samp(c, -0.005, alternative="greater").pvalue,
        stats.ttest_1samp(c, 0.005, alternative="less").pvalue,
    )
    assert rows.loc["C", "p_tost"] == pytest.approx(tost)
    # Holm over the two arms of the family
    p = [rows.loc["B", "p_perm"], rows.loc["C", "p_perm"]]
    lo, hi = sorted(p)
    assert sorted([rows.loc["B", "p_adj"], rows.loc["C", "p_adj"]]) == pytest.approx(
        [2 * lo, max(2 * lo, hi)]
    )
    assert rows.loc["B", "verdict"] == "different"
    assert rows.loc["C", "verdict"] == "equivalent"


def test_paired_off_leaves_compare_as_it_was(paired_dir: Path) -> None:
    runs = load_runs(paired_dir)
    plain = runs.compare(level="fog", metric="accuracy")
    paired = runs.compare(level="fog", metric="accuracy", paired=True, reference="A")

    assert "delta" not in plain.columns
    pd.testing.assert_frame_equal(paired[plain.columns], plain)
    assert paired["p_tost"].isna().all()  # no margin, no equivalence test


def test_paired_compare_pairs_only_the_shared_seeds(paired_dir: Path) -> None:
    for seed in range(3, 8):
        shutil.rmtree(paired_dir / f"B{seed}")

    out = load_runs(paired_dir).compare(
        level="fog", metric="accuracy", paired=True, reference="A"
    )

    b = out.set_index("topology_id").loc["B"]
    assert b["n_pairs"] == 3
    assert b["delta"] == pytest.approx(_differences("B")[:3].mean())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reference": "Z"}, "no runs of the reference"),
        ({"reference": None}, "needs a reference"),
        ({"reference": "A", "target": 0.5}, "per round"),
        ({"reference": {"scenario": "x"}}, "needs a value for each"),
    ],
)
def test_paired_compare_refuses_what_it_cannot_pair(
    paired_dir: Path, kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_runs(paired_dir).compare(
            level="fog", metric="accuracy", paired=True, **kwargs
        )


def test_paired_compare_refuses_two_runs_of_a_group_on_one_seed(
    paired_dir: Path,
) -> None:
    write_run(paired_dir, "A0bis", "A", 0, [("fog_0", "fog", 1, 0.5, {})])

    with pytest.raises(ValueError, match="several runs share"):
        load_runs(paired_dir).compare(
            level="fog", metric="accuracy", paired=True, reference="A"
        )


def test_the_report_adds_the_paired_columns(paired_dir: Path, tmp_path: Path) -> None:
    path = load_runs(paired_dir).report(
        tmp_path / "report.html", metrics=["accuracy"], reference="A", margin=0.005
    )

    html = path.read_text(encoding="utf-8")
    for text in ("delta", "p_adj", "p_tost", "d_z", "equivalent", "TOST within"):
        assert text in html


def test_a_reference_can_be_a_scenario_name_with_an_equals_sign(
    tmp_path: Path,
) -> None:
    for seed in range(2):
        for scenario, value in (("alpha=0.0", 0.5), ("alpha=1.0", 0.6 + seed / 10)):
            run_id = f"{scenario[-3:]}-{seed}"
            write_run(tmp_path, run_id, "T", seed, [("fog_0", "fog", 1, value, {})])
            meta = json.loads((tmp_path / run_id / "run.json").read_text("utf-8"))
            (tmp_path / run_id / "run.json").write_text(
                json.dumps(meta | {"scenario": scenario}), encoding="utf-8"
            )

    out = load_runs(tmp_path).compare(
        level="fog",
        metric="accuracy",
        by=["scenario"],
        paired=True,
        reference="alpha=0.0",
    )

    rows = out.set_index("scenario")
    assert rows.loc["alpha=0.0", "verdict"] == "reference"
    assert rows.loc["alpha=1.0", "delta"] == pytest.approx(0.15)
