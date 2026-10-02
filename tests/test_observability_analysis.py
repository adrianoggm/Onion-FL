"""Tests for load_runs, metrics, compare and the HTML report (issue #95).

The run folders are hand-written fixtures with known values, to check the
arithmetic of the comparisons; they are not experiment results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

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
