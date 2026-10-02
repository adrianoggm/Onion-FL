"""Tests for the ``onion_fl`` command line (issue #98).

The workspace holds a tiny CSV format fixture and an experiment with the stub
trainer and no scoring, so nothing is trained or evaluated (docs/RULES.md).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from onion_fl.experiment.cli import main

TOPOLOGY = {
    "name": "two_fogs",
    "levels": ["global", "fog", "edge"],
    "root": {"id": "cloud"},
    "fog": {
        "defaults": {"link_up": {"profile": "wifi"}},
        "nodes": [{"id": "fog_a", "home": "demo"}, {"id": "fog_b", "home": "demo"}],
    },
    "edge": {"link_up": {"profile": "4g"}},
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    rows = ["pp,cond,f1,f2"] + [
        f"{s},{'NT'[i % 2]},{s + i},{s * 2 - i}" for s in range(1, 13) for i in range(4)
    ]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "table.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    for folder in ("datasets", "topologies"):
        (tmp_path / folder).mkdir()
    (tmp_path / "datasets" / "demo.yaml").write_text(
        textwrap.dedent(
            f"""
            name: demo
            root: {(tmp_path / "raw").as_posix()}
            source: {{reader: csv, path: table.csv}}
            steps:
              - subject: {{column: pp}}
              - label: {{task: stress, column: cond, map: {{"N": 0, "T": 1}}}}
              - features: {{}}
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "topologies" / "two_fogs.yaml").write_text(
        yaml.safe_dump(TOPOLOGY), encoding="utf-8"
    )
    experiment = {
        "name": "cli_demo",
        "topology": "two_fogs",
        "data": {"datasets": {"demo": {}}, "roles": {"test": 0.25}},
        "learning": {
            "model": {"name": "modular_mlp", "adapter_width": 4, "trunk_hidden": [4]},
            "trainer": "stub",
        },
        "rounds": 2,
        "evaluation": {"global": {"every": None}},
        "paths": {
            "datasets": str(tmp_path / "datasets"),
            "topologies": str(tmp_path / "topologies"),
            "runs": str(tmp_path / "runs"),
            "cache": str(tmp_path / "cache"),
        },
    }
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    return tmp_path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_schema_prints_the_config_schema_and_the_plugins(capsys) -> None:
    code, out, _ = run(capsys, "schema")

    schema = json.loads(out)
    assert code == 0 and "plugins" in schema and "trainer" in schema["plugins"]


def test_schema_can_be_written_to_a_file(capsys, tmp_path: Path) -> None:
    code, _, _ = run(capsys, "schema", "--out", str(tmp_path / "schema.json"))

    assert code == 0 and "properties" in json.loads(
        (tmp_path / "schema.json").read_text(encoding="utf-8")
    )


def test_topology_show_draws_the_tree(capsys, workspace: Path) -> None:
    code, out, _ = run(
        capsys, "topology", "show", str(workspace / "topologies" / "two_fogs.yaml")
    )

    assert code == 0
    assert "topology_id" in out and "cloud" in out and "fog_b" in out and "wifi" in out


def test_topology_show_exports_the_graph(capsys, workspace: Path) -> None:
    code, out, _ = run(
        capsys,
        "topology",
        "show",
        str(workspace / "topologies" / "two_fogs.yaml"),
        "--graph",
    )

    graph = json.loads(out)
    assert code == 0 and [n["id"] for n in graph["nodes"]] == [
        "cloud",
        "fog_a",
        "fog_b",
    ]


def test_data_prepare_and_inspect(capsys, workspace: Path) -> None:
    common = [
        "--datasets-dir",
        str(workspace / "datasets"),
        "--cache-dir",
        str(workspace / "cache"),
    ]

    code, out, _ = run(capsys, "data", "prepare", "demo", *common)
    assert code == 0 and (Path(out.strip()) / "meta.json").exists()

    code, out, _ = run(capsys, "data", "inspect", "demo", *common)
    card = json.loads(out)
    assert code == 0 and card["n_subjects"] == 12 and card["n_samples"] == 48


def test_data_options_are_parsed_as_yaml_values(capsys, workspace: Path) -> None:
    common = [
        "--datasets-dir",
        str(workspace / "datasets"),
        "--cache-dir",
        str(workspace / "cache"),
    ]

    code, _, err = run(
        capsys, "data", "prepare", "demo", "--option", "colour=1", *common
    )

    assert code == 2 and "colour" in err


def test_plan_prints_the_dry_run(capsys, workspace: Path) -> None:
    code, out, _ = run(capsys, "plan", str(workspace / "exp.yaml"))

    (preview,) = json.loads(out)
    assert code == 0 and set(preview["composition"]) == {"fog_a", "fog_b"}
    assert not (workspace / "runs").exists()


def test_run_and_report(capsys, workspace: Path) -> None:
    code, out, _ = run(capsys, "run", str(workspace / "exp.yaml"))

    (path,) = [Path(line) for line in out.split()]
    assert (
        code == 0
        and json.loads((path / "run.json").read_text(encoding="utf-8"))["status"]
        == "finished"
    )

    report = workspace / "report.html"
    code, out, _ = run(
        capsys,
        "report",
        str(workspace / "exp.yaml"),
        "--out",
        str(report),
        "--metric",
        "train_loss",
    )
    assert code == 0 and "<svg" in report.read_text(encoding="utf-8")


def test_report_reads_run_folders_too(capsys, workspace: Path) -> None:
    run(capsys, "run", str(workspace / "exp.yaml"))
    report = workspace / "r.html"

    code, _, _ = run(capsys, "report", str(workspace / "runs"), "--out", str(report))

    assert code == 0 and report.exists()


def test_run_can_pick_one_scenario(capsys, workspace: Path) -> None:
    code, out, _ = run(capsys, "run", str(workspace / "exp.yaml"), "--scenario", "nope")

    assert code == 0 and out.strip() == ""


def test_the_real_mode_and_node_are_not_available_yet(capsys, workspace: Path) -> None:
    code, _, err = run(capsys, "run", str(workspace / "exp.yaml"), "--mode", "real")
    assert code == 2 and "real" in err

    code, _, err = run(
        capsys,
        "node",
        "--id",
        "fog_a",
        "--run",
        "x",
        "--config",
        str(workspace / "exp.yaml"),
    )
    assert code == 2 and "F8.1" in err


def test_config_errors_name_their_path(capsys, workspace: Path) -> None:
    bad = yaml.safe_load((workspace / "exp.yaml").read_text(encoding="utf-8")) | {
        "rounds": 0
    }
    (workspace / "bad.yaml").write_text(yaml.safe_dump(bad), encoding="utf-8")

    code, _, err = run(capsys, "plan", str(workspace / "bad.yaml"))

    assert code == 2 and "rounds" in err


def test_baseline_reports_a_missing_experiment(capsys, tmp_path: Path) -> None:
    code, _, err = run(capsys, "baseline", str(tmp_path / "missing.yaml"))

    assert code == 2 and "missing.yaml" in err


# --- the shipped examples -------------------------------------------------------------------


def test_the_example_topology_and_experiment_are_valid() -> None:
    from onion_fl.core.topology import load_topology
    from onion_fl.experiment.config import load_experiment
    from onion_fl.experiment.sweep import scenarios

    topology = load_topology("topologies/four_fogs.yaml")
    config = load_experiment("experiments/mix_ab.yaml")

    assert [leaf.id for leaf in topology.leaves()] == [
        "fog_a1",
        "fog_a2",
        "fog_b1",
        "fog_b2",
    ]
    assert len(scenarios(config)) == 9  # 3 alphas x 3 seeds


@pytest.mark.skipif(
    not (Path("data/SWELL").exists() and Path("data/SWEET/sample_subjects").exists()),
    reason="data/SWELL and data/SWEET not available",
)
def test_the_example_experiment_plans_on_real_data(capsys) -> None:
    code, out, _ = run(capsys, "plan", "experiments/mix_ab.yaml")

    assert code == 0 and len(json.loads(out)) == 9
