"""Tests for experiment configs, sweeps, plan and the runner (issue #97).

The dataset is a tiny CSV format fixture; edges use the stub trainer and the
scores come from a stub scorer, so nothing is trained or evaluated for real
(docs/RULES.md).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from onion_fl.experiment.config import (
    ConfigError,
    experiment_schema,
    load_experiment,
    parse_experiment,
)
from onion_fl.experiment.runner import plan, run_experiment, run_scenario
from onion_fl.experiment.sweep import scenarios
from onion_fl.observability.run import verify_run

TOPOLOGY = {
    "name": "two_fogs",
    "levels": ["global", "fog", "edge"],
    "root": {"id": "cloud"},
    "fog": {
        "nodes": [{"id": "fog_a", "home": "demo"}, {"id": "fog_b", "home": "demo"}]
    },
}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    rows = ["pp,cond,f1,f2"]
    for subject in range(1, 13):
        for i in range(4):
            rows.append(f"{subject},{'NT'[i % 2]},{subject + i},{subject * 2 - i}")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "table.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "datasets").mkdir()
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
    return tmp_path


def experiment(workspace: Path, **overrides) -> dict:
    base = {
        "name": "demo_exp",
        "topology": TOPOLOGY,
        "data": {
            "datasets": {"demo": {}},
            "roles": {"test": 0.25, "val": 0.17, "seed": 0},
            "placement": {"name": "mixing", "alpha": 0.0},
        },
        "learning": {
            "model": {"name": "modular_mlp", "adapter_width": 4, "trunk_hidden": [4]},
            "trainer": "stub",
        },
        "rounds": 2,
        "evaluation": {"global": {"every": 1}, "aggregators": {"every": 1}},
        "seeds": [0],
        "paths": {
            "datasets": str(workspace / "datasets"),
            "runs": str(workspace / "runs"),
            "cache": str(workspace / "cache"),
            "topologies": str(workspace / "topologies"),
        },
    }
    return base | overrides


def stub_score(model, data):
    return {"accuracy": 0.5}, data.n_samples


# --- validation ------------------------------------------------------------------------------


def test_a_valid_experiment_loads_from_yaml(workspace: Path) -> None:
    import yaml

    path = workspace / "exp.yaml"
    path.write_text(yaml.safe_dump(experiment(workspace)), encoding="utf-8")

    config = load_experiment(path)

    assert config.name == "demo_exp" and config.rounds == 2
    assert config.evaluation.global_.every == 1


@pytest.mark.parametrize(
    "change, where",
    [
        ({"rounds": 0}, "rounds"),
        ({"learning": {"trainer": {"name": "standard", "lr": -1}}}, "learning.trainer"),
        ({"learning": {"trainer": "teleport"}}, "learning.trainer"),
        ({"data": {"datasets": {"demo": {"colour": 1}}}}, "data.datasets.demo.colour"),
        (
            {
                "data": {
                    "datasets": {"demo": {}},
                    "placement": {"name": "mixing", "alpha": 2},
                }
            },
            "data.placement",
        ),
        ({"evaluation": {"metrics": ["accuracy", "vibes"]}}, "evaluation.metrics"),
        ({"sinks": ["carrier_pigeon"]}, "sinks"),
        ({"surprise": True}, "surprise"),
    ],
)
def test_errors_name_the_exact_path(workspace: Path, change: dict, where: str) -> None:
    with pytest.raises(ConfigError, match=where.replace(".", r"\.")):
        parse_experiment(experiment(workspace, **change))


def test_the_schema_includes_the_plugin_catalogue() -> None:
    schema = experiment_schema()

    assert {"topology", "data", "learning", "rounds", "evaluation", "sweep"} <= set(
        schema["properties"]
    )
    plugins = schema["plugins"]
    assert {p["name"] for p in plugins["trainer"]} >= {"standard", "fedprox", "stub"}
    assert {p["name"] for p in plugins["placement"]} >= {
        "mixing",
        "dirichlet",
        "pooled",
    }
    assert "params" in plugins["aggregator"][0]


# --- sweeps ------------------------------------------------------------------------------------


def test_a_sweep_is_the_product_of_its_values_times_the_seeds(workspace: Path) -> None:
    config = parse_experiment(
        experiment(
            workspace,
            seeds=[0, 1],
            sweep={"data.placement.alpha": [0.0, 1.0], "rounds": [1, 2, 3]},
        )
    )

    out = scenarios(config)

    assert len(out) == 2 * 3 * 2
    names = {s.name for s in out}
    assert "data.placement.alpha=0.0,rounds=1" in names and len(names) == 6
    by_name = {}
    for s in out:
        by_name.setdefault(s.name, set()).add(s.config_id)
    assert all(
        len(ids) == 1 for ids in by_name.values()
    )  # the seed is not in config_id
    assert len({next(iter(ids)) for ids in by_name.values()}) == 6


def test_without_a_sweep_there_is_one_scenario_per_seed(workspace: Path) -> None:
    out = scenarios(parse_experiment(experiment(workspace, seeds=[3, 4])))

    assert [(s.name, s.seed) for s in out] == [("base", 3), ("base", 4)]


def test_sweeping_a_plugin_given_by_name(workspace: Path) -> None:
    config = parse_experiment(
        experiment(workspace, sweep={"learning.trainer.shift": [1.0, 2.0]})
    )

    trainer = [s.config.learning.trainer for s in scenarios(config)]

    assert trainer == [{"name": "stub", "shift": 1.0}, {"name": "stub", "shift": 2.0}]


def test_a_bad_sweep_value_names_its_scenario(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace, sweep={"rounds": [1, 0]}))

    with pytest.raises(ConfigError, match="rounds=0"):
        scenarios(config)


# --- plan ---------------------------------------------------------------------------------------


def test_plan_previews_composition_and_traffic_without_running(workspace: Path) -> None:
    (preview,) = plan(parse_experiment(experiment(workspace)))

    assert preview["scenario"] == "base" and preview["seed"] == 0
    assert set(preview["composition"]) == {"fog_a", "fog_b"}
    assert sum(leaf["subjects"] for leaf in preview["composition"].values()) == 7
    roles = preview["roles"]["demo"]
    assert (len(roles["test"]), len(roles["val"]), len(roles["train"])) == (3, 2, 7)
    leaf_link = next(link for link in preview["traffic"] if link["child"] == "*")
    assert "trunk" in leaf_link["groups"]
    assert preview["graph"]["nodes"][0]["id"] == "cloud"
    assert not (workspace / "runs").exists()


# --- running ---------------------------------------------------------------------------------------


def test_a_scenario_runs_and_leaves_a_signed_folder(workspace: Path) -> None:
    (scenario,) = scenarios(parse_experiment(experiment(workspace)))

    path = run_scenario(scenario, evaluate=stub_score)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished" and meta["scenario"] == "base"
    assert meta["config_id"] == scenario.config_id
    assert len(meta["data_id"]) == 64 and verify_run(path)
    lines = (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    names = {json.loads(line)["name"] for line in lines}
    assert {"data.composition", "data.roles", "eval.accuracy", "run.finished"} <= names
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert summary["rounds"] == 2 and summary["final"]["global/demo"]["accuracy"] == 0.5


def test_a_failing_scenario_is_recorded_as_failed(workspace: Path) -> None:
    config = parse_experiment(
        experiment(
            workspace,
            learning={
                "init": {"name": "checkpoint", "path": str(workspace / "missing.npz")},
                "trainer": "stub",
            },
        )
    )
    (scenario,) = scenarios(config)

    with pytest.raises(FileNotFoundError):
        run_scenario(scenario, evaluate=stub_score)

    (run_json,) = (workspace / "runs").glob("*/run.json")
    assert json.loads(run_json.read_text(encoding="utf-8"))["status"] == "failed"


def test_a_codec_override_changes_the_topology_id(workspace: Path) -> None:
    def topology_id(**runtime):
        (scenario,) = scenarios(
            parse_experiment(experiment(workspace, runtime=runtime))
        )
        path = run_scenario(scenario, evaluate=stub_score)
        return json.loads((path / "run.json").read_text(encoding="utf-8"))[
            "topology_id"
        ]

    assert topology_id() != topology_id(codec="npz")


def test_an_experiment_runs_every_scenario_in_parallel(workspace: Path) -> None:
    config = parse_experiment(
        experiment(
            workspace,
            evaluation={"global": {"every": None}},  # no scoring in the workers
            seeds=[0, 1],
            sweep={"data.placement.alpha": [0.0, 1.0]},
        )
    )

    paths = run_experiment(config, workers=2)

    assert len(paths) == 4
    statuses = {
        json.loads((p / "run.json").read_text(encoding="utf-8"))["status"]
        for p in paths
    }
    assert statuses == {"finished"}


def test_the_real_mode_is_not_available_yet(workspace: Path) -> None:
    (scenario,) = scenarios(
        parse_experiment(experiment(workspace, runtime={"mode": "real"}))
    )

    with pytest.raises(NotImplementedError, match="real"):
        run_scenario(scenario)


@pytest.mark.skipif(not Path("data/SWELL").exists(), reason="data/SWELL not available")
def test_a_real_swell_experiment_runs_end_to_end(tmp_path: Path) -> None:
    config = parse_experiment(
        {
            "name": "swell_smoke",
            "topology": TOPOLOGY
            | {
                "fog": {
                    "nodes": [
                        {"id": "fog_a", "home": "swell"},
                        {"id": "fog_b", "home": "swell"},
                    ]
                }
            },
            "data": {
                "datasets": {"swell": {}},
                "roles": {"test": 0.2, "local_val": 0.2},
            },
            "learning": {"trainer": {"name": "standard", "local_epochs": 1}},
            "rounds": 2,
            "evaluation": {"edge": {"every": 1}},
            "paths": {"runs": str(tmp_path / "runs"), "cache": str(tmp_path / "cache")},
        }
    )

    (path,) = run_experiment(config)

    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    assert (
        summary["finished"]
        and 0.0 <= summary["final"]["global/swell"]["accuracy"] <= 1.0
    )


def test_topologies_are_found_by_name(workspace: Path) -> None:
    import yaml

    (workspace / "topologies").mkdir()
    (workspace / "topologies" / "two_fogs.yaml").write_text(
        yaml.safe_dump(TOPOLOGY), encoding="utf-8"
    )

    (preview,) = plan(parse_experiment(experiment(workspace, topology="two_fogs")))

    assert preview["graph"]["name"] == "two_fogs"


def test_the_edge_device_sets_the_training_time(workspace: Path) -> None:
    topology = TOPOLOGY | {"edge": {"device": {"samples_per_second": 10}}}
    (scenario,) = scenarios(parse_experiment(experiment(workspace, topology=topology)))

    path = run_scenario(scenario, evaluate=stub_score)

    closed = [
        json.loads(line)
        for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"round.closed"' in line and '"fog_a"' in line
    ]
    assert closed and all(e["value"] >= 1.0 for e in closed)  # 10 stub examples at 10/s


def test_live_sinks_can_be_switched_on(workspace: Path) -> None:
    sinks = [{"name": "prometheus", "port": 0}, "otel"]
    (scenario,) = scenarios(parse_experiment(experiment(workspace, sinks=sinks)))

    path = run_scenario(scenario, evaluate=stub_score)

    assert (
        json.loads((path / "run.json").read_text(encoding="utf-8"))["status"]
        == "finished"
    )


def test_a_custom_scorer_cannot_go_to_worker_processes(workspace: Path) -> None:
    with pytest.raises(ValueError, match="workers=1"):
        run_experiment(
            parse_experiment(experiment(workspace)), workers=2, evaluate=stub_score
        )


def test_a_sharing_that_does_not_fit_the_model_is_rejected(workspace: Path) -> None:
    learning = {"sharing": "independent", "trainer": "stub"}

    with pytest.raises(ConfigError, match="learning.sharing"):
        parse_experiment(experiment(workspace, learning=learning))


def test_a_sweep_path_through_a_value_is_rejected(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace, sweep={"rounds.x": [1]}))

    with pytest.raises(ConfigError, match="rounds is not a mapping"):
        scenarios(config)


def test_a_run_that_never_finishes_its_rounds_is_incomplete(workspace: Path) -> None:
    lossy = {"profile": {"preset": "lan", "loss": 0.9}}
    topology = TOPOLOGY | {
        "fog": {
            "defaults": {"link_up": lossy, "hello_retry": None},
            "nodes": TOPOLOGY["fog"]["nodes"],
        },
        "edge": {"link_up": lossy, "hello_retry": None},
    }
    (scenario,) = scenarios(parse_experiment(experiment(workspace, topology=topology)))

    path = run_scenario(scenario, evaluate=stub_score)

    assert (
        json.loads((path / "run.json").read_text(encoding="utf-8"))["status"]
        == "incomplete"
    )
