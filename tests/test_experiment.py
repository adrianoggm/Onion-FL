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
    assert {p["name"] for p in plugins["transport"]} == {"memory", "mqtt"}


def test_fedrep_needs_sharing_that_keeps_the_heads_local(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {"trainer": "fedrep"}

    with pytest.raises(ConfigError, match="fedrep"):
        plan(parse_experiment(experiment(workspace, learning=learning)))
    local = learning | {"sharing": "fedper"}
    assert plan(parse_experiment(experiment(workspace, learning=local)))


def test_a_bad_scenario_fails_before_any_run_starts(workspace: Path) -> None:
    pairs = [["fedavg", "stub"], ["fedavg", "fedrep"]]
    config = parse_experiment(
        experiment(workspace, sweep={"learning.sharing,learning.trainer": pairs})
    )

    with pytest.raises(ConfigError, match="fedrep"):
        run_experiment(config)
    assert not list((workspace / "runs").glob("*"))


def test_the_experiment_can_set_the_root_server_optimizer(workspace: Path) -> None:
    from onion_fl.experiment.runner import resolve_topology

    learning = experiment(workspace)["learning"] | {"server_optimizer": "fedadam"}
    config = parse_experiment(experiment(workspace, learning=learning))

    assert resolve_topology(config).root.settings["server_optimizer"] == "fedadam"


def test_an_unset_server_optimizer_stays_out_of_the_config(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    assert "server_optimizer" not in config.dump()["learning"]


@pytest.mark.parametrize(
    "trainer, optimizer, message",
    [
        ("fednova", "replace", "needs server_optimizer 'fednova'"),
        ("scaffold", "replace", "needs server_optimizer 'scaffold'"),
        ("stub", "fednova", "fednova needs the fednova trainer"),
    ],
)
def test_paired_trainers_and_optimizers_are_checked_both_ways(
    workspace: Path, trainer: str, optimizer: str, message: str
) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": trainer,
        "server_optimizer": optimizer,
    }

    with pytest.raises(ConfigError, match=message):
        plan(parse_experiment(experiment(workspace, learning=learning)))


def test_a_trainer_that_needs_an_optimizer_is_checked(
    workspace: Path, monkeypatch
) -> None:
    from onion_fl.core.registry import PluginSpec
    from onion_fl.learning.trainers import Stub, StubParams, trainers

    class Needy(Stub):
        server_optimizer = "fedadam"

    spec = PluginSpec("needy_test", Needy, "t", "d", StubParams)
    monkeypatch.setitem(trainers._specs, "needy_test", spec)
    learning = experiment(workspace)["learning"] | {"trainer": "needy_test"}

    with pytest.raises(ConfigError, match="fedadam"):
        plan(parse_experiment(experiment(workspace, learning=learning)))
    ok = learning | {"server_optimizer": "fedadam"}
    assert plan(parse_experiment(experiment(workspace, learning=ok)))


@pytest.mark.parametrize("pair", ["fednova", "feddyn"])
def test_server_side_pairs_refuse_groups_held_below_the_root(
    workspace: Path, pair: str
) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": pair,
        "server_optimizer": pair,
        "sharing": "zone",
    }

    with pytest.raises(ConfigError, match="zone|below the root"):
        plan(parse_experiment(experiment(workspace, learning=learning)))


def test_moon_needs_the_feature_layers_shared(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {"trainer": "moon"}
    local = learning | {"sharing": "lg_fedavg"}

    with pytest.raises(ConfigError, match="moon.*lg_fedavg"):
        plan(parse_experiment(experiment(workspace, learning=local)))
    assert plan(parse_experiment(experiment(workspace, learning=learning)))


def test_paired_optimizers_refuse_late_updates(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": "fednova",
        "server_optimizer": "fednova",
    }
    fog = TOPOLOGY["fog"] | {"defaults": {"staleness": "next_round"}}
    topology = TOPOLOGY | {"fog": fog}

    with pytest.raises(ConfigError, match="staleness"):
        plan(
            parse_experiment(
                experiment(workspace, learning=learning, topology=topology)
            )
        )


@pytest.mark.parametrize("pair", ["scaffold", "feddyn"])
def test_edge_memory_pairs_refuse_rounds_that_close_at_quorum(
    workspace: Path, pair: str
) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": pair,
        "server_optimizer": pair,
    }
    fog = TOPOLOGY["fog"] | {"defaults": {"close_at_quorum": True, "quorum": 1}}
    topology = TOPOLOGY | {"fog": fog}

    with pytest.raises(ConfigError, match="close_at_quorum"):
        plan(
            parse_experiment(
                experiment(workspace, learning=learning, topology=topology)
            )
        )


def test_the_global_model_can_be_scored_on_the_validation_subjects(
    workspace: Path,
) -> None:
    from onion_fl.experiment.runner import _scenario_data, edge_specs

    def root_evaluators(evaluation: dict) -> list[str]:
        raw = experiment(workspace, evaluation=evaluation)
        (scenario,) = scenarios(parse_experiment(raw))
        topology, split, placement, _ = _scenario_data(scenario)
        edges, _ = edge_specs(scenario, topology, split, placement)
        return [spec.id for spec in edges[topology.root.id]]

    test = root_evaluators({"global": {"every": 1}})
    val = root_evaluators({"global": {"every": 1, "subjects": "val"}})

    assert test and all(i.startswith("test-") for i in test)
    assert val and all(i.startswith("gval-") for i in val)
    assert (
        "subjects"
        not in parse_experiment(experiment(workspace)).dump()["evaluation"]["global"]
    )


def test_feddyn_needs_the_same_alpha_on_both_sides(workspace: Path) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": {"name": "feddyn", "alpha": 0.1},
        "server_optimizer": {"name": "feddyn", "alpha": 0.2},
    }

    with pytest.raises(ConfigError, match="alpha"):
        plan(parse_experiment(experiment(workspace, learning=learning)))


@pytest.mark.parametrize(
    "extra",
    [
        {"privacy": "local_dp"},
        {"attack": "sign_flip"},
        {"learning": {"aggregator": "dp_fedavg"}},
        {"learning": {"aggregator": "norm_clip"}},
    ],
)
@pytest.mark.parametrize("trainer", ["scaffold", "fednova"])
def test_auxiliary_arrays_cannot_bypass_a_clip_noise_or_attack(
    workspace: Path, trainer: str, extra: dict
) -> None:
    learning = experiment(workspace)["learning"] | {"trainer": trainer}
    if trainer == "fednova":
        learning["server_optimizer"] = "fednova"
    extra = dict(extra)  # the parametrized dict is shared between trainers
    learning |= extra.pop("learning", {})

    with pytest.raises(ConfigError, match="auxiliary arrays"):
        plan(parse_experiment(experiment(workspace, learning=learning, **extra)))


def test_the_experiment_can_set_the_leaf_aggregators(workspace: Path) -> None:
    from onion_fl.experiment.runner import resolve_topology

    learning = experiment(workspace)["learning"] | {"aggregator": "median"}
    config = parse_experiment(experiment(workspace, learning=learning))
    topology = resolve_topology(config)

    leaves = {leaf.id for leaf in topology.leaves()}
    for node in topology.nodes:
        expected = "median" if node.id in leaves else None
        assert node.settings.get("aggregator") == expected, node.id


def test_an_unset_aggregator_stays_out_of_the_config(workspace: Path) -> None:
    dumped = parse_experiment(experiment(workspace)).dump()["learning"]

    assert "aggregator" not in dumped


def test_malicious_edges_are_a_seeded_fraction_of_each_dataset(
    workspace: Path,
) -> None:
    from onion_fl.experiment.runner import _scenario_data, malicious_edges

    attack = {"name": "sign_flip", "fraction": 0.5}
    config = parse_experiment(experiment(workspace, attack=attack))
    (scenario,) = scenarios(config)
    _, split, _, _ = _scenario_data(scenario)

    chosen = malicious_edges(scenario.config, split.clients, scenario.seed)

    assert len(chosen) == round(0.5 * len(split.clients))
    assert chosen == malicious_edges(scenario.config, split.clients, scenario.seed)


def test_a_malicious_fraction_rounds_half_up(workspace: Path) -> None:
    from types import SimpleNamespace

    from onion_fl.experiment.runner import malicious_edges

    attack = {"name": "sign_flip", "fraction": 0.5}
    config = parse_experiment(experiment(workspace, attack=attack))
    clients = [SimpleNamespace(id=f"e{i}", dataset="a") for i in range(5)]

    assert len(malicious_edges(config, clients, 0)) == 3  # 2.5, not banker's 2


def test_the_data_record_names_the_malicious_edges(workspace: Path) -> None:
    from onion_fl.experiment.runner import _scenario_data, record_data

    class Recorder:
        def __init__(self) -> None:
            self.names: list[str] = []

        def record(self, name, value=None, **tags) -> None:
            self.names.append(name)

    attack = {"name": "sign_flip", "fraction": 0.5}
    (scenario,) = scenarios(parse_experiment(experiment(workspace, attack=attack)))
    _, split, placement, _ = _scenario_data(scenario)
    run = Recorder()

    record_data(run, scenario, split, placement)  # real runs record through it too

    assert "data.attack" in run.names


def test_an_unset_attack_stays_out_of_the_config(workspace: Path) -> None:
    assert "attack" not in parse_experiment(experiment(workspace)).dump()


def test_an_unknown_scenario_name_is_an_error(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    with pytest.raises(ConfigError, match="nope.*base"):
        run_experiment(config, only="nope")


def test_finetuned_scores_need_a_finetune_trainer(workspace: Path) -> None:
    raw = experiment(workspace, evaluation={"edge": {"models": ["finetuned"]}})

    with pytest.raises(ConfigError, match="finetune"):
        parse_experiment(raw)


def test_a_finetune_trainer_needs_finetuned_scores(workspace: Path) -> None:
    raw = experiment(
        workspace,
        evaluation={"edge": {"models": ["local"], "finetune": "standard"}},
    )

    with pytest.raises(ConfigError, match="finetuned"):
        parse_experiment(raw)


def test_an_unset_finetune_stays_out_of_the_config(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace))

    assert "finetune" not in config.dump()["evaluation"]["edge"]


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


def test_paths_joined_by_commas_are_swept_together(workspace: Path) -> None:
    config = parse_experiment(
        experiment(
            workspace,
            sweep={"learning.trainer.shift,rounds": [[1.0, 1], [2.0, 3]]},
        )
    )

    out = scenarios(config)

    assert [s.name for s in out] == [
        "learning.trainer.shift,rounds=1.0,1",
        "learning.trainer.shift,rounds=2.0,3",
    ]
    assert [(s.config.learning.trainer["shift"], s.config.rounds) for s in out] == [
        (1.0, 1),
        (2.0, 3),
    ]


def test_a_plugin_given_whole_is_labelled_compactly(workspace: Path) -> None:
    trainers = [{"name": "stub", "shift": 2.0}, {"name": "stub", "shift": 3.0}]
    config = parse_experiment(
        experiment(workspace, sweep={"learning.trainer": trainers})
    )

    names = [s.name for s in scenarios(config)]

    assert names == [
        "learning.trainer=stub(shift=2.0)",
        "learning.trainer=stub(shift=3.0)",
    ]


def test_a_joint_sweep_key_may_have_spaces_after_its_commas(workspace: Path) -> None:
    config = parse_experiment(
        experiment(workspace, sweep={"learning.trainer.shift, rounds": [[2.0, 3]]})
    )

    (scenario,) = scenarios(config)

    assert scenario.name == "learning.trainer.shift,rounds=2.0,3"
    assert scenario.config.rounds == 3


@pytest.mark.parametrize("value", [[1.0], "fedavg"])
def test_a_joint_sweep_value_needs_one_entry_per_path(workspace: Path, value) -> None:
    config = parse_experiment(
        experiment(workspace, sweep={"learning.trainer.shift,rounds": [value]})
    )

    with pytest.raises(
        ConfigError, match="learning.trainer.shift,rounds: .*one per path"
    ):
        scenarios(config)


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


def test_a_custom_scorer_cannot_go_to_real_node_processes(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace, runtime={"mode": "real"}))
    (scenario,) = scenarios(config)

    with pytest.raises(ValueError, match="node processes"):
        run_scenario(scenario, evaluate=stub_score)


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


def test_a_run_keeps_its_final_global_model(workspace: Path) -> None:
    import numpy as np

    (scenario,) = scenarios(parse_experiment(experiment(workspace)))

    path = run_scenario(scenario, evaluate=stub_score)

    with np.load(path / "model.npz", allow_pickle=False) as saved:
        assert "trunk.0.weight" in saved.files


def test_plan_warns_about_lossy_links_without_deadline(workspace: Path) -> None:
    lossy = {"profile": {"preset": "lan", "loss": 0.1}}
    topology = TOPOLOGY | {
        "fog": {"defaults": {"link_up": lossy}, "nodes": TOPOLOGY["fog"]["nodes"]},
        "edge": {"link_up": lossy},
    }

    (preview,) = plan(parse_experiment(experiment(workspace, topology=topology)))
    (safe,) = plan(parse_experiment(experiment(workspace)))

    assert (
        len(preview["warnings"]) == 3
    )  # cloud (from the fogs) and each fog (from its edges)
    assert all("deadline" in w for w in preview["warnings"])
    assert safe["warnings"] == []


def test_every_simulated_run_writes_its_bundle_and_signs_it(workspace: Path) -> None:
    from onion_fl.continuum.bundle import load_bundle

    (scenario,) = scenarios(parse_experiment(experiment(workspace)))
    path = run_scenario(scenario, evaluate=stub_score)

    bundle = load_bundle(path / "bundle")
    assert bundle.snapshot.round == 2
    assert bundle.lineage == {
        "version": 2,
        "run_id": path.name,
        "parent": None,
    }
    assert bundle.preprocessing["demo"]["features"] == ["f1", "f2"]
    assert bundle.schema["demo"] == {"task": "stress", "n_classes": 2}
    assert verify_run(path)
    (path / "bundle" / "server.npz").write_bytes(b"tampered")
    assert not verify_run(path)
