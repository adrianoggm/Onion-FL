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
        ({"sweep": {"seeds": [[0], [1]]}}, "sweep"),  # it would do nothing
        ({"sweep": {"rounds, sweep.rounds": [[1, [2]]]}}, "sweep"),
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


@pytest.mark.parametrize(
    "trainer", ["fedbabu", {"name": "standard", "frozen": ["trunk"]}]
)
def test_central_noise_cannot_reach_a_frozen_group(workspace: Path, trainer) -> None:
    learning = experiment(workspace)["learning"] | {
        "trainer": trainer,
        "aggregator": "dp_fedavg",
    }

    with pytest.raises(ConfigError, match="frozen"):
        plan(parse_experiment(experiment(workspace, learning=learning)))


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


def test_the_scenarios_of_a_process_share_one_metrics_server(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onion_fl.observability import sinks as live

    bound = []
    monkeypatch.setattr(live, "start_http_server", lambda port, **_: bound.append(port))
    monkeypatch.setattr(live, "_SERVED", {})
    sinks = [{"name": "prometheus", "port": 9999}]
    config = parse_experiment(experiment(workspace, sinks=sinks, seeds=[0, 1]))

    run_experiment(config, evaluate=stub_score)

    assert bound == [9999]  # a second bind of the port would fail on Linux


def test_worker_processes_cannot_share_a_metrics_port(workspace: Path) -> None:
    config = parse_experiment(experiment(workspace, sinks=["prometheus"], seeds=[0, 1]))

    with pytest.raises(ConfigError, match="workers"):
        run_experiment(config, workers=2)


def _mark(item: tuple[Path, int]) -> int:
    import time

    folder, i = item
    if i == 1:
        raise RuntimeError("scenario 1 broke")
    time.sleep(3.0 if i == 0 else 0.2)
    (folder / f"{i}.done").touch()
    return i


def test_a_parallel_sweep_stops_at_its_first_failure(tmp_path: Path) -> None:
    from onion_fl.experiment.runner import _in_parallel

    with pytest.raises(RuntimeError, match="scenario 1"):  # while scenario 0 runs
        _in_parallel(_mark, [(tmp_path, i) for i in range(20)], workers=2)

    later = [p for p in tmp_path.glob("*.done") if p.stem != "0"]
    assert len(later) < 8  # the ones handed to a worker; not the other 18
    assert _in_parallel(_mark, [(tmp_path, 3), (tmp_path, 2)], workers=2) == [3, 2]


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


def test_plan_warns_when_a_lost_evaluation_would_block_the_end(workspace: Path) -> None:
    lossy = {"profile": {"preset": "lan", "loss": 0.1}}
    topology = TOPOLOGY | {"edge": {"link_up": lossy}}  # the evaluators' link too

    (preview,) = plan(parse_experiment(experiment(workspace, topology=topology)))

    assert any(w.startswith("cloud:") for w in preview["warnings"])


def test_plan_warns_when_a_lost_update_keeps_a_child_out_for_good(
    workspace: Path,
) -> None:
    lossy = {"profile": {"preset": "lan", "loss": 0.1}}
    buffering = {"close_at_quorum": True, "quorum": 1, "deadline": 10}
    fog = {"defaults": buffering, "nodes": TOPOLOGY["fog"]["nodes"]}
    kept = {"staleness": {"name": "next_round"}}
    bounded = {"staleness": {"name": "next_round", "max_staleness": 2}}

    def warnings(defaults: dict) -> list[str]:
        topology = TOPOLOGY | {
            "fog": fog | {"defaults": buffering | defaults},
            "edge": {"link_up": lossy},
        }
        (preview,) = plan(parse_experiment(experiment(workspace, topology=topology)))
        return [w for w in preview["warnings"] if "max_staleness" in w]

    assert len(warnings(kept)) == 2  # one per fog
    assert warnings(bounded) == []


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
        "algorithms": {
            "trainer": "stub",
            "server_optimizer": "replace",
            "root": "cloud",
            "privacy": {"edges": None, "aggregators": {}},
        },
    }
    assert bundle.preprocessing["demo"]["features"] == ["f1", "f2"]
    assert bundle.schema["demo"] == {"task": "stress", "n_classes": 2}
    assert verify_run(path)
    (path / "bundle" / "server.npz").write_bytes(b"tampered")
    assert not verify_run(path)


# --- continuing a run (continuum C2) ------------------------------------------------


def _final_model(path: Path) -> dict:
    import numpy as np

    with np.load(path / "model.npz") as archive:
        return {k: archive[k] for k in archive.files}


def _noisy(workspace: Path, **learning) -> dict:
    base = experiment(workspace)["learning"] | {
        "trainer": {"name": "stub", "noise": 0.1}
    }
    return base | learning


def test_a_continued_run_equals_one_that_never_stopped(workspace: Path) -> None:
    import numpy as np

    straight = experiment(workspace, learning=_noisy(workspace), rounds=4)
    (whole,) = scenarios(parse_experiment(straight))
    expected = _final_model(run_scenario(whole, evaluate=stub_score))

    first = experiment(workspace, learning=_noisy(workspace), rounds=2)
    (head,) = scenarios(parse_experiment(first))
    parent = run_scenario(head, evaluate=stub_score)
    resume = {"name": "run", "run": parent.name}
    second = experiment(workspace, learning=_noisy(workspace, init=resume), rounds=2)
    (tail,) = scenarios(parse_experiment(second))
    child = run_scenario(tail, evaluate=stub_score)

    for key, value in expected.items():
        np.testing.assert_array_equal(_final_model(child)[key], value, err_msg=key)
    meta = json.loads((child / "run.json").read_text(encoding="utf-8"))
    parent_meta = json.loads((parent / "run.json").read_text(encoding="utf-8"))
    assert meta["parent"] == {
        "run_id": parent.name,
        "run_hash": parent_meta["run_hash"],
        "version": 2,
    }
    from onion_fl.continuum.bundle import load_bundle

    assert load_bundle(child / "bundle").lineage["version"] == 4
    assert verify_run(child)


def test_a_tampered_parent_is_refused(workspace: Path) -> None:
    (head,) = scenarios(parse_experiment(experiment(workspace)))
    parent = run_scenario(head, evaluate=stub_score)
    (parent / "summary.json").write_text("{}", encoding="utf-8")
    learning = experiment(workspace)["learning"] | {
        "init": {"name": "run", "run": parent.name}
    }

    with pytest.raises(ConfigError, match="does not verify"):
        plan(parse_experiment(experiment(workspace, learning=learning)))


def test_a_continuation_can_start_from_a_fresh_model(workspace: Path) -> None:
    from onion_fl.experiment.runner import _scenario_data, build_scenario

    (head,) = scenarios(parse_experiment(experiment(workspace)))
    parent = run_scenario(head, evaluate=stub_score)
    resume = {"name": "run", "run": parent.name, "restore": {"model": False}}
    learning = experiment(workspace)["learning"] | {"init": resume}
    (tail,) = scenarios(parse_experiment(experiment(workspace, learning=learning)))
    topology, split, placement, _ = _scenario_data(tail)

    federation = build_scenario(tail, topology, split, placement, evaluate=stub_score)

    trained = _final_model(parent)
    state = federation.coordinator.state
    assert federation.coordinator.round == 2  # the numbering still continues
    assert any(not (state[k] == trained[k]).all() for k in trained)


def test_a_parent_can_be_named_by_experiment_and_seed(workspace: Path) -> None:
    parent_config = experiment(workspace, name="parent_exp", seeds=[0, 1])
    parents = {
        s.seed: run_scenario(s, evaluate=stub_score)
        for s in scenarios(parse_experiment(parent_config))
    }
    learning = experiment(workspace)["learning"] | {
        "init": {"name": "run", "run": "experiment:parent_exp"}
    }
    child_config = experiment(workspace, learning=learning, seeds=[1])
    (child,) = scenarios(parse_experiment(child_config))

    path = run_scenario(child, evaluate=stub_score)

    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["parent"]["run_id"] == parents[1].name


def test_a_continuation_refuses_test_subjects_that_trained_the_parent(
    workspace: Path,
) -> None:
    (head,) = scenarios(parse_experiment(experiment(workspace)))
    parent = run_scenario(head, evaluate=stub_score)
    raw = experiment(workspace)
    raw["learning"] = raw["learning"] | {"init": {"name": "run", "run": parent.name}}
    raw["data"] = raw["data"] | {"roles": {"test": 0.25, "val": 0.17, "seed": 1}}

    with pytest.raises(ConfigError, match="keeps its role"):
        plan(parse_experiment(raw))


@pytest.mark.parametrize(
    "test, val, refused",
    [
        (["2"], ["3", "4"], True),  # 1: test -> train
        (["1", "2"], ["3"], True),  # 4: val -> train
        (["1", "2", "5"], ["3", "4"], True),  # 5: train -> test
        (["1", "2", "3"], ["4"], True),  # 3: val -> test
        (["1", "2"], ["3", "4", "5"], True),  # 5: train -> val
        (["1"], ["2", "3", "4"], True),  # 2: test -> val
        (["1", "2", "12"], ["3", "4"], False),  # 12 is new: it may join any role
    ],
    ids=[
        "test_to_train",
        "val_to_train",
        "train_to_test",
        "val_to_test",
        "train_to_val",
        "test_to_val",
        "new_subject",
    ],
)
def test_a_subject_keeps_its_role_along_a_lineage(
    workspace: Path, test: list[str], val: list[str], refused: bool
) -> None:
    def roles(raw: dict, override: dict) -> dict:
        raw["data"]["roles"] = raw["data"]["roles"] | {"overrides": {"demo": override}}
        return raw

    first = {"test": ["1", "2"], "val": ["3", "4"], "exclude": ["12"]}
    (head,) = scenarios(parse_experiment(roles(experiment(workspace), first)))
    parent = run_scenario(head, evaluate=stub_score)
    child = roles(_child(workspace, parent.name), {"test": test, "val": val})

    if refused:
        with pytest.raises(ConfigError, match="keeps its role"):
            plan(parse_experiment(child))
    else:
        assert plan(parse_experiment(child))


def _parent_run(workspace: Path, **overrides) -> Path:
    (head,) = scenarios(parse_experiment(experiment(workspace, **overrides)))
    return run_scenario(head, evaluate=stub_score)


def _child(workspace: Path, run: str, restore: dict | None = None, **overrides) -> dict:
    init = {"name": "run", "run": run} | ({"restore": restore} if restore else {})
    learning = overrides.pop("learning", experiment(workspace)["learning"])
    return experiment(workspace, learning=learning | {"init": init}, **overrides)


@pytest.mark.parametrize(
    "change, match",
    [
        (
            {
                "learning": {
                    "model": {
                        "name": "modular_mlp",
                        "adapter_width": 4,
                        "trunk_hidden": [4],
                    },
                    "trainer": "standard",
                }
            },
            "edge_state",
        ),
        (
            {
                "learning": {
                    "model": {
                        "name": "modular_mlp",
                        "adapter_width": 4,
                        "trunk_hidden": [4],
                    },
                    "trainer": "stub",
                    "server_optimizer": "fedavgm",
                }
            },
            "server_state",
        ),
        ({"seeds": [1]}, "seed"),
        ({"topology": {**TOPOLOGY, "root": {"id": "server"}}}, "root"),
        ({"runtime": {"mode": "real"}}, "real"),
    ],
    ids=["trainer", "server_optimizer", "seed", "root", "real"],
)
def test_a_continuation_that_cannot_be_exact_is_refused(
    workspace: Path, change: dict, match: str
) -> None:
    parent = _parent_run(workspace)

    with pytest.raises(ConfigError, match=match):
        plan(parse_experiment(_child(workspace, parent.name, **change)))


def test_a_changed_trainer_is_allowed_without_edge_state(workspace: Path) -> None:
    parent = _parent_run(workspace)
    learning = experiment(workspace)["learning"] | {"trainer": "standard"}

    raw = _child(workspace, parent.name, {"edge_state": False}, learning=learning)
    assert plan(parse_experiment(raw))


def test_an_unfinished_parent_is_refused(workspace: Path) -> None:
    from onion_fl.observability.run import compute_run_hash

    parent = _parent_run(workspace)
    meta = json.loads((parent / "run.json").read_text(encoding="utf-8"))
    meta["status"] = "incomplete"
    (parent / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    meta["run_hash"] = compute_run_hash(parent)
    (parent / "run.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ConfigError, match="finished"):
        plan(parse_experiment(_child(workspace, parent.name)))


def test_a_swept_parent_experiment_must_name_its_scenario(workspace: Path) -> None:
    sweep = {"learning.trainer": ["stub", {"name": "stub", "shift": 2.0}]}
    parents = [
        run_scenario(s, evaluate=stub_score)
        for s in scenarios(
            parse_experiment(experiment(workspace, name="parent_exp", sweep=sweep))
        )
    ]
    chosen = json.loads((parents[1] / "run.json").read_text(encoding="utf-8"))[
        "scenario"
    ]

    with pytest.raises(ConfigError, match="scenarios"):
        plan(parse_experiment(_child(workspace, "experiment:parent_exp")))
    raw = _child(workspace, f"experiment:parent_exp/{chosen}")
    (child,) = scenarios(parse_experiment(raw))
    path = run_scenario(child, evaluate=stub_score)
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    assert meta["parent"]["run_id"] == parents[1].name


def test_a_fresh_model_also_leaves_the_zones_fresh(workspace: Path) -> None:
    from onion_fl.experiment.runner import _scenario_data, build_scenario

    zone = experiment(workspace)["learning"] | {
        "sharing": {"name": "zone", "level": "fog"}
    }
    parent = _parent_run(workspace, learning=zone)
    raw = _child(workspace, parent.name, {"model": False}, learning=zone)
    (tail,) = scenarios(parse_experiment(raw))
    topology, split, placement, _ = _scenario_data(tail)

    federation = build_scenario(tail, topology, split, placement, evaluate=stub_score)

    assert all(not fog.zone for fog in federation.aggregators.values())


def test_a_typo_in_restore_is_an_error(workspace: Path) -> None:
    with pytest.raises(ConfigError, match="edge_states"):
        parse_experiment(_child(workspace, "x", {"edge_states": False}))


def test_a_fresh_model_also_leaves_the_edge_models_fresh(workspace: Path) -> None:
    import numpy as np

    from onion_fl.experiment.runner import _scenario_data, build_scenario
    from onion_fl.learning.model import state_arrays

    fedper = experiment(workspace)["learning"] | {"sharing": "fedper"}  # local heads
    parent = _parent_run(workspace, learning=fedper)

    def heads(restore: dict) -> dict:
        raw = _child(workspace, parent.name, restore, learning=fedper)
        (tail,) = scenarios(parse_experiment(raw))
        topology, split, placement, _ = _scenario_data(tail)
        federation = build_scenario(
            tail, topology, split, placement, evaluate=stub_score
        )
        return {
            (edge_id, key): value
            for edge_id, edge in federation.edges.items()
            for key, value in state_arrays(edge.model).items()
            if key.startswith("head.")
        }

    kept = heads({})
    fresh = heads({"model": False})  # edge_state stays on
    nothing = heads({"model": False, "edge_state": False, "server_state": False})

    assert any(not np.array_equal(kept[k], v) for k, v in nothing.items())  # trained
    for key, value in nothing.items():
        np.testing.assert_array_equal(fresh[key], value, err_msg=str(key))


def test_restore_keeps_an_edges_algorithm_memory_but_never_old_weights() -> None:
    import numpy as np

    from onion_fl.experiment.runner import _restore_parts
    from onion_fl.learning.trainers import RestoreParams
    from onion_fl.roles import FederationSnapshot, NodeState

    one = np.ones(1)
    arrays = ["model/w", "memory/_personal/w", "memory/_c_i/w"]
    weights = {"kind": "module", "training": True, "weights": True}
    memory = {"_personal": weights, "_c_i": {"kind": "arrays"}}
    edge = NodeState(dict.fromkeys(arrays, one), {"rng": {}, "memory": memory})
    snapshot = FederationSnapshot(2, {"e1": edge}, "cloud")

    def kept(**restore) -> tuple:
        parts = _restore_parts(snapshot, RestoreParams(**restore), {"e1"})
        node = parts.nodes["e1"]
        return sorted(node.arrays), node.meta

    assert kept(model=False) == (
        ["memory/_c_i/w"],
        {"rng": {}, "memory": {"_c_i": {"kind": "arrays"}}},
    )
    assert kept(edge_state=False) == (
        ["memory/_personal/w", "model/w"],
        {"memory": {"_personal": weights}},
    )
    assert kept(model=False, edge_state=False) == ([], {"memory": {}})


def test_restore_keeps_the_stream_of_an_edge_with_a_dp_budget() -> None:
    import numpy as np

    from onion_fl.experiment.runner import _restore_parts
    from onion_fl.learning.trainers import RestoreParams
    from onion_fl.roles import FederationSnapshot, NodeState

    private = NodeState({"privacy/rdp": np.ones(1)}, {"rng": {"s": 1}})
    plain = NodeState({}, {"rng": {"s": 2}})
    snapshot = FederationSnapshot(2, {"e1": private, "e2": plain}, "cloud")

    parts = _restore_parts(snapshot, RestoreParams(edge_state=False), {"e1", "e2"})

    # a fresh stream would replay the parent's noise, release for release
    assert parts.nodes["e1"].meta == {"rng": {"s": 1}}
    assert parts.nodes["e2"].meta == {}


def test_restore_puts_scaffolds_global_control_variate_under_server_state() -> None:
    import numpy as np

    from onion_fl.experiment.runner import _restore_parts
    from onion_fl.learning.trainers import RestoreParams
    from onion_fl.roles import FederationSnapshot, NodeState

    one = np.ones(1)
    arrays = ["state/trunk.0.weight", "state/scaffold/trunk.0.weight", "server/m"]
    root = NodeState(dict.fromkeys(arrays, one), {"rng": {}})
    snapshot = FederationSnapshot(2, {"cloud": root}, "cloud")

    def kept(**restore) -> list[str]:
        parts = _restore_parts(snapshot, RestoreParams(**restore), set())
        return sorted(parts.nodes["cloud"].arrays)

    assert kept(model=False) == ["server/m", "state/scaffold/trunk.0.weight"]
    assert kept(server_state=False) == ["state/trunk.0.weight"]


@pytest.mark.parametrize("trainer", ["scaffold", "feddyn"])
def test_a_paired_algorithm_restores_its_server_and_edge_state_together(
    workspace: Path, trainer: str
) -> None:
    learning = experiment(workspace)["learning"] | {"trainer": trainer}
    raw = _child(workspace, "anything", {"server_state": False}, learning=learning)

    with pytest.raises(ConfigError, match="together"):
        plan(parse_experiment(raw))


@pytest.mark.parametrize(
    "parent_change, child_change",
    [
        ({"privacy": {"name": "local_dp"}}, {}),
        ({}, {"privacy": {"name": "local_dp"}}),
        (
            {
                "topology": TOPOLOGY
                | {
                    "fog": {
                        "defaults": {"aggregator": "dp_fedavg"},
                        "nodes": TOPOLOGY["fog"]["nodes"],
                    }
                }
            },
            {},
        ),
    ],
    ids=["local_dropped", "local_added", "central_dropped"],
)
def test_a_continuation_keeps_the_dp_mechanism_of_its_parent(
    workspace: Path, parent_change: dict, child_change: dict
) -> None:
    parent = _parent_run(workspace, **parent_change)

    with pytest.raises(ConfigError, match="privacy"):
        plan(parse_experiment(_child(workspace, parent.name, **child_change)))


ONE_FOG = {
    "name": "one_fog",
    "levels": ["global", "fog", "edge"],
    "root": {"id": "cloud"},
    "fog": {"nodes": [{"id": "fog_a", "home": "demo"}]},
}


def _private(topology: dict) -> dict:
    fog = {"defaults": {"aggregator": "dp_fedavg"}, "nodes": topology["fog"]["nodes"]}
    return topology | {"fog": fog}


def test_a_dp_budget_outlives_a_generation_without_its_aggregator(
    workspace: Path,
) -> None:
    import numpy as np

    from onion_fl.continuum.bundle import load_bundle

    a = _parent_run(workspace, topology=_private(TOPOLOGY))
    b_raw = _child(workspace, a.name, topology=_private(ONE_FOG))
    (b_scenario,) = scenarios(parse_experiment(b_raw))
    b = run_scenario(b_scenario, evaluate=stub_score)

    a_fog = load_bundle(a / "bundle").snapshot.nodes["fog_b"]
    b_bundle = load_bundle(b / "bundle")
    np.testing.assert_array_equal(
        b_bundle.snapshot.nodes["fog_b"].arrays["aggregator/rdp"],
        a_fog.arrays["aggregator/rdp"],
    )
    assert b_bundle.lineage["algorithms"]["privacy"]["aggregators"] == {
        "fog_a": "dp_fedavg",
        "fog_b": "dp_fedavg",
    }
    assert plan(
        parse_experiment(_child(workspace, b.name, topology=_private(TOPOLOGY)))
    )
    with pytest.raises(ConfigError, match="privacy"):
        plan(parse_experiment(_child(workspace, b.name)))  # fog_b without its DP


@pytest.mark.parametrize("restore", [None, {"model": False}], ids=["all", "no_model"])
def test_a_dp_budget_outlives_a_generation_without_its_edge(
    workspace: Path, restore: dict | None
) -> None:
    import numpy as np

    from onion_fl.continuum.bundle import load_bundle

    descriptor = (workspace / "datasets" / "demo.yaml").read_text(encoding="utf-8")
    (workspace / "datasets" / "other.yaml").write_text(
        descriptor.replace("name: demo", "name: other"), encoding="utf-8"
    )
    dp = {"name": "local_dp", "clip": 10.0}
    raw = _datasets(experiment(workspace, privacy=dp), {"demo": {}, "other": {}})
    (a_scenario,) = scenarios(parse_experiment(raw))
    a = run_scenario(a_scenario, evaluate=stub_score)
    b_raw = _datasets(_child(workspace, a.name, restore, privacy=dp), {"demo": {}})
    (b_scenario,) = scenarios(parse_experiment(b_raw))
    b = run_scenario(b_scenario, evaluate=stub_score)

    a_nodes = load_bundle(a / "bundle").snapshot.nodes
    b_nodes = load_bundle(b / "bundle").snapshot.nodes
    others = [n for n in a_nodes if n.startswith("other-")]
    assert others
    for node_id in others:
        np.testing.assert_array_equal(
            b_nodes[node_id].arrays["privacy/rdp"],
            a_nodes[node_id].arrays["privacy/rdp"],
        )
        assert b_nodes[node_id].meta["rng"] == a_nodes[node_id].meta["rng"]
        # what B did not keep of A, it does not pass on either
        held = any(k.startswith("model/") for k in b_nodes[node_id].arrays)
        assert held == (restore is None)


@pytest.mark.parametrize("per_client, refused", [(2, True), (1, False)])
def test_a_new_dp_edge_may_not_hold_subjects_that_already_released(
    workspace: Path, per_client: int, refused: bool
) -> None:
    dp = {"name": "local_dp", "clip": 10.0}
    held = {"test": ["1", "2", "3"], "val": ["4", "5"]}

    def roles(raw: dict, exclude: list[str]) -> dict:
        raw["data"]["roles"] = raw["data"]["roles"] | {
            "subjects_per_client": per_client,
            "overrides": {"demo": held | {"exclude": exclude}},
        }
        return raw

    a = run_scenario(
        *scenarios(parse_experiment(roles(experiment(workspace, privacy=dp), ["12"]))),
        evaluate=stub_score,
    )
    child = roles(_child(workspace, a.name, privacy=dp), [])  # 12 joins: regrouped

    if refused:
        with pytest.raises(ConfigError, match="released"):
            plan(parse_experiment(child))
    else:
        assert plan(parse_experiment(child))


def test_a_continuation_may_change_the_dp_noise(workspace: Path) -> None:
    parent = _parent_run(workspace, privacy={"name": "local_dp", "sigma": 0.5})

    raw = _child(workspace, parent.name, privacy={"name": "local_dp", "sigma": 1.0})
    assert plan(parse_experiment(raw))


def _datasets(raw: dict, datasets: dict) -> dict:
    raw["data"] = raw["data"] | {"datasets": dict.fromkeys(datasets, {})}
    raw["data"]["roles"] = raw["data"]["roles"] | {"overrides": datasets}
    return raw


def test_the_lineage_remembers_a_dataset_a_generation_did_not_load(
    workspace: Path,
) -> None:
    from onion_fl.continuum.bundle import load_bundle

    descriptor = (workspace / "datasets" / "demo.yaml").read_text(encoding="utf-8")
    (workspace / "datasets" / "other.yaml").write_text(
        descriptor.replace("name: demo", "name: other"), encoding="utf-8"
    )
    first = {"demo": {}, "other": {"test": ["11", "12"], "val": []}}
    raw = _datasets(experiment(workspace), first)
    (a_scenario,) = scenarios(parse_experiment(raw))
    a = run_scenario(a_scenario, evaluate=stub_score)
    (b_scenario,) = scenarios(
        parse_experiment(_datasets(_child(workspace, a.name), {"demo": {}}))
    )
    b = run_scenario(b_scenario, evaluate=stub_score)

    a_bundle, b_bundle = load_bundle(a / "bundle"), load_bundle(b / "bundle")
    assert b_bundle.preprocessing["other"] == a_bundle.preprocessing["other"]
    assert b_bundle.schema["other"] == a_bundle.schema["other"]
    assert b_bundle.roles["other"]["train"] == a_bundle.roles["other"]["train"]
    assert "1" in a_bundle.roles["other"]["train"]
    back = {"other": {"test": ["1", "2"], "val": []}}  # trained in A, not in B
    with pytest.raises(ConfigError, match="keeps its role"):
        plan(parse_experiment(_datasets(_child(workspace, b.name), back)))


def test_frozen_local_preprocessing_is_refused(workspace: Path) -> None:
    parent = _parent_run(workspace)
    raw = _child(workspace, parent.name)
    raw["data"]["roles"] = raw["data"]["roles"] | {"scaler": "local"}

    with pytest.raises(ConfigError, match="local preprocessing"):
        plan(parse_experiment(raw))
    raw = _child(workspace, parent.name, {"preprocessing": False})
    raw["data"]["roles"] = raw["data"]["roles"] | {"scaler": "local"}
    assert plan(parse_experiment(raw))


@pytest.mark.parametrize(
    "model",
    [
        {"adapter_width": 8},
        {"trunk_hidden": [4, 4]},
        {"trunk": "per_dataset"},
        {"heads": "per_dataset"},
        {"adapters": "shared"},
    ],
    ids=["adapter_width", "trunk_hidden", "trunk", "heads", "adapters"],
)
def test_a_restored_model_must_keep_its_structure(workspace: Path, model) -> None:
    parent = _parent_run(workspace)
    learning = experiment(workspace)["learning"]
    learning = learning | {"model": learning["model"] | model}

    with pytest.raises(ConfigError, match="structure"):
        plan(parse_experiment(_child(workspace, parent.name, learning=learning)))
    fresh = _child(workspace, parent.name, {"model": False}, learning=learning)
    assert plan(parse_experiment(fresh))


def test_a_restored_model_must_keep_each_datasets_task(workspace: Path) -> None:
    parent = _parent_run(workspace)
    descriptor = workspace / "datasets" / "demo.yaml"
    descriptor.write_text(
        descriptor.read_text(encoding="utf-8").replace("task: stress", "task: mood"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="task"):
        plan(parse_experiment(_child(workspace, parent.name)))


# --- real runs that could not start or end (QA3, #175) ------------------------------


def test_the_root_process_hosts_the_global_validation_evaluators(
    workspace: Path,
) -> None:
    from onion_fl.experiment.real import _group_members
    from onion_fl.experiment.runner import _scenario_data, edge_specs

    raw = experiment(workspace)
    raw["evaluation"] = raw["evaluation"] | {"global": {"every": 1, "subjects": "val"}}
    (scenario,) = scenarios(parse_experiment(raw))
    topology, split, placement, _ = _scenario_data(scenario)
    edges, _ = edge_specs(scenario, topology, split, placement)

    hosted = _group_members(scenario.config, topology, placement, "cloud")
    expected = {spec.id for spec in edges["cloud"]}
    assert expected and expected <= hosted


def test_a_real_run_refuses_links_that_cannot_cross_processes(workspace: Path) -> None:
    from onion_fl.experiment.real import run_real

    topology = TOPOLOGY | {"edge": {"link_up": {"transport": "memory"}}}
    (scenario,) = scenarios(parse_experiment(experiment(workspace, topology=topology)))

    with pytest.raises(ConfigError, match="memory"):
        run_real(scenario)


def test_a_real_run_that_fails_to_launch_is_marked_failed(
    workspace: Path, monkeypatch
) -> None:
    import onion_fl.experiment.real as real

    def broken(*args, **kwargs):
        raise OSError("cannot start a process")

    monkeypatch.setattr(real, "check_brokers", lambda topology: None)
    monkeypatch.setattr(real.subprocess, "Popen", broken)
    (scenario,) = scenarios(parse_experiment(experiment(workspace)))

    with pytest.raises(OSError):
        real.run_real(scenario)
    (run,) = (workspace / "runs").iterdir()
    assert (
        json.loads((run / "run.json").read_text(encoding="utf-8"))["status"] == "failed"
    )


# --- silently wrong configurations (QA2, #174) --------------------------------------


def test_two_entries_cannot_load_the_same_dataset(workspace: Path) -> None:
    raw = experiment(workspace)
    twin = {"descriptor": str(workspace / "datasets" / "demo.yaml")}
    raw["data"] = raw["data"] | {"datasets": {"demo": {}, "demo_again": twin}}

    with pytest.raises(ConfigError, match="demo"):
        plan(parse_experiment(raw))


def _attacked(workspace: Path, placement: dict) -> tuple[list[str], list[str]]:
    from onion_fl.experiment.runner import _scenario_data, edge_specs, malicious_edges

    raw = experiment(workspace, attack={"name": "sign_flip", "fraction": 0.5})
    raw["data"] = raw["data"] | {"placement": placement}
    (scenario,) = scenarios(parse_experiment(raw))
    topology, split, placement_, _ = _scenario_data(scenario)
    edges, _ = edge_specs(scenario, topology, split, placement_)
    attacked = sorted(s.id for group in edges.values() for s in group if s.attack)
    old = sorted(malicious_edges(scenario.config, split.clients, scenario.seed))
    return attacked, old


def test_an_attack_reaches_the_edges_of_a_merging_placement(workspace: Path) -> None:
    attacked, _ = _attacked(workspace, {"name": "pooled"})

    assert attacked and all(edge.endswith("-pooled") for edge in attacked)


def test_an_attack_picks_the_same_edges_as_before_without_merging(
    workspace: Path,
) -> None:
    attacked, old = _attacked(workspace, {"name": "mixing", "alpha": 0.0})

    assert attacked == old and attacked
