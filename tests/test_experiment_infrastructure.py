"""Tests for running a federation on an infrastructure (issue #186, spec I1 §5).

Protocol tests only: edges use the stub trainer, and the experiment tests the
tiny CSV format fixture of ``test_experiment`` (docs/RULES.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from onion_fl.core.topology import parse_topology
from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import plan, run_scenario
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.assignment import assignments, resolve
from onion_fl.infrastructure.model import parse_infrastructure
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.runtime.devices import compute_models
from onion_fl.runtime.network import resolve_profile
from tests.test_experiment import experiment, stub_score, workspace  # noqa: F401
from tests.test_infrastructure import lab
from tests.test_roles_protocol import INITIAL, A, model

# --- the implicit infrastructure gives the same run ----------------------------------


def _fields(preset: str) -> dict:
    p = resolve_profile(preset)
    return {
        "latency": p.latency_s,
        "jitter": p.jitter_s,
        "distribution": p.distribution,
        "bandwidth_up": p.bandwidth_up_bps,
        "bandwidth_down": p.bandwidth_down_bps,
        "loss": p.loss,
    }


def test_an_infrastructure_that_reproduces_the_implicit_one_gives_the_same_run() -> (
    None
):
    """One host per node and one link per logical link, with its profile."""
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", "deadline": "60s"},
            "fog": {
                "defaults": {"deadline": "30s", "link_up": {"profile": "wifi"}},
                "nodes": [{"id": "fog_0"}, {"id": "fog_1"}],
            },
            "edge": {"link_up": {"profile": "4g"}},
        }
    )
    leaves = {"fog_0": ["e0", "e1"], "fog_1": ["e2", "e3"]}

    def edges() -> dict:  # each run gets fresh models, compute at 50 samples/s
        def spec(edge: str) -> EdgeSpec:
            i = int(edge[1:])
            return EdgeSpec(
                id=edge,
                model=model(A),
                trainer=trainers.create("stub", {"shift": i, "examples": 10 + i}),
                compute=compute_models.create(
                    "samples_per_second", {"samples_per_second": 50.0}
                ),
            )

        return {leaf: [spec(e) for e in group] for leaf, group in leaves.items()}

    def run(**kw):
        federation = build_federation(
            topology, edges(), initial_state=INITIAL, rounds=3, seed=7, **kw
        )
        federation.run()
        return federation

    nodes = ["cloud", "fog_0", "fog_1", *[e for g in leaves.values() for e in g]]
    infra = parse_infrastructure(
        {
            "profiles": {"server": {}, "device": {"train_rate": 50.0}},
            "sites": {
                n: {"hosts": {n: "device" if n.startswith("e") else "server"}}
                for n in nodes
            },
            "network": {
                "links": [
                    {"between": [f, "cloud"], **_fields("wifi")}
                    for f in ("fog_0", "fog_1")
                ]
                + [
                    {"between": [e, leaf], **_fields("4g")}
                    for leaf, group in leaves.items()
                    for e in group
                ]
            },
        }
    )
    resolved = resolve(
        infra,
        topology,
        root="cloud",
        aggregators={"fog_0": "fog_0", "fog_1": "fog_1"},
        edges=assignments.create("explicit", {"hosts": {e: e for e in nodes[3:]}}),
        leaves=leaves,
    )
    assert {resolved.train_rate(e) for e in nodes[3:]} == {50.0}

    plain = run()
    compiled = run(link_profiles={c: p for c, (_, p) in resolved.links.items()})

    assert compiled.runtime.events == plain.runtime.events
    assert plain.coordinator.finished
    for key, value in plain.coordinator.state.items():
        np.testing.assert_array_equal(compiled.coordinator.state[key], value)


# --- experiment ----------------------------------------------------------------------


def with_infrastructure(workspace: Path, **changes) -> dict:  # noqa: F811
    return experiment(
        workspace,
        infrastructure={
            "name": lab(),
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-1", "fog_b": "fog-2"},
                "edges": {"name": "explicit", "hosts": {"*": "field-pi-1"}},
            },
        },
        **changes,
    )


def test_without_an_infrastructure_the_config_and_its_id_do_not_change(
    workspace: Path,  # noqa: F811
) -> None:
    dumped = parse_experiment(experiment(workspace)).dump()
    assert "infrastructure" not in dumped
    assert "infrastructures" not in dumped["paths"]


def test_a_run_on_an_infrastructure_signs_it_and_plan_shows_the_hosts(
    workspace: Path,  # noqa: F811
) -> None:
    config = parse_experiment(with_infrastructure(workspace))
    (preview,) = plan(config)
    infra = preview["infrastructure"]
    edges = [n for n in infra["hosts"] if n not in ("cloud", "fog_a", "fog_b")]
    assert edges and {infra["hosts"][e] for e in edges} == {"field-pi-1"}
    assert infra["links"]["fog_a"]["path"] == ["fog", "dc"]

    (scenario,) = scenarios(config)
    meta = json.loads(
        (run_scenario(scenario, evaluate=stub_score) / "run.json").read_text("utf-8")
    )
    assert meta["backend"] == "sim" and meta["status"] == "finished"
    assert meta["infrastructure_id"] == parse_infrastructure(lab()).infrastructure_id

    (plain,) = scenarios(parse_experiment(experiment(workspace)))
    meta = json.loads(
        (run_scenario(plain, evaluate=stub_score) / "run.json").read_text("utf-8")
    )
    assert meta["backend"] == "sim" and meta["infrastructure_id"] is None


def test_the_infrastructure_is_found_by_name_in_its_folder(
    workspace: Path,  # noqa: F811
) -> None:
    import yaml

    folder = workspace / "infrastructures"
    folder.mkdir()
    (folder / "lab.yaml").write_text(yaml.safe_dump(lab()), encoding="utf-8")
    raw = with_infrastructure(workspace)
    raw["infrastructure"]["name"] = "lab"
    raw["paths"]["infrastructures"] = str(folder)
    (preview,) = plan(parse_experiment(raw))
    assert preview["infrastructure"]["name"] == "lab"


@pytest.mark.parametrize(
    "topology_change, message",
    [
        ({"edge": {"link_up": {"profile": "4g"}}}, r"link_up.profile at \['edge'\]"),
        ({"edge": {"device": {"samples_per_second": 10}}}, "edge.device"),
    ],
)
def test_profiles_in_the_topology_are_refused_with_an_infrastructure(
    workspace: Path,  # noqa: F811
    topology_change: dict,
    message: str,
) -> None:
    from tests.test_experiment import TOPOLOGY

    raw = with_infrastructure(workspace, topology=TOPOLOGY | topology_change)
    with pytest.raises(ConfigError, match=message):
        plan(parse_experiment(raw))


def test_an_infrastructure_is_refused_in_a_real_run(
    workspace: Path,  # noqa: F811
) -> None:
    with pytest.raises(ConfigError, match="simulated only for now"):
        parse_experiment(with_infrastructure(workspace, runtime={"mode": "real"}))


def test_an_assignment_to_a_missing_host_says_so_in_plan(
    workspace: Path,  # noqa: F811
) -> None:
    raw = with_infrastructure(workspace)
    raw["infrastructure"]["assignment"]["root"] = "moon-1"
    with pytest.raises(ConfigError, match="infrastructure: assignment: cloud"):
        plan(parse_experiment(raw))


def test_the_assignment_seed_moves_edges_but_never_their_data(
    workspace: Path,  # noqa: F811
) -> None:
    def preview(seed: int) -> dict:
        raw = with_infrastructure(workspace)
        raw["infrastructure"]["name"] = lab(
            sites=lab()["sites"] | {"field": {"pool": {"pi": 12}}}
        )
        raw["infrastructure"]["assignment"]["edges"] = {
            "name": "one_per_device",
            "seed": seed,
        }
        (out,) = plan(parse_experiment(raw))
        return out

    a, b, again = preview(0), preview(1), preview(0)
    for key in ("roles", "composition", "data"):
        assert a[key] == b[key]
    assert a["infrastructure"]["hosts"] == again["infrastructure"]["hosts"]
    assert a["infrastructure"]["hosts"] != b["infrastructure"]["hosts"]
    assert a["config_id"] != b["config_id"]


def test_the_assignment_can_be_swept(workspace: Path) -> None:  # noqa: F811
    raw = with_infrastructure(
        workspace,
        sweep={"infrastructure.assignment.services.mqtt_broker": ["fog-1", "fog-2"]},
    )
    found = scenarios(parse_experiment(raw))
    assert len({s.config_id for s in found}) == 2
    (preview,) = plan(parse_experiment(with_infrastructure(workspace)))
    assert preview["infrastructure"]["load"]["field-pi-1"]
