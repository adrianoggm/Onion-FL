"""Tests for running a federation on an infrastructure (issue #186, spec I1 §5).

Protocol tests only: edges use the stub trainer, and the experiment tests the
tiny CSV format fixture of ``test_experiment`` (docs/RULES.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onion_fl.core.topology import parse_topology
from onion_fl.experiment.config import ConfigError, parse_experiment
from onion_fl.experiment.runner import _on_machines, _paths, plan, run_scenario
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.assignment import assignments, resolve
from onion_fl.infrastructure.model import (
    InfrastructureError,
    anchor_traces,
    parse_infrastructure,
)
from onion_fl.infrastructure.network import PhysicalNetwork, travel
from onion_fl.infrastructure.paths import hops
from onion_fl.learning.model import state_arrays
from onion_fl.learning.trainers import trainers
from onion_fl.roles import EdgeSpec, build_federation
from onion_fl.runtime.devices import compute_models
from onion_fl.runtime.network import PathChannel, resolve_profile
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


@pytest.mark.parametrize("bandwidth", ["shared", "dedicated"])
def test_an_infrastructure_that_reproduces_the_implicit_one_gives_the_same_run(
    bandwidth: str,
) -> None:
    """One host per node and one link per logical link, with its profile: shared
    and dedicated bandwidth give the run without an infrastructure."""
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

    def run(resolved=None, **kw):
        federation = build_federation(
            topology, edges(), initial_state=INITIAL, rounds=3, seed=7, **kw
        )
        if resolved is not None:  # on the hosts, with their CPUs
            _on_machines(federation.runtime, resolved, topology)
        federation.run()
        return federation

    nodes = ["cloud", "fog_0", "fog_1", *[e for g in leaves.values() for e in g]]
    infra = parse_infrastructure(
        {
            # CPUs but one node per host, no aggregate rate: as today, bit for bit
            "profiles": {
                "server": {"cpus": 1},
                "device": {"train_rate": 50.0, "cpus": 1},
            },
            "sites": {
                n: {"hosts": {n: "device" if n.startswith("e") else "server"}}
                for n in nodes
            },
            "network": {
                "bandwidth": bandwidth,
                "links": [
                    {"between": [f, "cloud"], **_fields("wifi")}
                    for f in ("fog_0", "fog_1")
                ]
                + [
                    {"between": [e, leaf], **_fields("4g")}
                    for leaf, group in leaves.items()
                    for e in group
                ],
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
    _, channels = _paths(resolved)
    assert (channels is None) == (bandwidth == "dedicated")
    compiled = run(
        resolved,
        link_profiles={c: p for c, (_, p) in resolved.links.items()},
        link_channels=channels,
    )

    assert compiled.runtime.events == plain.runtime.events
    assert plain.coordinator.finished
    for key, value in plain.coordinator.state.items():
        np.testing.assert_array_equal(compiled.coordinator.state[key], value)


def test_a_fog_that_combines_twice_the_bytes_takes_twice_as_long() -> None:
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"nodes": [{"id": "fog_0"}, {"id": "fog_1"}]},
        }
    )
    leaves = {"fog_0": ["e0", "e1"], "fog_1": ["e2"]}
    federation = build_federation(
        topology,
        {
            leaf: [
                EdgeSpec(id=e, model=model(A), trainer=trainers.create("stub", {}))
                for e in group
            ]
            for leaf, group in leaves.items()
        },
        initial_state=INITIAL,
        rounds=1,
    )
    infra = parse_infrastructure(
        {
            "profiles": {"fog": {"aggregate_rate": 1000.0}, "box": {}},
            "sites": {"all": {"hosts": {"cloud": "box", "f0": "fog", "f1": "fog"}}},
        }
    )
    resolved = resolve(
        infra,
        topology,
        root="cloud",
        aggregators={"fog_0": "f0", "fog_1": "f1"},
        edges=assignments.create("explicit", {"hosts": {"*": "cloud"}}),
        leaves=leaves,
    )
    _on_machines(federation.runtime, resolved, topology)
    federation.run()

    def busy(fog: str) -> float:
        events = federation.runtime.events
        closed = next(
            e["t"] for e in events if e["node"] == fog and e["name"] == "round.closed"
        )
        sent = next(
            e["t"]
            for e in events
            if e["node"] == fog
            and e["name"] == "message.sent"
            and e["tags"]["kind"] == "update"
        )
        return sent - closed

    sent = sum(np.asarray(v).nbytes for v in state_arrays(model(A)).values())
    assert busy("fog_1") == pytest.approx(sent / 1000.0)  # one update's bytes
    assert busy("fog_0") == pytest.approx(2 * busy("fog_1"))


# --- physical network ---------------------------------------------------------------


def network(**network) -> PhysicalNetwork:
    """field -> fog -> dc, fixed latencies, as lab() with this network."""
    links = [
        {"between": ["field", "fog"], "latency": "10ms", "bandwidth": "100Mbps"},
        {"between": ["fog", "dc"], "latency": "70ms", "bandwidth": "10Mbps"},
    ]
    return PhysicalNetwork(
        parse_infrastructure(lab(network={"links": links} | network))
    )


def channel(net: PhysicalNetwork, src: str, dst: str, seed: int = 0) -> PathChannel:
    _, crossed = hops(net.infra, src, dst)
    return PathChannel(net, travel(crossed, "up"), "up", np.random.default_rng(seed))


def test_a_message_takes_the_latency_plus_its_size_over_the_narrowest_bandwidth() -> (
    None
):
    for mode in ("shared", "dedicated"):
        up = channel(network(bandwidth=mode), "field-pi-1", "cloud-1")
        assert up.schedule(0.0, 1000) == pytest.approx(0.08 + 8000 / 1e7)


@pytest.mark.parametrize("mode, last", [("shared", 20), ("dedicated", 10)])
def test_two_flows_over_one_link_get_half_of_it_each_when_shared(mode, last) -> None:
    """Ten messages of 1250 bytes per flow (1 ms each at 10 Mbps), all sent at 0."""
    net = network(bandwidth=mode)
    a = channel(net, "field-pi-1", "cloud-1")
    b = channel(net, "field-pi-2", "cloud-1")
    arrivals = []
    for _ in range(10):
        a.schedule(0.0, 1250)
        arrivals.append(b.schedule(0.0, 1250))
    # the 100 Mbps field link is ten times faster: the 10 Mbps one is the bottleneck
    assert max(arrivals) - 0.08 == pytest.approx(last * 1e-3, rel=0.06)


def test_a_link_down_loses_what_is_sent_meanwhile_and_comes_back() -> None:
    net = network(
        schedule=[
            {"between": ["fog", "dc"], "at": "1s", "down": True},
            {"between": ["dc", "fog"], "at": "2s", "down": False},
            {"between": ["fog", "dc"], "at": "3s", "set": {"latency": "170ms"}},
        ]
    )
    up = channel(net, "field-pi-1", "cloud-1")
    assert up.schedule(0.5, 0) == pytest.approx(0.58)
    assert up.schedule(1.5, 0) is None
    assert up.schedule(2.5, 0) == pytest.approx(2.58)
    assert up.schedule(3.5, 0) == pytest.approx(3.68)
    names = [name for _, name, _ in net.events()]
    assert names == ["link.down", "link.up", "link.changed"]


def test_a_trace_becomes_one_change_per_row(tmp_path: Path) -> None:
    trace = tmp_path / "wan.csv"
    trace.write_text("t,latency,loss\n10,20ms,0.1\n20,,0.2\n", encoding="utf-8")
    net = network(schedule=[{"between": ["fog", "dc"], "trace": str(trace)}])
    fields, up = net.state("link:fog|dc", 15)
    assert up and (fields.latency, fields.loss) == (pytest.approx(0.02), 0.1)
    fields, _ = net.state("link:fog|dc", 25)
    assert (fields.latency, fields.loss) == (pytest.approx(0.02), 0.2)


@pytest.mark.parametrize("trace", ["../secret.csv", "ABSOLUTE"])
def test_a_trace_outside_the_infrastructures_folder_is_refused(
    tmp_path: Path, trace: str
) -> None:
    folder = tmp_path / "infrastructures"
    folder.mkdir()
    (tmp_path / "secret.csv").write_text("t,loss\n0,0.5\n", encoding="utf-8")
    trace = str(tmp_path / "secret.csv") if trace == "ABSOLUTE" else trace
    raw = {"network": {"schedule": [{"between": ["fog", "dc"], "trace": trace}]}}
    with pytest.raises(InfrastructureError, match="give a path inside"):
        anchor_traces(raw, folder)


def test_a_bad_trace_row_does_not_echo_the_file(tmp_path: Path) -> None:
    (tmp_path / "wan.csv").write_text("t,loss\nsecret,0.1\n", encoding="utf-8")
    raw = lab(
        network={
            "links": [{"between": ["fog", "dc"]}],
            "schedule": [{"between": ["fog", "dc"], "trace": "wan.csv"}],
        }
    )
    net = parse_infrastructure(anchor_traces(raw, tmp_path))
    with pytest.raises(InfrastructureError, match="row 1") as error:
        PhysicalNetwork(net)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"between": ["field", "dc"], "down": True}, "no link between"),
        ({"between": ["fog", "dc"], "down": True, "set": {"loss": 0.1}}, "exactly one"),
    ],
)
def test_a_malformed_schedule_is_refused(change: dict, message: str) -> None:
    with pytest.raises(InfrastructureError, match=message):
        network(schedule=[change])


def test_an_mqtt_link_goes_through_the_broker() -> None:
    infra = parse_infrastructure(lab(services={"mqtt_broker": "cloud-1"}))
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {
                "defaults": {"link_up": {"transport": "memory"}},
                "nodes": [{"id": "fog_0"}],
            },
        }
    )
    resolved = resolve(
        infra,
        topology,
        root="cloud-1",
        aggregators={"fog_0": "fog-1"},
        edges=assignments.create("explicit", {"hosts": {"*": "field-pi-1"}}),
        leaves={"fog_0": ["e0"]},
    )
    # e0 -> fog_0 over mqtt: field -> fog -> dc (the broker) -> fog
    assert resolved.links["e0"][0] == ["field", "fog", "dc", "fog"]
    assert resolved.links["e0"][1]["latency_s"] == pytest.approx(0.01 + 0.07 + 0.07)
    assert resolved.links["fog_0"][0] == ["fog", "dc"]  # memory: no broker


def test_a_scheduled_link_down_drops_the_updates_that_cross_it_until_it_is_back() -> (
    None
):
    """Rounds of 10 s of training; fog_0's uplink is down in [15, 25)."""
    topology = parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud", "deadline": "30s"},
            "fog": {"nodes": [{"id": "fog_0"}, {"id": "fog_1"}]},
        }
    )
    leaves = {"fog_0": ["e0"], "fog_1": ["e1"]}
    infra = parse_infrastructure(
        {
            "profiles": {"server": {}, "device": {"train_rate": 1.0}},
            "sites": {
                "dc": {"hosts": {"cloud": "server"}},
                "f0": {"hosts": {"fog_0": "server", "e0": "device"}},
                "f1": {"hosts": {"fog_1": "server", "e1": "device"}},
            },
            "network": {
                "links": [
                    {"between": ["f0", "dc"], "latency": 0.01},
                    {"between": ["f1", "dc"], "latency": 0.01},
                ],
                "schedule": [
                    {"between": ["f0", "dc"], "at": 15, "down": True},
                    {"between": ["f0", "dc"], "at": 25, "down": False},
                ],
            },
        }
    )
    resolved = resolve(
        infra,
        topology,
        root="cloud",
        aggregators={"fog_0": "fog_0", "fog_1": "fog_1"},
        edges=assignments.create("explicit", {"hosts": {"e0": "e0", "e1": "e1"}}),
        leaves=leaves,
    )

    def run() -> Any:
        net, channels = _paths(resolved)  # a fresh network: queues start empty
        federation = build_federation(
            topology,
            {
                leaf: [
                    EdgeSpec(
                        id=e,
                        model=model(A),
                        trainer=trainers.create("stub", {"examples": 10}),
                        compute=compute_models.create(
                            "samples_per_second", {"samples_per_second": 1.0}
                        ),
                    )
                    for e in group
                ]
                for leaf, group in leaves.items()
            },
            initial_state=INITIAL,
            rounds=3,
            seed=3,
            link_channels=channels,
        )
        _on_machines(federation.runtime, resolved, topology, net)
        federation.run()
        return federation

    events = run().runtime.events
    dropped = [
        e["t"] for e in events if e["name"] == "link.dropped" and e["node"] == "fog_0"
    ]
    back = [
        e["t"]
        for e in events
        if e["name"] == "message.delivered"
        and e["node"] == "cloud"
        and e["tags"]["src"] == "fog_0"
        and e["tags"]["kind"] == "update"
    ]
    assert dropped and all(15 <= t < 25 for t in dropped)
    assert any(t < 15 for t in back) and any(t >= 25 for t in back)
    assert [(e["t"], e["name"]) for e in events if e["node"] == "network"] == [
        (15, "link.down"),
        (25, "link.up"),
    ]
    assert run().runtime.events == events  # the same seed: the same events


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


def test_a_host_without_memory_for_its_nodes_is_refused(
    workspace: Path,  # noqa: F811
) -> None:
    raw = with_infrastructure(workspace)
    raw["infrastructure"]["name"] = lab(
        profiles={"vm": {"train_rate": 100}, "pi": {"memory": "1KB"}}
    )
    with pytest.raises(
        ConfigError, match="host field-pi-1 needs .* MiB for the models"
    ):
        plan(parse_experiment(raw))
