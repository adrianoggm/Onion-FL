"""Tests for the emulated backend (issue #195, spec I1 §10–§11).

Nothing here needs Docker: the Compose project, its setup scripts and the tc
commands are checked as text. The run in containers, against the simulation,
is in ``test_runtime_equivalence``. Edges use the stub trainer on a CSV format
fixture, so nothing is trained for real (docs/RULES.md).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from onion_fl.experiment.backends import EmulatedBackend
from onion_fl.experiment.config import ConfigError, experiment_schema, parse_experiment
from onion_fl.experiment.emulated import NUMERIC, pins
from onion_fl.experiment.runner import plan
from onion_fl.experiment.sweep import scenarios
from onion_fl.infrastructure.backends import Backend, backends
from onion_fl.infrastructure.emulators import (
    compose_project,
    layout,
    netem,
    queue_limit,
    refusals,
)
from onion_fl.infrastructure.model import LinkFields, parse_infrastructure
from onion_fl.runtime.real import RealRuntime
from tests.test_experiment import experiment, workspace  # noqa: F401
from tests.test_infrastructure import lab

HOSTS = ["cloud-1", "fog-1", "fog-2", "field-pi-1", "field-pi-2"]
ROOT = Path(__file__).resolve().parents[1]


def lab_infra(**changes):
    return parse_infrastructure(lab(services={"mqtt_broker": "cloud-1"}, **changes))


def project(infra, hosts=HOSTS, broker="cloud-1"):
    return compose_project(
        layout(infra, hosts, broker), project="onionfl-test", image="img"
    )


# --- the generator -------------------------------------------------------------------


def test_every_host_gets_a_container_and_every_site_a_router() -> None:
    lay = layout(lab_infra(), HOSTS, "cloud-1")

    assert lay.summary() == {
        "containers": 9,
        "hosts": 5,
        "routers": 3,
        "broker": "cloud-1",
        "networks": 5,
    }
    # One /24 per site LAN (sorted), then one per link (in order).
    assert [str(n) for n in lay.lans.values()] == [
        "10.195.0.0/24",
        "10.195.1.0/24",
        "10.195.2.0/24",
    ]
    assert str(lay.nets[("field", "fog")]) == "10.195.3.0/24"
    assert str(lay.addresses["cloud-1"]) == "10.195.0.3"
    assert [str(lay.addresses[h]) for h in ("fog-1", "fog-2")] == [
        "10.195.2.3",
        "10.195.2.4",
    ]
    assert str(lay.routers["fog"]) == "10.195.2.2"


def test_a_link_is_shaped_at_both_ends_each_with_its_direction() -> None:
    links = [
        {
            "between": ["field", "fog"],
            "latency": "10ms",
            "bandwidth_up": "100Mbps",
            "bandwidth_down": "20Mbps",
            "loss": 0.01,
        },
        {"between": ["fog", "dc"], "latency": "70ms", "bandwidth": "10Mbps"},
    ]
    em = project(lab_infra(network={"links": links}))

    up = em.shaping["router-field"]
    down = [c for c in em.shaping["router-fog"] if "10.195.3." in c]
    assert up == [
        "tc qdisc add dev $(dev 10.195.3.2) root netem limit 1083 delay 10000us "
        "rate 100000000bit loss 1%"
    ]
    assert down == [
        "tc qdisc add dev $(dev 10.195.3.3) root netem limit 1017 delay 10000us "
        "rate 20000000bit loss 1%"
    ]


def test_routes_follow_the_path_of_least_latency() -> None:
    em = project(lab_infra())

    # field reaches dc through fog, and dc reaches field through fog.
    assert "ip route add 10.195.0.0/24 via 10.195.3.3" in em.scripts["router-field"]
    assert "ip route add 10.195.1.0/24 via 10.195.4.2" in em.scripts["router-dc"]
    # A host reaches every other site through its site's router.
    assert "ip route add 10.195.0.0/16 via 10.195.1.2" in em.scripts["host-field-pi-1"]


def test_the_lan_is_shaped_only_towards_hosts_of_the_same_site() -> None:
    em = project(lab_infra())

    fog = em.shaping["host-fog-1"]
    assert fog[0].endswith("prio bands 2 priomap " + " ".join(["0"] * 16))
    assert "netem limit 1133 delay 1000us 200us distribution normal" in fog[1]
    assert fog[2].endswith("match ip dst 10.195.2.0/24 flowid 1:2")
    assert em.shaping["host-cloud-1"] == []  # alone in its site


def test_profiles_become_cpu_and_memory_limits() -> None:
    profiles = {"vm": {"cpus": 2, "memory": "1GB"}, "pi": {"cpus": 1}}
    services = project(lab_infra(profiles=profiles)).compose["services"]

    assert services["host-cloud-1"]["cpus"] == 2
    assert services["host-cloud-1"]["mem_limit"] == 2**30
    assert services["host-field-pi-1"]["cpus"] == 1
    assert "mem_limit" not in services["host-field-pi-1"]
    assert all(
        s["cap_add"] == ["NET_ADMIN"] for n, s in services.items() if n != "broker"
    )


def test_the_broker_runs_behind_the_shaping_of_its_host() -> None:
    em = project(lab_infra())

    broker = em.compose["services"]["broker"]
    assert broker["network_mode"] == "service:host-cloud-1"
    assert em.broker == "10.195.0.3:1883"
    assert em.container("field-pi-1") == "onionfl-test-host-field-pi-1"
    assert all(c.startswith("onionfl-test-") for c in em.containers)


def test_a_run_on_one_site_needs_no_router() -> None:
    em = project(lab_infra(), hosts=["fog-1", "fog-2"], broker="fog-1")

    assert em.layout.containers == 3  # two hosts and the broker
    assert not any("ip route" in s for s in em.scripts.values())


def test_the_queue_holds_what_is_in_flight_plus_netems_default() -> None:
    # 500 frames of 1514 bytes are in flight on 12.112 Mbit/s over 500 ms.
    assert queue_limit(LinkFields(latency=0.5), 12_112_000) == 1500
    assert queue_limit(LinkFields(latency=0.001), None) == 1083  # as 1 Gbit/s
    lossless = LinkFields(latency=0.002, jitter=0.0005, distribution="fixed")
    assert netem(lossless, None) == "netem limit 1166 delay 2000us"  # no jitter


def test_what_netem_cannot_reproduce_is_refused() -> None:
    lognormal = {
        "between": ["field", "fog"],
        "latency": "10ms",
        "jitter": "2ms",
        "distribution": "lognormal",
    }
    infra = lab_infra(
        network={
            "links": [lognormal, {"between": ["fog", "dc"], "latency": "70ms"}],
            "bandwidth": "dedicated",
            "schedule": [{"between": ["fog", "dc"], "at": "5s", "down": True}],
        }
    )

    reasons = " ".join(refusals(infra))
    assert "network.schedule" in reasons and "dedicated" in reasons
    assert "network.links[0]: netem has no lognormal delay" in reasons
    assert refusals(lab_infra()) == []


# --- the backend -----------------------------------------------------------------------


def on_lab(workspace: Path, **backend) -> dict:  # noqa: F811
    return experiment(
        workspace,
        backend={"name": "emulated", "project": "onionfl-test", **backend},
        infrastructure={
            "name": lab(),
            "assignment": {
                "root": "cloud-1",
                "aggregators": {"fog_a": "fog-1", "fog_b": "fog-2"},
                "edges": {"name": "colocated", "per_host": 3},
                "services": {"mqtt_broker": "cloud-1"},
            },
        },
    )


def test_the_emulated_backend_is_registered_and_exported() -> None:
    assert "emulated" in backends.names()
    names = {p["name"] for p in experiment_schema()["plugins"]["backend"]}
    assert "emulated" in names
    assert isinstance(EmulatedBackend(), Backend)


def test_the_emulated_backend_needs_an_infrastructure(
    workspace: Path,  # noqa: F811
) -> None:
    with pytest.raises(ConfigError, match="give the experiment an infrastructure"):
        parse_experiment(experiment(workspace, backend="emulated"))


def test_plan_shows_the_containers_and_checks_their_cap(
    workspace: Path,  # noqa: F811
) -> None:
    (preview,) = plan(parse_experiment(on_lab(workspace)))

    shown = preview["backend"]
    assert shown["name"] == "emulated" and shown["max_containers"] == 64
    assert shown["containers"] == shown["hosts"] + shown["routers"] + 1
    with pytest.raises(ConfigError, match="exceed max_containers 3"):
        plan(parse_experiment(on_lab(workspace, max_containers=3)))


def test_prepare_writes_the_project_and_the_paths_the_containers_see(
    workspace: Path,  # noqa: F811
) -> None:
    config = parse_experiment(on_lab(workspace))
    (scenario,) = scenarios(config)

    prepared = EmulatedBackend(project="onionfl-test").prepare(scenario)

    folder = prepared.folder
    compose = yaml.safe_load((folder / "compose.yaml").read_text(encoding="utf-8"))
    assert compose == prepared.emulation.compose
    for script in folder.glob("*.sh"):  # run by sh in Linux: LF only
        assert b"\r" not in script.read_bytes()
    document = json.loads((prepared.run.path / "scenario.json").read_text("utf-8"))
    groups = document["groups"]
    assert sorted(groups) == prepared.groups
    assert "cloud" in groups["cloud-1"] and "fog_a" in groups["fog-1"]
    nodes = [n for members in groups.values() for n in members]
    assert len(nodes) == len(set(nodes))  # every node in one place
    assert any(n.startswith("test-") for n in groups["cloud-1"])  # root evaluators
    assert document["broker"] == prepared.emulation.broker
    assert (prepared.run.path / document["initial"]).is_file()  # the launcher's bits
    assert all(
        p.startswith("/onionfl/paths/") for p in document["config"]["paths"].values()
    )
    volumes = compose["services"]["host-cloud-1"]["volumes"]
    writable = [v["target"] for v in volumes if not v["read_only"]]
    assert writable == ["/onionfl/paths/runs"]
    assert prepared.command("cloud-1")[:3] == [
        "docker",
        "exec",
        "onionfl-test-" + prepared.run.run_id.rsplit("-", 1)[-1] + "-host-cloud-1",
    ]
    meta = prepared.run.meta
    assert meta["backend"] == "emulated" and meta["infrastructure_id"]
    assert meta["emulation"]["containers"] == prepared.emulation.layout.containers
    prepared.run.finish(status="failed")  # nothing ran


def test_a_node_process_sees_the_launchers_tracing(monkeypatch) -> None:
    """``docker exec`` passes no environment on, so the command names it (#198)."""
    from types import SimpleNamespace

    from onion_fl.experiment.emulated import EmulatedPlan

    plan = EmulatedPlan(
        run=SimpleNamespace(run_id="20261110T000000Z-abc"),
        scenario=None,
        groups=["cloud-1"],
        emulation=SimpleNamespace(container=lambda host: f"p-host-{host}"),
        folder=Path("."),
        params=None,
    )
    monkeypatch.delenv("ONIONFL_TRACE", raising=False)
    assert plan.command("cloud-1")[:3] == ["docker", "exec", "p-host-cloud-1"]
    monkeypatch.setenv("ONIONFL_TRACE", "1")
    assert plan.command("cloud-1")[:5] == [
        "docker",
        "exec",
        "-e",
        "ONIONFL_TRACE=1",
        "p-host-cloud-1",
    ]


def test_prepare_pins_each_host_to_the_cpus_it_was_calibrated_on(
    workspace: Path,  # noqa: F811
) -> None:
    config = parse_experiment(on_lab(workspace))
    (scenario,) = scenarios(config)
    hosts = EmulatedBackend(project="onionfl-test").prepare(scenario)
    used = sorted(hosts.emulation.layout.hosts)
    hosts.run.finish(status="failed")
    # a mapping keeps each host's CPUs, whatever other hosts the run leaves out
    by_host = {h: [2 * i + 2] for i, h in enumerate(used)} | {"unused": [40]}
    pin = {"cpus": by_host, "others": "20-23"}

    prepared = EmulatedBackend(project="onionfl-test", pin=pin).prepare(scenario)

    services = prepared.emulation.compose["services"]
    lay = prepared.emulation.layout
    assert {h: services[lay.services[h]]["cpuset"] for h in used} == {
        h: str(c[0]) for h, c in by_host.items() if h in used
    }
    assert services["broker"]["cpuset"] == "20-23"
    assert prepared.run.meta["emulation"]["cpusets"]["broker"] == "20-23"
    prepared.run.finish(status="failed")  # nothing ran
    short = {"cpus": {used[0]: [2]}, "others": "20"}
    with pytest.raises(ConfigError, match="pin gives it 0"):
        EmulatedBackend(project="onionfl-test", pin=short).prepare(scenario)


def test_the_image_takes_the_ranges_of_pyproject() -> None:
    dockerfile = (
        ROOT / "src" / "onion_fl" / "infrastructure" / "emulated.Dockerfile"
    ).read_text(encoding="utf-8")
    ranges = set(re.findall(r'"([a-z][a-z0-9-]*[<>=][^"]*)"', dockerfile))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    wanted = {
        d.split("  ")[0].replace(" ", "") for d in project["project"]["dependencies"]
    }
    assert wanted <= ranges | {r.replace(" ", "") for r in ranges}
    assert set(pins()) == set(NUMERIC)


def test_a_real_runtime_can_point_every_mqtt_link_at_one_broker() -> None:
    runtime = RealRuntime("r", broker="10.195.0.3:1883", heartbeat_s=None)
    runtime.add_link("a", "b", transport={"name": "mqtt", "broker": "localhost:1883"})
    runtime.add_link("b", "c", transport="mqtt")

    (carrier,) = runtime._transports.values()
    assert (carrier.host, carrier.port) == ("10.195.0.3", 1883)
