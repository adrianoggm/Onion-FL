"""Tests for the emulated backend (issue #195, spec I1 §10–§11).

Nothing here needs Docker: the Compose project, its setup scripts and the tc
commands are checked as text.
"""

from __future__ import annotations

from onion_fl.infrastructure.emulators import (
    compose_project,
    layout,
    netem,
    queue_limit,
    refusals,
)
from onion_fl.infrastructure.model import LinkFields, parse_infrastructure
from tests.test_infrastructure import lab

HOSTS = ["cloud-1", "fog-1", "fog-2", "field-pi-1", "field-pi-2"]


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
