"""Tests for the infrastructure core: model, paths, assignment and its import
boundary (issue #186, spec I1 §5, §6). Nothing is trained here."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onion_fl.core.registry import PluginError
from onion_fl.core.topology import parse_topology
from onion_fl.infrastructure.assignment import assignments, largest_remainder, resolve
from onion_fl.infrastructure.model import (
    InfrastructureError,
    load_infrastructure,
    parse_infrastructure,
    quantity,
)
from onion_fl.infrastructure.paths import LOOPBACK, effective

SRC = Path(__file__).resolve().parents[1] / "src"


def lab(**changes) -> dict:
    return {
        "name": "lab",
        "profiles": {"vm": {"train_rate": 100}, "pi": {"train_rate": 10}},
        "sites": {
            "dc": {"hosts": {"cloud-1": "vm"}},
            "fog": {"hosts": {"fog-1": "vm", "fog-2": "vm"}},
            "field": {"pool": {"pi": 3}},
        },
        "network": {
            "links": [
                {
                    "between": ["field", "fog"],
                    "latency": "10ms",
                    "bandwidth": "100Mbps",
                },
                {"between": ["fog", "dc"], "latency": "70ms", "bandwidth": "10Mbps"},
            ]
        },
    } | changes


# --- model ---------------------------------------------------------------------------


def test_quantities_take_units() -> None:
    assert quantity("10ms", "duration") == pytest.approx(0.01)
    assert quantity("100Mbps", "bandwidth") == 1e8
    assert quantity("16GB", "memory") == 16 * 2**30
    assert quantity(0.5, "duration") == 0.5
    with pytest.raises(ValueError, match="duration such as"):
        quantity("10 parsecs", "duration")


def test_a_pool_expands_into_named_hosts() -> None:
    infra = parse_infrastructure(lab())
    assert infra.hosts()["field-pi-3"] == ("field", "pi")
    assert infra.profile_of("field-pi-1").train_rate == 10


@pytest.mark.parametrize(
    "change, message",
    [
        ({"sites": {"a": {"hosts": {"h": "gpu"}}}}, "unknown profile 'gpu'"),
        (
            {"sites": {"a": {"hosts": {"h": "vm"}}, "b": {"hosts": {"h": "vm"}}}},
            "host 'h' is in sites",
        ),
        ({"network": {"links": [{"between": ["dc", "mars"]}]}}, "unknown site 'mars'"),
        (
            {
                "network": {
                    "links": [{"between": ["dc", "fog"]}, {"between": ["fog", "dc"]}]
                }
            },
            "linked twice",
        ),
        ({"services": {"mqtt_broker": "nowhere"}}, "unknown host 'nowhere'"),
    ],
)
def test_malformed_infrastructures_say_where(change: dict, message: str) -> None:
    with pytest.raises(InfrastructureError, match=message):
        parse_infrastructure(lab(**change))


def test_the_id_ignores_the_name_and_the_link_order() -> None:
    base = parse_infrastructure(lab())
    links = lab()["network"]["links"]
    same = parse_infrastructure(lab(name="other", network={"links": links[::-1]}))
    slower = [links[0], links[1] | {"latency": "80ms"}]
    assert same.infrastructure_id == base.infrastructure_id
    assert (
        parse_infrastructure(lab(network={"links": slower})).infrastructure_id
        != base.infrastructure_id
    )


def test_the_example_infrastructure_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    infra = load_infrastructure(root / "infrastructures" / "edge_fog_cloud_lab.yaml")
    assert len(infra.hosts()) == 43 and infra.services == {"mqtt_broker": "cloud-1"}


# --- paths ---------------------------------------------------------------------------


def test_paths_on_one_host_one_site_and_across_sites() -> None:
    infra = parse_infrastructure(lab())

    assert effective(infra, "fog-1", "fog-1") == (["fog-1"], LOOPBACK)
    path, lan = effective(infra, "fog-1", "fog-2")
    assert path == ["fog"] and lan["latency_s"] == 0.001

    path, profile = effective(infra, "field-pi-1", "cloud-1")
    assert path == ["field", "fog", "dc"]
    assert profile["latency_s"] == pytest.approx(0.08)
    assert profile["bandwidth_up_bps"] == profile["bandwidth_down_bps"] == 1e7


def test_a_path_composes_losses_jitter_and_each_directions_bandwidth() -> None:
    links = [
        {"between": ["field", "fog"], "loss": 0.1, "jitter": 3}
        | {"bandwidth_up": 5, "bandwidth_down": 7},
        # listed from dc to fog: going field -> dc crosses it against its up
        {"between": ["dc", "fog"], "loss": 0.2, "jitter": 4}
        | {"bandwidth_up": 9, "bandwidth_down": 2},
    ]
    infra = parse_infrastructure(lab(network={"links": links}))

    _, up = effective(infra, "field-pi-1", "cloud-1")
    _, down = effective(infra, "cloud-1", "field-pi-1")

    assert up["loss"] == pytest.approx(1 - 0.9 * 0.8)
    assert up["jitter_s"] == pytest.approx(5.0)
    assert (up["bandwidth_up_bps"], up["bandwidth_down_bps"]) == (2, 7)
    assert (down["bandwidth_up_bps"], down["bandwidth_down_bps"]) == (7, 2)


def test_a_one_segment_path_keeps_the_segment_exactly() -> None:
    fields = {
        "latency": 0.005,
        "jitter": 0.002,
        "distribution": "normal",
        "loss": 0.005,
    }
    infra = parse_infrastructure(
        lab(network={"links": [{"between": ["field", "fog"], **fields}]})
    )
    _, profile = effective(infra, "field-pi-1", "fog-1")
    assert profile["loss"] == 0.005 and profile["distribution"] == "normal"


def test_the_path_of_least_latency_wins_and_a_missing_one_is_refused() -> None:
    links = [
        {"between": ["field", "dc"], "latency": 1.0},
        {"between": ["field", "fog"], "latency": 0.1},
        {"between": ["fog", "dc"], "latency": 0.1},
    ]
    infra = parse_infrastructure(lab(network={"links": links}))
    assert effective(infra, "field-pi-1", "cloud-1")[0] == ["field", "fog", "dc"]

    island = parse_infrastructure(lab(network={"links": links[1:2]}))
    with pytest.raises(InfrastructureError, match="no physical path"):
        effective(island, "field-pi-1", "cloud-1")


# --- assignment ----------------------------------------------------------------------


def tree():
    return parse_topology(
        {
            "levels": ["global", "fog", "edge"],
            "root": {"id": "cloud"},
            "fog": {"nodes": [{"id": "fog_0"}, {"id": "fog_1"}]},
        }
    )


def test_an_explicit_assignment_takes_names_and_patterns_in_order() -> None:
    edges = assignments.create(
        "explicit", {"hosts": {"e1": "field-pi-1", "e*": "field-pi-2"}}
    )
    resolved = resolve(
        parse_infrastructure(lab()),
        tree(),
        root="cloud-1",
        aggregators={"fog_0": "fog-1", "fog_1": "fog-2"},
        edges=edges,
        leaves={"fog_0": ["e1", "e2"], "fog_1": ["e3"]},
    )
    assert resolved.hosts["e1"] == "field-pi-1"
    assert resolved.hosts["e3"] == "field-pi-2"
    assert resolved.parents["e3"] == "fog_1" and resolved.train_rate("e1") == 10
    assert set(resolved.links) == {"fog_0", "fog_1", "e1", "e2", "e3"}


@pytest.mark.parametrize(
    "aggregators, hosts, message",
    [
        ({"fog_0": "fog-1"}, {"*": "fog-1"}, r"no host for \['fog_1'\]"),
        ({"fog_0": "fog-1", "fog_1": "fog-9"}, {"*": "fog-1"}, "unknown host 'fog-9'"),
        ({"fog_0": "fog-1", "fog_1": "fog-1"}, {"x": "fog-1"}, "no host for edges"),
    ],
)
def test_an_incomplete_assignment_is_refused(aggregators, hosts, message) -> None:
    with pytest.raises(InfrastructureError, match=message):
        resolve(
            parse_infrastructure(lab()),
            tree(),
            root="cloud-1",
            aggregators=aggregators,
            edges=assignments.create("explicit", {"hosts": hosts}),
            leaves={"fog_0": ["e1"]},
        )


# --- strategies ----------------------------------------------------------------------


def field(**pool) -> object:
    sites = lab()["sites"] | {"field": {"pool": pool}, "lab": {"pool": {"vm": 2}}}
    return parse_infrastructure(lab(sites=sites))


EDGES = {f"e{i}": "swell" if i % 2 else "wesad" for i in range(10)}


def assign(name: str, infra=None, edges=EDGES, **params) -> dict[str, str]:
    infra = infra or field(pi=6, vm=4)
    return assignments.create(name, params).assign(edges, infra)


def test_largest_remainder_rounds_deterministically() -> None:
    assert largest_remainder({"a": 0.5, "b": 0.3, "c": 0.2}, 10) == {
        "a": 5,
        "b": 3,
        "c": 2,
    }
    assert largest_remainder({"b": 0.5, "a": 0.5}, 7) == {"a": 4, "b": 3}


def test_a_profile_mix_keeps_its_fractions_on_distinct_devices() -> None:
    hosts = assign("profile_mix", mix={"pi": 0.6, "vm": 0.4}, sites=["field"])
    assert len(set(hosts.values())) == 10
    assert sorted(h.split("-")[1] for h in hosts.values()) == ["pi"] * 6 + ["vm"] * 4


def test_a_strategy_depends_on_its_own_seed_only() -> None:
    first = assign("one_per_device", seed=1)
    assert assign("one_per_device", seed=1) == first
    assert assign("one_per_device", seed=2) != first
    assert len(set(first.values())) == 10


def test_by_dataset_and_colocated() -> None:
    hosts = assign("by_dataset", datasets={"swell": "pi"}, default="vm")
    assert {h.split("-")[1] for e, h in hosts.items() if EDGES[e] == "swell"} == {"pi"}
    assert {h.split("-")[1] for e, h in hosts.items() if EDGES[e] == "wesad"} == {"vm"}

    packed = assign("colocated", per_host=4)
    assert sorted(list(packed.values()).count(h) for h in set(packed.values())) == [
        2,
        4,
        4,
    ]


@pytest.mark.parametrize(
    "name, params, message",
    [
        ("one_per_device", {"sites": ["lab"]}, "10 edges need a pool device each"),
        ("one_per_device", {"sites": ["moon"]}, "unknown sites"),
        ("profile_mix", {"mix": {"pi": 0.9, "gpu": 0.1}}, "need a gpu device each"),
        ("by_dataset", {"datasets": {"swell": "pi"}}, "no profile for dataset 'wesad'"),
        ("colocated", {"per_host": 1, "sites": ["lab"]}, "need 10 devices"),
    ],
)
def test_an_impossible_assignment_says_why(name, params, message) -> None:
    with pytest.raises(InfrastructureError, match=message):
        assign(name, **params)


def test_mix_fractions_must_add_up_to_one() -> None:
    with pytest.raises(PluginError, match="add up to 0.9"):
        assignments.create("profile_mix", {"mix": {"pi": 0.5, "vm": 0.4}})


def test_services_can_move_to_another_host() -> None:
    infra = parse_infrastructure(lab(services={"mqtt_broker": "cloud-1"}))
    common = {
        "root": "cloud-1",
        "aggregators": {"fog_0": "fog-1", "fog_1": "fog-2"},
        "edges": assignments.create("explicit", {"hosts": {"*": "fog-1"}}),
        "leaves": {"fog_0": ["e1"]},
    }
    assert resolve(infra, tree(), **common).services == {"mqtt_broker": "cloud-1"}
    moved = resolve(infra, tree(), services={"mqtt_broker": "fog-2"}, **common)
    assert moved.services == {"mqtt_broker": "fog-2"}
    assert moved.load()["fog-1"] == ["e1", "fog_0"]
    with pytest.raises(InfrastructureError, match="mqtt_broker goes to unknown"):
        resolve(infra, tree(), services={"mqtt_broker": "moon"}, **common)


# --- boundaries ----------------------------------------------------------------------

FORBIDDEN = (
    "onion_fl.learning",
    "onion_fl.roles",
    "onion_fl.experiment",
    "onion_fl.data",
)


def _imports(module: str) -> set[str]:
    parts = module.split(".")
    path = SRC.joinpath(*parts)
    path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
            found |= {f"{node.module}.{alias.name}" for alias in node.names}
    return {
        m
        for m in found
        if m.startswith("onion_fl")
        and (
            SRC.joinpath(*m.split(".")).is_dir()
            or SRC.joinpath(*m.split(".")).with_suffix(".py").exists()
        )
    }


def reached(roots: list[str]) -> set[str]:
    """Every onion_fl module that ``roots`` import, directly or not."""
    seen, todo = set(), list(roots)
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo += _imports(module)
    return seen


def test_the_infrastructure_core_depends_on_core_only() -> None:
    roots = [
        f"onion_fl.infrastructure.{p.stem}"
        for p in (SRC / "onion_fl" / "infrastructure").glob("*.py")
    ]
    bad = sorted(m for m in reached(roots) if m.startswith(FORBIDDEN))
    assert not bad, f"onion_fl.infrastructure reaches {bad}"


def test_the_boundary_check_sees_a_forbidden_import() -> None:
    assert any(
        m.startswith("onion_fl.learning")
        for m in reached(["onion_fl.experiment.config"])
    )


def test_assignment_is_an_axis_of_the_schema() -> None:
    from onion_fl.experiment.config import experiment_schema

    assert "explicit" in {
        p["name"] for p in experiment_schema()["plugins"]["assignment"]
    }
