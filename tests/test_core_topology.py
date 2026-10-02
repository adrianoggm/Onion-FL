"""Tests for the topology model, topology_id and to_graph (issue #80)."""

from __future__ import annotations

import copy
import json

import pytest

from onion_fl.core.topology import (
    Topology,
    TopologyError,
    load_topology,
    parse_topology,
)


def four_fogs() -> dict:
    return {
        "name": "four-fogs",
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud", "aggregator": "fedavg", "server_optimizer": "replace"},
        "fog": {
            "defaults": {
                "aggregator": "fedavg",
                "quorum": 1.0,
                "deadline": "30s",
                "link_up": {"transport": "mqtt", "profile": "wifi"},
            },
            "nodes": [
                {"id": "fog_a1", "home": "swell"},
                {"id": "fog_a2", "home": "swell"},
                {"id": "fog_b1", "home": "sweet"},
                {"id": "fog_b2", "home": "sweet", "quorum": 0.8},
            ],
        },
        "edge": {
            "link_up": {"transport": "mqtt", "profile": "4g"},
            "device": {"samples_per_second": 2000},
        },
    }


def regions() -> dict:
    return {
        "name": "regions",
        "levels": ["global", "region", "fog", "edge"],
        "root": {"id": "cloud"},
        "region": {"nodes": [{"id": "north"}, {"id": "south"}]},
        "fog": {
            "defaults": {"link_up": {"profile": "lan"}},
            "nodes": [
                {"id": "fog_1", "parent": "north"},
                {"id": "fog_2", "parent": "north"},
                {"id": "fog_3", "parent": "south"},
            ],
        },
        "edge": {"link_up": {"profile": "4g"}},
    }


def as_general(topology: Topology) -> dict:
    return {
        "name": topology.name,
        "levels": list(topology.levels),
        "nodes": [node.model_dump() for node in topology.nodes],
        "edge": topology.edge.model_dump(),
    }


# --- compact form -----------------------------------------------------------


def test_compact_form_builds_root_and_aggregators() -> None:
    topo = parse_topology(four_fogs())

    assert topo.root.id == "cloud"
    assert topo.root.parent is None
    assert topo.role("cloud") == "coordinator"
    assert [n.id for n in topo.children("cloud")] == [
        "fog_a1",
        "fog_a2",
        "fog_b1",
        "fog_b2",
    ]
    assert all(topo.role(n.id) == "aggregator" for n in topo.children("cloud"))


def test_level_defaults_merge_with_node_overrides() -> None:
    topo = parse_topology(four_fogs())

    assert topo.node("fog_a1").settings["quorum"] == 1.0
    assert topo.node("fog_b2").settings["quorum"] == 0.8
    assert topo.node("fog_b2").settings["home"] == "sweet"
    assert topo.node("fog_a1").link_up.profile == "wifi"
    assert topo.node("fog_a1").link_up.codec == "json"  # default


def test_node_link_overrides_merge_key_by_key() -> None:
    raw = four_fogs()
    raw["fog"]["nodes"][0]["link_up"] = {"profile": "lora"}

    link = parse_topology(raw).node("fog_a1").link_up

    assert link.profile == "lora"
    assert link.transport == "mqtt"  # kept from the level defaults


def test_root_settings_and_edge_template_are_kept() -> None:
    topo = parse_topology(four_fogs())

    assert topo.root.settings == {"aggregator": "fedavg", "server_optimizer": "replace"}
    assert topo.edge.link_up.profile == "4g"
    assert topo.edge.settings == {"device": {"samples_per_second": 2000}}


def test_deeper_trees_use_explicit_parents() -> None:
    topo = parse_topology(regions())

    assert [n.id for n in topo.children("north")] == ["fog_1", "fog_2"]
    assert topo.parent("fog_3").id == "south"
    assert [n.id for n in topo.leaves()] == ["fog_1", "fog_2", "fog_3"]


def test_parent_is_required_when_the_level_above_is_ambiguous() -> None:
    raw = regions()
    del raw["fog"]["nodes"][2]["parent"]

    with pytest.raises(TopologyError, match=r"fog\.nodes\[2\]\.parent"):
        parse_topology(raw)


def test_unknown_top_level_keys_are_rejected() -> None:
    raw = four_fogs()
    raw["fogs"] = raw.pop("fog")

    with pytest.raises(TopologyError, match="fogs"):
        parse_topology(raw)


# --- general form -----------------------------------------------------------


def test_general_form_is_equivalent_to_the_compact_one() -> None:
    compact = parse_topology(four_fogs())
    general = parse_topology(as_general(compact))

    assert general == compact
    assert general.topology_id == compact.topology_id


# --- validation -------------------------------------------------------------


def _general_four_fogs() -> dict:
    return as_general(parse_topology(four_fogs()))


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda g: g["nodes"].append(dict(g["nodes"][1])), "duplicate"),
        (lambda g: g["nodes"][1].update(parent="nowhere"), "parent"),
        (lambda g: g["nodes"][1].update(level="edge"), "edge level"),
        (lambda g: g["nodes"][1].update(id="fog/0"), "id"),
        (lambda g: g["nodes"].append({"id": "cloud2", "level": "global"}), "root"),
        (lambda g: g["nodes"][1]["link_up"].update(codec="protobuf"), "codec"),
        (lambda g: g.update(levels=["global"]), "levels"),
        (lambda g: g.update(levels=["global", "fog", "fog", "edge"]), "levels"),
        (lambda g: g.update(levels=["global", "region", "fog", "edge"]), "region"),
    ],
    ids=[
        "duplicate-id",
        "unknown-parent",
        "node-at-edge-level",
        "unsafe-id",
        "two-roots",
        "unknown-codec",
        "too-few-levels",
        "repeated-level",
        "empty-level",
    ],
)
def test_invalid_trees_are_rejected(mutate, fragment: str) -> None:
    general = _general_four_fogs()
    mutate(general)

    with pytest.raises(TopologyError, match=fragment):
        parse_topology(general)


def test_parent_must_sit_on_the_level_just_above() -> None:
    raw = regions()
    raw["fog"]["nodes"][0]["parent"] = "cloud"

    with pytest.raises(TopologyError, match="level"):
        parse_topology(raw)


def test_leaf_aggregators_must_be_on_the_last_aggregation_level() -> None:
    raw = regions()
    raw["region"]["nodes"].append({"id": "east"})  # a region without fogs

    with pytest.raises(TopologyError, match="east"):
        parse_topology(raw)


# --- topology_id ------------------------------------------------------------


def _id(raw: dict) -> str:
    return parse_topology(raw).topology_id


def test_topology_id_is_a_sha256_hex_digest() -> None:
    topology_id = _id(four_fogs())

    assert len(topology_id) == 64
    int(topology_id, 16)


def test_topology_id_ignores_name_order_and_explicit_defaults() -> None:
    base = _id(four_fogs())

    renamed = four_fogs() | {"name": "otra"}
    reordered = four_fogs()
    reordered["fog"]["nodes"].reverse()
    explicit = four_fogs()
    explicit["fog"]["defaults"]["link_up"]["codec"] = "json"

    assert _id(renamed) == _id(reordered) == _id(explicit) == base


def test_topology_id_ignores_aggregation_settings() -> None:
    raw = four_fogs()
    raw["fog"]["nodes"][3]["quorum"] = 0.5

    assert _id(raw) == _id(four_fogs())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["fog"]["nodes"][0].update(link_up={"profile": "lora"}),
        lambda r: r["fog"]["nodes"][0].update(link_up={"codec": "npz"}),
        lambda r: r["fog"]["nodes"][0].update(link_up={"transport": "grpc"}),
        lambda r: r["edge"]["link_up"].update(profile="lan"),
        lambda r: r["fog"]["nodes"].pop(),
    ],
    ids=["profile", "codec", "transport", "edge-link", "node-removed"],
)
def test_topology_id_changes_with_structure_or_links(mutate) -> None:
    raw = copy.deepcopy(four_fogs())
    mutate(raw)

    assert _id(raw) != _id(four_fogs())


# --- to_graph ---------------------------------------------------------------


def test_to_graph_lists_nodes_links_and_edge_template() -> None:
    topo = parse_topology(four_fogs())

    graph = json.loads(json.dumps(topo.to_graph()))

    assert graph["topology_id"] == topo.topology_id
    assert graph["levels"] == ["global", "fog", "edge"]
    assert {n["id"]: n["role"] for n in graph["nodes"]} == {
        "cloud": "coordinator",
        "fog_a1": "aggregator",
        "fog_a2": "aggregator",
        "fog_b1": "aggregator",
        "fog_b2": "aggregator",
    }
    link = next(link for link in graph["links"] if link["src"] == "fog_b2")
    assert link == {
        "src": "fog_b2",
        "dst": "cloud",
        "transport": "mqtt",
        "codec": "json",
        "profile": "wifi",
    }
    assert graph["edge"]["link_up"]["profile"] == "4g"
    assert graph["edge"]["level"] == "edge"


# --- loading ----------------------------------------------------------------


def test_load_topology_reads_yaml(tmp_path) -> None:
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "four_fogs.yaml"
    path.write_text(yaml.safe_dump(four_fogs()), encoding="utf-8")

    assert load_topology(path) == parse_topology(four_fogs())


def test_load_topology_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(TopologyError, match="not found"):
        load_topology(tmp_path / "missing.yaml")


# --- remaining guards -------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda g: g["nodes"][1].update(level="continent"), "unknown level"),
        (lambda g: g["nodes"][0].update(link_up={"profile": "lan"}), "root"),
    ],
    ids=["unknown-level", "root-with-link"],
)
def test_general_form_guards(mutate, fragment: str) -> None:
    general = _general_four_fogs()
    mutate(general)

    with pytest.raises(TopologyError, match=fragment):
        parse_topology(general)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda r: r.update(levels="global"), "levels"),
        (lambda r: r["root"].pop("id"), r"root\.id"),
        (lambda r: r["fog"].update(children=[]), "fog: unknown keys"),
        (lambda r: r["fog"]["nodes"][0].pop("id"), r"fog\.nodes\[0\]\.id"),
    ],
    ids=[
        "levels-not-a-list",
        "root-without-id",
        "unknown-block-key",
        "node-without-id",
    ],
)
def test_compact_form_guards(mutate, fragment: str) -> None:
    raw = four_fogs()
    mutate(raw)

    with pytest.raises(TopologyError, match=fragment):
        parse_topology(raw)


def test_unknown_node_lookup_and_non_mapping_input_are_errors() -> None:
    with pytest.raises(TopologyError, match="unknown node"):
        parse_topology(four_fogs()).node("fog_z9")
    with pytest.raises(TopologyError, match="mapping"):
        parse_topology(["not", "a", "mapping"])  # type: ignore[arg-type]
