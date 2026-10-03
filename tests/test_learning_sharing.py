"""Tests for sharing scopes, presets and per-link traffic (issue #83)."""

from __future__ import annotations

import pytest

from onion_fl.core.registry import PluginError
from onion_fl.core.topology import parse_topology
from onion_fl.learning.model import ModularMLPConfig
from onion_fl.learning.sharing import (
    SharingError,
    SharingPolicy,
    keys_crossing,
    keys_held_at,
    not_local,
    sharing,
    stored_at,
    traffic,
)

GROUPS = ["adapter.swell", "trunk", "head.stress_binary"]

FOUR_FOGS = parse_topology(
    {
        "levels": ["global", "fog", "edge"],
        "root": {"id": "cloud"},
        "fog": {"nodes": [{"id": "fog_a"}, {"id": "fog_b"}]},
    }
)

REGIONS = parse_topology(
    {
        "levels": ["global", "region", "fog", "edge"],
        "root": {"id": "cloud"},
        "region": {"nodes": [{"id": "north"}]},
        "fog": {"nodes": [{"id": "fog_1"}, {"id": "fog_2"}]},
    }
)


def by_link(entries: list[dict]) -> dict[tuple[str, str], list[str]]:
    return {(e["child"], e["parent"]): e["groups"] for e in entries}


# --- scopes and rules ---------------------------------------------------------


def test_default_scope_applies_to_unmatched_groups() -> None:
    policy = SharingPolicy(rules={"head.*": "local"})

    assert policy.scope_of("trunk") == "global"
    assert policy.scope_of("head.stress_binary") == "local"


def test_exact_rules_beat_patterns_and_longer_patterns_beat_shorter_ones() -> None:
    policy = SharingPolicy(
        rules={
            "head.*": "local",
            "head.stress_*": "level:fog",
            "head.stress_binary": "global",
        }
    )

    assert policy.scope_of("head.stress_binary") == "global"
    assert policy.scope_of("head.stress_3class") == "level:fog"
    assert policy.scope_of("head.other") == "local"


@pytest.mark.parametrize("scope", ["regional", "level:", "level:fog:x", "LOCAL"])
def test_invalid_scopes_are_rejected(scope: str) -> None:
    with pytest.raises(ValueError):
        SharingPolicy(rules={"head.*": scope})
    with pytest.raises(ValueError):
        SharingPolicy(default=scope)


@pytest.mark.parametrize("level", ["region", "global", "edge"])
def test_level_scopes_must_name_an_aggregation_level(level: str) -> None:
    policy = SharingPolicy(rules={"head.*": f"level:{level}"})

    with pytest.raises(SharingError, match=level):
        policy.validate_for(FOUR_FOGS)


def test_a_valid_level_scope_passes() -> None:
    SharingPolicy(rules={"head.*": "level:fog"}).validate_for(FOUR_FOGS)


# --- presets ------------------------------------------------------------------


def test_every_preset_and_custom_are_registered() -> None:
    assert sharing.names() == [
        "custom",
        "fedavg",
        "fedper",
        "harmonized",
        "independent",
        "zone",
    ]


def scopes(policy: SharingPolicy) -> dict[str, str]:
    return {group: policy.scope_of(group) for group in GROUPS}


def test_groups_a_trainer_keeps_local_are_checked_against_the_policy() -> None:
    groups = ["adapter.swell", "trunk", "head.stress_binary"]

    assert not_local(sharing.create("fedper"), groups, ["head*"]) == []
    assert not_local(sharing.create("fedavg"), groups, ["head*"]) == [
        "head.stress_binary"
    ]
    assert not_local(sharing.create("fedavg"), groups, []) == []


def test_fedavg_shares_everything_globally() -> None:
    assert set(scopes(sharing.create("fedavg")).values()) == {"global"}


def test_fedper_keeps_heads_local() -> None:
    assert scopes(sharing.create("fedper")) == {
        "adapter.swell": "global",
        "trunk": "global",
        "head.stress_binary": "local",
    }


def test_zone_aggregates_heads_per_zone_on_any_level() -> None:
    assert sharing.create("zone").scope_of("head.stress_binary") == "level:fog"
    assert (
        sharing.create("zone", {"level": "region"}).scope_of("head.x") == "level:region"
    )
    assert sharing.create("zone").scope_of("trunk") == "global"


def test_independent_needs_per_dataset_trunk_and_heads() -> None:
    policy = sharing.create("independent")

    with pytest.raises(SharingError, match="trunk"):
        policy.check_model(ModularMLPConfig())
    policy.check_model(ModularMLPConfig(trunk="per_dataset", heads="per_dataset"))


def test_harmonized_needs_a_shared_adapter() -> None:
    policy = sharing.create("harmonized")

    with pytest.raises(SharingError, match="adapters"):
        policy.check_model(ModularMLPConfig())
    policy.check_model(ModularMLPConfig(adapters="shared"))


def test_custom_policy_from_params() -> None:
    policy = sharing.create(
        "custom", {"rules": {"adapter.*": "local"}, "default": "level:fog"}
    )

    assert policy.scope_of("adapter.swell") == "local"
    assert policy.scope_of("trunk") == "level:fog"


def test_preset_params_are_validated() -> None:
    with pytest.raises(PluginError):
        sharing.create("custom", {"rules": {"trunk": "everywhere"}})


# --- traffic per link -----------------------------------------------------------


def test_fedavg_moves_every_group_on_every_link() -> None:
    entries = traffic(FOUR_FOGS, sharing.create("fedavg"), GROUPS)

    assert by_link(entries) == {
        ("fog_a", "cloud"): GROUPS,
        ("fog_b", "cloud"): GROUPS,
        ("*", "*"): GROUPS,
    }


def test_local_groups_never_travel() -> None:
    entries = traffic(FOUR_FOGS, sharing.create("fedper"), GROUPS)

    assert all("head.stress_binary" not in e["groups"] for e in entries)


def test_zone_groups_stop_at_their_level() -> None:
    links = by_link(traffic(FOUR_FOGS, sharing.create("zone"), GROUPS))

    assert links[("fog_a", "cloud")] == ["adapter.swell", "trunk"]
    assert links[("*", "*")] == GROUPS


def test_traffic_on_a_deeper_tree() -> None:
    policy = SharingPolicy(rules={"head.*": "level:region", "adapter.*": "level:fog"})

    links = by_link(traffic(REGIONS, policy, GROUPS))

    assert links[("north", "cloud")] == ["trunk"]
    assert links[("fog_1", "north")] == ["trunk", "head.stress_binary"]
    assert links[("*", "*")] == GROUPS


def test_traffic_entries_carry_their_levels() -> None:
    entries = traffic(REGIONS, sharing.create("fedavg"), GROUPS)

    levels = {(e["child_level"], e["parent_level"]) for e in entries}
    assert levels == {("region", "global"), ("fog", "region"), ("edge", "fog")}


def test_traffic_validates_the_policy_against_the_topology() -> None:
    with pytest.raises(SharingError, match="region"):
        traffic(FOUR_FOGS, sharing.create("zone", {"level": "region"}), GROUPS)


# --- what aggregators keep --------------------------------------------------------


def test_each_level_stores_the_groups_scoped_to_it() -> None:
    policy = SharingPolicy(rules={"head.*": "level:region", "adapter.*": "level:fog"})

    assert stored_at(REGIONS, "global", policy, GROUPS) == ["trunk"]
    assert stored_at(REGIONS, "region", policy, GROUPS) == ["head.stress_binary"]
    assert stored_at(REGIONS, "fog", policy, GROUPS) == ["adapter.swell"]


def test_stored_at_rejects_the_edge_level() -> None:
    with pytest.raises(SharingError, match="edge"):
        stored_at(REGIONS, "edge", sharing.create("fedavg"), GROUPS)


# --- keys for the round protocol (issue #91) -------------------------------------------

LEVELS = ["global", "region", "fog", "edge"]
KEYS = [
    "adapter.a.0.weight",
    "trunk.0.weight",
    "head.stress.weight",
    "head.stress.bias",
]


def test_keys_crossing_a_link_follow_the_scopes() -> None:
    policy = sharing.create(
        "custom", {"rules": {"head.*": "level:fog", "adapter.*": "local"}}
    )

    assert keys_crossing(KEYS, policy, LEVELS, "fog") == KEYS[1:]  # edge <-> fog
    assert keys_crossing(KEYS, policy, LEVELS, "region") == ["trunk.0.weight"]
    assert keys_crossing(KEYS, policy, LEVELS, "global") == ["trunk.0.weight"]


def test_level_scoped_keys_travel_up_to_their_level() -> None:
    policy = sharing.create("zone", {"level": "region"})

    assert "head.stress.weight" in keys_crossing(KEYS, policy, LEVELS, "region")
    assert "head.stress.weight" not in keys_crossing(KEYS, policy, LEVELS, "global")


def test_keys_held_at_each_level() -> None:
    policy = sharing.create("zone", {"level": "fog"})

    assert keys_held_at(KEYS, policy, LEVELS, "fog") == KEYS[2:]
    assert keys_held_at(KEYS, policy, LEVELS, "region") == []
    assert keys_held_at(KEYS, policy, LEVELS, "global") == KEYS[:2]
