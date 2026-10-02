from __future__ import annotations

"""Sharing policy: how far each parameter group travels (spec §7.2).

==================  ==========================================================
``global``          aggregated all the way up to the coordinator
``level:<name>``    aggregated up to that level; its aggregator keeps it and
                    injects it into the model it sends down to its subtree
``local``           never leaves the edge
==================  ==========================================================

A group crosses the link between a child and its parent, in both directions,
when its scope is ``global`` or ``level:X`` with X at the parent's level or
above it.
"""

import fnmatch
import re
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onion_fl.core.registry import Registry
from onion_fl.core.topology import Topology
from onion_fl.learning.model import ModularMLPConfig, group_of

SCOPE = r"^(global|local|level:[A-Za-z0-9_.-]+)$"


class SharingError(ValueError):
    """A sharing policy does not fit the topology or the model."""


def _check_rules(rules: dict[str, str]) -> dict[str, str]:
    for pattern, scope in rules.items():
        if not re.fullmatch(SCOPE, scope):
            raise ValueError(
                f"rule {pattern!r}: scope {scope!r} must be global, local or level:<name>"
            )
    return rules


class SharingPolicy(BaseModel):
    """Scope of each parameter group, by exact name or glob pattern."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "custom"
    rules: dict[str, str] = Field(default_factory=dict)
    default: str = Field("global", pattern=SCOPE)
    model_requirements: dict[str, str] = Field(default_factory=dict)

    _valid_rules = field_validator("rules")(_check_rules)

    def scope_of(self, group: str) -> str:
        """Exact rule first, then the longest matching pattern, then the default."""
        if group in self.rules:
            return self.rules[group]
        matches = [p for p in self.rules if fnmatch.fnmatchcase(group, p)]
        return self.rules[max(matches, key=len)] if matches else self.default

    def scopes(self) -> list[str]:
        return [self.default, *self.rules.values()]

    def validate_for(self, topology: Topology) -> None:
        """Every ``level:<name>`` must be an aggregation level between root and edge."""
        allowed = topology.levels[1:-1]
        for scope in self.scopes():
            if scope.startswith("level:") and scope[len("level:") :] not in allowed:
                raise SharingError(
                    f"scope {scope!r}: the level must be one of the aggregation levels {allowed}"
                )

    def check_model(self, config: ModularMLPConfig) -> None:
        for field, required in self.model_requirements.items():
            actual = getattr(config, field)
            if actual != required:
                raise SharingError(
                    f"sharing {self.name!r} needs the model with {field}={required!r}, "
                    f"got {actual!r}"
                )


def crosses(scope: str, parent_level: str, levels: Sequence[str]) -> bool:
    """Does a group with ``scope`` cross a link whose parent sits at ``parent_level``?"""
    if scope == "global":
        return True
    if scope == "local":
        return False
    depth = list(levels).index
    return depth(scope[len("level:") :]) <= depth(parent_level)


def keys_crossing(
    keys: Iterable[str],
    policy: SharingPolicy,
    levels: Sequence[str],
    parent_level: str,
) -> list[str]:
    """State keys that travel on a link (both ways) whose parent is at ``parent_level``."""
    return [
        k for k in keys if crosses(policy.scope_of(group_of(k)), parent_level, levels)
    ]


def keys_held_at(
    keys: Iterable[str], policy: SharingPolicy, levels: Sequence[str], level: str
) -> list[str]:
    """Keys an aggregator of ``level`` keeps: global at the root, ``level:<level>`` below."""
    wanted = "global" if level == levels[0] else f"level:{level}"
    return [k for k in keys if policy.scope_of(group_of(k)) == wanted]


def traffic(
    topology: Topology, policy: SharingPolicy, groups: Sequence[str]
) -> list[dict[str, Any]]:
    """Groups that cross each link, for the front's preview.

    One entry per topology link plus one ``("*", "*")`` entry for the links
    between the leaf aggregators and their edges. The rule is symmetric, so
    ``groups`` is what goes up and what comes down.
    """
    policy.validate_for(topology)
    entries = [
        {
            "child": node.id,
            "parent": node.parent,
            "child_level": node.level,
            "parent_level": topology.node(node.parent).level,
        }
        for node in topology.nodes
        if node.parent is not None
    ]
    entries.append(
        {
            "child": "*",
            "parent": "*",
            "child_level": topology.levels[-1],
            "parent_level": topology.levels[-2],
        }
    )
    for entry in entries:
        entry["groups"] = [
            g
            for g in groups
            if crosses(policy.scope_of(g), entry["parent_level"], topology.levels)
        ]
    return entries


def stored_at(
    topology: Topology, level: str, policy: SharingPolicy, groups: Sequence[str]
) -> list[str]:
    """Groups an aggregator of ``level`` keeps: the global model at the root, its zone model below."""
    if level not in topology.levels[:-1]:
        raise SharingError(
            f"level {level!r} is not an aggregation level of {topology.levels[:-1]}"
        )
    wanted = "global" if level == topology.levels[0] else f"level:{level}"
    return [g for g in groups if policy.scope_of(g) == wanted]


# --- presets ------------------------------------------------------------------

sharing = Registry("sharing")


@sharing.register(
    "fedavg",
    title="FedAvg",
    description="Todos los grupos se agregan hasta el coordinador.",
    explain="Un único modelo global para todos los edges.",
)
def fedavg() -> SharingPolicy:
    return SharingPolicy(name="fedavg")


@sharing.register(
    "fedper",
    title="FedPer",
    description="Tronco y adaptadores globales; las cabezas no salen del edge.",
    explain="Personalización: cada edge conserva su propia cabeza.",
)
def fedper() -> SharingPolicy:
    return SharingPolicy(name="fedper", rules={"head.*": "local"})


class ZoneParams(BaseModel):
    level: str = Field("fog", description="Nivel donde se agrega cada cabeza de zona")


@sharing.register(
    "zone",
    title="Por zona",
    description="Tronco y adaptadores globales; las cabezas se agregan por zona.",
    params=ZoneParams,
    explain="Cada agregador del nivel elegido mantiene un modelo de zona útil para sus edges.",
)
def zone(level: str = "fog") -> SharingPolicy:
    return SharingPolicy(name="zone", rules={"head.*": f"level:{level}"})


@sharing.register(
    "independent",
    title="Modelos independientes",
    description="Un modelo por dataset sobre la misma infraestructura.",
    explain="Exige trunk y heads por dataset: los datasets no comparten ningún parámetro.",
)
def independent() -> SharingPolicy:
    return SharingPolicy(
        name="independent",
        model_requirements={"trunk": "per_dataset", "heads": "per_dataset"},
    )


@sharing.register(
    "harmonized",
    title="Features armonizadas",
    description="Un adaptador común sobre features armonizadas; todo global.",
    explain="Exige adapters='shared': todos los datasets deben tener las mismas features.",
)
def harmonized() -> SharingPolicy:
    return SharingPolicy(name="harmonized", model_requirements={"adapters": "shared"})


class CustomParams(BaseModel):
    rules: dict[str, str] = Field(
        default_factory=dict, description="Patrón de grupo -> alcance"
    )
    default: str = Field("global", pattern=SCOPE)

    _valid_rules = field_validator("rules")(_check_rules)


@sharing.register(
    "custom",
    title="Personalizada",
    description="Alcances propios por grupo o patrón (adapter.*, trunk, head.<tarea>...).",
    params=CustomParams,
)
def custom(
    rules: dict[str, str] | None = None, default: str = "global"
) -> SharingPolicy:
    return SharingPolicy(rules=rules or {}, default=default)
