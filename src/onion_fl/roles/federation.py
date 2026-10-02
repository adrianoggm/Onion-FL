from __future__ import annotations

"""Build the nodes and links of a topology on a runtime (spec §6, §9).

Each aggregation node reads its round settings from the topology::

    aggregator: fedavg | {name: trimmed_mean, beta: 0.2}
    server_optimizer: replace        # root only
    quorum: 1.0 | 2                  # float = fraction, int = count
    deadline: 30s
    participation: all | {name: fraction, p: 0.5}
    staleness: drop | {name: next_round, weighting: {name: polynomial, a: 0.5}}
    register_timeout: 10s

Edges hang from the leaf aggregators and use the topology's edge link.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from onion_fl.core.topology import Topology
from onion_fl.learning.aggregators import aggregators, server_optimizers
from onion_fl.learning.sharing import SharingPolicy
from onion_fl.learning.sharing import sharing as sharing_presets
from onion_fl.roles.nodes import Aggregator, Coordinator, Edge
from onion_fl.roles.policies import (
    create,
    parse_duration,
    participations,
    stalenesses,
)
from onion_fl.runtime.sim import SimRuntime


@dataclass
class EdgeSpec:
    """One edge (or evaluator) under a leaf aggregator."""

    id: str
    model: Any
    data: Any = None
    trainer: Any = None
    train: bool = True
    compute: Any = None
    availability: Any = None


@dataclass
class Federation:
    runtime: Any
    coordinator: Coordinator
    aggregators: dict[str, Aggregator] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def run(self, until: float | None = None) -> None:
        self.runtime.run(until)


def _sharing(value: str | Mapping[str, Any] | SharingPolicy | None) -> SharingPolicy:
    if isinstance(value, SharingPolicy):
        return value
    return create(sharing_presets, value, default="fedavg")


def _round_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "aggregator": create(aggregators, settings.get("aggregator"), "fedavg"),
        "quorum": settings.get("quorum", 1.0),
        "deadline": parse_duration(settings.get("deadline")),
        "participation": create(participations, settings.get("participation"), "all"),
        "staleness": create(stalenesses, settings.get("staleness"), "drop"),
        "register_timeout": parse_duration(settings.get("register_timeout")),
    }


def build_federation(
    topology: Topology,
    edges: Mapping[str, Sequence[EdgeSpec]],
    *,
    initial_state: Mapping[str, np.ndarray],
    rounds: int,
    sharing: str | Mapping[str, Any] | SharingPolicy | None = None,
    seed: int = 0,
    runtime: Any = None,
) -> Federation:
    """Coordinator, aggregators and edges of ``topology`` on ``runtime`` (a new SimRuntime)."""
    policy = _sharing(sharing)
    policy.validate_for(topology)
    runtime = runtime or SimRuntime(seed=seed)
    leaves = {leaf.id for leaf in topology.leaves()}
    stray = sorted(set(edges) - leaves)
    if stray:
        raise ValueError(f"edges under {stray}, which are not leaf aggregators")
    common = {"levels": topology.levels, "sharing": policy}
    root = topology.root
    children = {
        node.id: [c.id for c in topology.children(node.id)]
        + [e.id for e in edges.get(node.id, [])]
        for node in topology.nodes
    }
    coordinator = Coordinator(
        root.id,
        children[root.id],
        state=initial_state,
        rounds=rounds,
        server_optimizer=create(
            server_optimizers, root.settings.get("server_optimizer"), "replace"
        ),
        level=root.level,
        **common,
        **_round_settings(root.settings),
    )
    federation = Federation(runtime=runtime, coordinator=coordinator)
    runtime.add_node(coordinator)
    for node in topology.nodes:
        if node.parent is None:
            continue
        aggregator = Aggregator(
            node.id,
            children[node.id],
            parent=node.parent,
            level=node.level,
            **common,
            **_round_settings(node.settings),
        )
        federation.aggregators[node.id] = aggregator
        runtime.add_node(aggregator)
    for leaf, specs in edges.items():
        for spec in specs:
            node = Edge(
                spec.id,
                leaf,
                model=spec.model,
                data=spec.data,
                trainer=spec.trainer,
                train=spec.train,
                **common,
            )
            federation.edges[spec.id] = node
            runtime.add_node(node, compute=spec.compute, availability=spec.availability)
    for node in topology.nodes:
        if node.parent is not None:
            link = node.link_up
            runtime.add_link(
                node.id, node.parent, profile=link.profile, codec=link.codec
            )
    edge_link = topology.edge.link_up
    for leaf, specs in edges.items():
        for spec in specs:
            runtime.add_link(
                spec.id, leaf, profile=edge_link.profile, codec=edge_link.codec
            )
    return federation
