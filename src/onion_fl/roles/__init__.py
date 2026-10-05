"""Roles of the round protocol: coordinator, aggregator, edge and evaluator (spec §6)."""

from onion_fl.roles.federation import EdgeSpec, Federation, build_federation
from onion_fl.roles.nodes import Aggregator, Coordinator, Edge
from onion_fl.roles.snapshot import (
    FederationSnapshot,
    NodeState,
    restore_federation,
    snapshot_federation,
)

__all__ = [
    "Aggregator",
    "Coordinator",
    "Edge",
    "EdgeSpec",
    "Federation",
    "FederationSnapshot",
    "NodeState",
    "build_federation",
    "restore_federation",
    "snapshot_federation",
]
