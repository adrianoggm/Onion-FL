"""Roles of the round protocol: coordinator, aggregator, edge and evaluator (spec §6)."""

from onion_fl.roles.federation import EdgeSpec, Federation, build_federation
from onion_fl.roles.nodes import Aggregator, Coordinator, Edge

__all__ = [
    "Aggregator",
    "Coordinator",
    "Edge",
    "EdgeSpec",
    "Federation",
    "build_federation",
]
