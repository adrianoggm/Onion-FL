from __future__ import annotations

"""Assignment of the logical federation to physical hosts (I1 §4, §5).

The coordinator and the aggregators are given a host each by name; the edges,
by an ``assignments`` plugin. :func:`resolve` turns the assignment into what the
runtimes consume: the effective profile of each logical link and the host, and
so the profile, of each node.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry
from onion_fl.core.topology import Topology
from onion_fl.infrastructure.model import Infrastructure, InfrastructureError
from onion_fl.infrastructure.paths import effective

assignments = Registry("assignment")


class ExplicitParams(BaseModel):
    hosts: dict[str, str] = Field(
        min_length=1,
        description="edge, o patrón como 'swell-*', -> host; gana el primero que encaje",
    )


@assignments.register(
    "explicit",
    title="Explícita",
    description="Cada edge al host que se indique, por nombre o por patrón.",
    params=ExplicitParams,
    explain="Varios edges en un mismo host comparten la máquina.",
)
class Explicit:
    def __init__(self, hosts: dict[str, str]) -> None:
        self.hosts = hosts

    def assign(
        self, edges: Sequence[str], infra: Infrastructure, seed: int
    ) -> dict[str, str]:
        out, missing = {}, []
        for edge in edges:
            host = next(
                (h for pattern, h in self.hosts.items() if fnmatchcase(edge, pattern)),
                None,
            )
            if host is None:
                missing.append(edge)
            else:
                out[edge] = host
        if missing:
            raise InfrastructureError(
                f"explicit assignment: no host for edges {missing}"
            )
        return out


@dataclass(frozen=True)
class Resolved:
    """Where every logical node runs and what each logical link becomes."""

    infrastructure: Infrastructure
    hosts: dict[str, str]  # node -> host
    parents: dict[str, str] = field(default_factory=dict)  # child -> parent
    links: dict[str, tuple[list[str], dict[str, Any]]] = field(default_factory=dict)
    # child -> (path, profile of the link to its parent)

    @property
    def infrastructure_id(self) -> str:
        return self.infrastructure.infrastructure_id

    def train_rate(self, node: str) -> float | None:
        return self.infrastructure.profile_of(self.hosts[node]).train_rate

    def preview(self) -> dict[str, Any]:
        return {
            "infrastructure_id": self.infrastructure_id,
            "name": self.infrastructure.name,
            "hosts": dict(sorted(self.hosts.items())),
            "links": {
                child: {"path": path, "profile": profile}
                for child, (path, profile) in sorted(self.links.items())
            },
        }


def resolve(
    infra: Infrastructure,
    topology: Topology,
    *,
    root: str,
    aggregators: Mapping[str, str],
    edges: Any,
    leaves: Mapping[str, Sequence[str]],
    seed: int,
) -> Resolved:
    """Assign every node and compile every link.

    ``leaves`` maps each leaf aggregator to the ids of its training edges;
    ``edges`` is an ``assignments`` plugin.
    """
    known = infra.hosts()
    names = [n.id for n in topology.nodes if n.parent is not None]
    missing = sorted(set(names) - set(aggregators))
    if missing:
        raise InfrastructureError(f"assignment.aggregators: no host for {missing}")
    extra = sorted(set(aggregators) - set(names))
    if extra:
        raise InfrastructureError(
            f"assignment.aggregators: {extra} are not aggregators of the topology"
        )
    ids = [e for leaf in sorted(leaves) for e in leaves[leaf]]
    hosts = {topology.root.id: root, **aggregators, **edges.assign(ids, infra, seed)}
    for node, host in hosts.items():
        if host not in known:
            raise InfrastructureError(
                f"assignment: {node} goes to unknown host {host!r}"
            )
    parent = {n.id: n.parent for n in topology.nodes if n.parent is not None}
    parent |= {e: leaf for leaf, group in leaves.items() for e in group}
    links = {
        child: effective(infra, hosts[child], hosts[up]) for child, up in parent.items()
    }
    return Resolved(infra, hosts, parent, links)
