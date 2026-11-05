from __future__ import annotations

"""Assignment of the logical federation to physical hosts (I1 §4, §5).

The coordinator and the aggregators are given a host each by name; the edges,
by an ``assignments`` plugin. :func:`resolve` turns the assignment into what the
runtimes consume: the effective profile of each logical link and the host, and
so the profile, of each node.

The strategies draw devices from the pools (``sites.<site>.pool``); named hosts
are servers. Each strategy has its own ``seed``, like the roles: changing it
moves edges between devices, never their data.
"""

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any

from pydantic import BaseModel, Field, field_validator

from onion_fl.core.registry import Registry
from onion_fl.core.topology import Topology
from onion_fl.infrastructure.model import Infrastructure, InfrastructureError
from onion_fl.infrastructure.paths import effective

assignments = Registry("assignment")


# --- strategies ----------------------------------------------------------------------


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

    def assign(self, edges: Mapping[str, str], infra: Infrastructure) -> dict[str, str]:
        out, missing = {}, []
        for edge in sorted(edges):
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


class PoolParams(BaseModel):
    seed: int = Field(0, description="Semilla propia de la asignación")
    sites: list[str] | None = Field(
        None,
        description="Sitios de cuyos pools salen los dispositivos; por defecto, todos",
    )


class _Pool:
    """Edges to distinct pool devices, both in an order drawn from the own seed."""

    name = ""

    def __init__(self, seed: int = 0, sites: list[str] | None = None) -> None:
        self.seed, self.sites = seed, sites

    def devices(self, infra: Infrastructure) -> dict[str, list[str]]:
        """Pool devices per profile, in a seeded order."""
        unknown = sorted(set(self.sites or ()) - set(infra.sites))
        if unknown:
            raise InfrastructureError(f"{self.name}: unknown sites {unknown}")
        rng = random.Random(f"{self.seed}/devices")
        out: dict[str, list[str]] = {}
        for site in sorted(self.sites or infra.sites):
            for profile, count in sorted(infra.sites[site].pool.items()):
                out.setdefault(profile, []).extend(
                    f"{site}-{profile}-{i}" for i in range(1, count + 1)
                )
        for hosts in out.values():
            hosts.sort()
            rng.shuffle(hosts)
        return out

    def edges(self, edges: Mapping[str, str]) -> list[str]:
        order = sorted(edges)
        random.Random(f"{self.seed}/edges").shuffle(order)
        return order

    def take(
        self, wanted: Mapping[str, list[str]], devices: Mapping[str, list[str]]
    ) -> dict[str, str]:
        """``wanted``: profile -> edges; each edge gets its own device of the profile."""
        out = {}
        for profile, group in sorted(wanted.items()):
            pool = devices.get(profile, [])
            if len(pool) < len(group):
                raise InfrastructureError(
                    f"{self.name}: {len(group)} edges need a {profile} device each, "
                    f"but the pool has {len(pool)}"
                )
            out |= dict(zip(group, pool, strict=False))
        return out


@assignments.register(
    "one_per_device",
    title="Un dispositivo por edge",
    description="Cada edge a un dispositivo distinto del pool, en orden sembrado.",
    params=PoolParams,
)
class OnePerDevice(_Pool):
    name = "one_per_device"

    def assign(self, edges: Mapping[str, str], infra: Infrastructure) -> dict[str, str]:
        devices = sorted(h for hosts in self.devices(infra).values() for h in hosts)
        random.Random(f"{self.seed}/devices").shuffle(devices)
        return self.take({"pool": self.edges(edges)}, {"pool": devices})


class ProfileMixParams(PoolParams):
    mix: dict[str, float] = Field(
        min_length=1, description="perfil -> fracción de los edges; suman 1"
    )

    @field_validator("mix")
    @classmethod
    def _fractions(cls, value: dict[str, float]) -> dict[str, float]:
        if any(f <= 0 for f in value.values()):
            raise ValueError("every fraction must be > 0")
        if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9):
            raise ValueError(f"the fractions add up to {sum(value.values()):g}, not 1")
        return value


def largest_remainder(weights: Mapping[str, float], n: int) -> dict[str, int]:
    """Integer counts that add up to ``n``; ties go to the first name."""
    quotas = {k: w * n for k, w in weights.items()}
    counts = {k: math.floor(q) for k, q in quotas.items()}
    left = n - sum(counts.values())
    for k in sorted(quotas, key=lambda k: (counts[k] - quotas[k], k))[:left]:
        counts[k] += 1
    return counts


@assignments.register(
    "profile_mix",
    title="Mezcla de perfiles",
    description="Una fracción de los edges en cada perfil, cada uno en su dispositivo.",
    params=ProfileMixParams,
    explain="Las fracciones se redondean por resto mayor; qué edge va a qué perfil "
    "sale de la semilla.",
)
class ProfileMix(_Pool):
    name = "profile_mix"

    def __init__(self, mix: dict[str, float], **kw: Any) -> None:
        super().__init__(**kw)
        self.mix = mix

    def assign(self, edges: Mapping[str, str], infra: Infrastructure) -> dict[str, str]:
        order, wanted = self.edges(edges), {}
        for profile, count in sorted(largest_remainder(self.mix, len(order)).items()):
            wanted[profile], order = order[:count], order[count:]
        return self.take(wanted, self.devices(infra))


class ByDatasetParams(PoolParams):
    datasets: dict[str, str] = Field(min_length=1, description="dataset -> perfil")
    default: str | None = Field(None, description="Perfil de los demás datasets")


@assignments.register(
    "by_dataset",
    title="Por dataset",
    description="El perfil de cada edge depende de su dataset (p. ej. WESAD a low_power_iot).",
    params=ByDatasetParams,
)
class ByDataset(_Pool):
    name = "by_dataset"

    def __init__(
        self, datasets: dict[str, str], default: str | None = None, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.datasets, self.default = datasets, default

    def assign(self, edges: Mapping[str, str], infra: Infrastructure) -> dict[str, str]:
        wanted: dict[str, list[str]] = {}
        for edge in self.edges(edges):
            profile = self.datasets.get(edges[edge], self.default)
            if profile is None:
                raise InfrastructureError(
                    f"by_dataset: no profile for dataset {edges[edge]!r}; add it or "
                    "set default"
                )
            wanted.setdefault(profile, []).append(edge)
        return self.take(wanted, self.devices(infra))


class ColocatedParams(PoolParams):
    per_host: int = Field(ge=1, description="Edges por dispositivo")


@assignments.register(
    "colocated",
    title="Varios edges por dispositivo",
    description="Los edges, de per_host en per_host, en dispositivos del pool.",
    params=ColocatedParams,
    explain="Sirve para estudiar la contención de los edges que comparten máquina.",
)
class Colocated(_Pool):
    name = "colocated"

    def __init__(self, per_host: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.per_host = per_host

    def assign(self, edges: Mapping[str, str], infra: Infrastructure) -> dict[str, str]:
        devices = sorted(h for hosts in self.devices(infra).values() for h in hosts)
        random.Random(f"{self.seed}/devices").shuffle(devices)
        order = self.edges(edges)
        needed = -(-len(order) // self.per_host)
        if len(devices) < needed:
            raise InfrastructureError(
                f"colocated: {len(order)} edges, {self.per_host} per device, need "
                f"{needed} devices but the pool has {len(devices)}"
            )
        return {edge: devices[i // self.per_host] for i, edge in enumerate(order)}


# --- resolution ----------------------------------------------------------------------


@dataclass(frozen=True)
class Resolved:
    """Where every logical node runs and what each logical link becomes."""

    infrastructure: Infrastructure
    hosts: dict[str, str]  # node -> host
    parents: dict[str, str] = field(default_factory=dict)  # child -> parent
    links: dict[str, tuple[list[str], dict[str, Any]]] = field(default_factory=dict)
    # child -> (path, profile of the link to its parent)
    services: dict[str, str] = field(default_factory=dict)  # service -> host

    @property
    def infrastructure_id(self) -> str:
        return self.infrastructure.infrastructure_id

    def train_rate(self, node: str) -> float | None:
        return self.infrastructure.profile_of(self.hosts[node]).train_rate

    def load(self) -> dict[str, list[str]]:
        """The logical nodes each host runs."""
        out: dict[str, list[str]] = {}
        for node, host in sorted(self.hosts.items()):
            out.setdefault(host, []).append(node)
        return dict(sorted(out.items()))

    def preview(self) -> dict[str, Any]:
        return {
            "infrastructure_id": self.infrastructure_id,
            "name": self.infrastructure.name,
            "hosts": dict(sorted(self.hosts.items())),
            "load": self.load(),
            "services": dict(sorted(self.services.items())),
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
    datasets: Mapping[str, str] | None = None,
    services: Mapping[str, str] | None = None,
) -> Resolved:
    """Assign every node and compile every link.

    ``leaves`` maps each leaf aggregator to the ids of its training edges and
    ``datasets`` each edge to its dataset; ``edges`` is an ``assignments`` plugin.
    ``services`` moves the infrastructure's services to other hosts.
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
    datasets = datasets or {}
    ids = {e: datasets.get(e, "") for leaf in sorted(leaves) for e in leaves[leaf]}
    hosts = {topology.root.id: root, **aggregators, **edges.assign(ids, infra)}
    placed = {**infra.services, **(services or {})}
    for node, host in [*hosts.items(), *placed.items()]:
        if host not in known:
            raise InfrastructureError(
                f"assignment: {node} goes to unknown host {host!r}"
            )
    parent = {n.id: n.parent for n in topology.nodes if n.parent is not None}
    parent |= {e: leaf for leaf, group in leaves.items() for e in group}
    links = {
        child: effective(infra, hosts[child], hosts[up]) for child, up in parent.items()
    }
    return Resolved(infra, hosts, parent, links, placed)
