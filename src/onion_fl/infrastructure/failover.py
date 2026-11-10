from __future__ import annotations

"""Where an edge goes when its aggregator's host is down (I1 §9).

A ``failovers`` plugin chooses the edge's new parent among the candidates: the
leaf aggregators whose host is up and that have room (``max_children``). The
edge waits ``after`` seconds for its parent's host before it moves; a
candidate list that is empty leaves it where it is.
"""

from collections.abc import Mapping
from fnmatch import fnmatchcase

from pydantic import BaseModel, Field, field_validator

from onion_fl.core.registry import Registry
from onion_fl.infrastructure.model import Infrastructure, quantity
from onion_fl.infrastructure.paths import combine, hops, segment

failovers = Registry("failover")


@failovers.register(
    "none",
    title="Sin reasignación",
    description="Un edge cuyo agregador cae se queda sin padre hasta que vuelve.",
)
class NoFailover:
    after = 0.0

    def choose(
        self, edge: str, host: str, candidates: Mapping[str, str], infra: Infrastructure
    ) -> str | None:
        return None


class AfterParams(BaseModel):
    after: float = Field(
        0.0,
        ge=0,
        description="Tiempo que el edge espera al host de su agregador antes de "
        "cambiar de padre (s)",
    )

    _after = field_validator("after", mode="before")(lambda v: quantity(v, "duration"))


@failovers.register(
    "nearest_alive",
    title="El más cercano vivo",
    description="Al agregador vivo, con sitio, de menor latencia desde el host "
    "del edge.",
    params=AfterParams,
    explain="Los empates van al primero por nombre.",
)
class NearestAlive:
    def __init__(self, after: float = 0.0) -> None:
        self.after = after

    def choose(
        self, edge: str, host: str, candidates: Mapping[str, str], infra: Infrastructure
    ) -> str | None:
        def latency(parent: str) -> float:
            _, crossed = hops(infra, host, candidates[parent])
            return combine([(segment(infra, k), f) for k, f in crossed])["latency_s"]

        return min(sorted(candidates), key=latency, default=None)


class ExplicitParams(AfterParams):
    parents: dict[str, list[str]] = Field(
        min_length=1,
        description="edge, o patrón como 'swell-*', -> agregadores de reserva en "
        "orden; gana el primer patrón que encaje",
    )


@failovers.register(
    "explicit",
    title="Explícita",
    description="Cada edge al primer agregador vivo, con sitio, de su lista.",
    params=ExplicitParams,
)
class ExplicitFailover:
    def __init__(self, parents: dict[str, list[str]], after: float = 0.0) -> None:
        self.parents, self.after = parents, after

    def choose(
        self, edge: str, host: str, candidates: Mapping[str, str], infra: Infrastructure
    ) -> str | None:
        order = next(
            (p for pattern, p in self.parents.items() if fnmatchcase(edge, pattern)),
            [],
        )
        return next((p for p in order if p in candidates), None)
