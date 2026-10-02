from __future__ import annotations

"""Round policies: participation, staleness and its weighting, quorum and deadlines (spec §6.2)."""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from onion_fl.core.registry import Registry

participations = Registry("participation")


@participations.register(
    "all", title="Todos", description="Participan todos los hijos registrados."
)
class AllChildren:
    def select(
        self, children: Sequence[str], round: int, rng: np.random.Generator
    ) -> list[str]:
        return sorted(children)


class FractionParams(BaseModel):
    p: float = Field(1.0, gt=0, le=1, description="Fracción de hijos por ronda")


@participations.register(
    "fraction",
    title="Fracción",
    description="Cada ronda participa una fracción p de los hijos, elegida con el rng del nodo.",
    params=FractionParams,
)
class Fraction:
    def __init__(self, p: float = 1.0) -> None:
        self.p = p

    def select(
        self, children: Sequence[str], round: int, rng: np.random.Generator
    ) -> list[str]:
        ordered = sorted(children)
        if not ordered:
            return []
        n = max(1, math.floor(self.p * len(ordered) + 0.5))
        return sorted(rng.choice(ordered, size=n, replace=False).tolist())


stale_weightings = Registry("stale_weighting")


class ConstantParams(BaseModel):
    factor: float = Field(1.0, gt=0, le=1)


@stale_weightings.register(
    "constant",
    title="Constante",
    description="El update tardío pesa un factor fijo.",
    params=ConstantParams,
)
class Constant:
    def __init__(self, factor: float = 1.0) -> None:
        self.value = factor

    def factor(self, staleness: int) -> float:
        return self.value


class PolynomialParams(BaseModel):
    a: float = Field(0.5, ge=0, description="Exponente: (1 + antigüedad)^-a")


@stale_weightings.register(
    "polynomial",
    title="Polinómica",
    description="El peso decae como (1 + antigüedad)^-a (FedAsync).",
    params=PolynomialParams,
)
class Polynomial:
    def __init__(self, a: float = 0.5) -> None:
        self.a = a

    def factor(self, staleness: int) -> float:
        return float((1 + staleness) ** -self.a)


stalenesses = Registry("staleness")


@stalenesses.register(
    "drop", title="Descartar", description="Los updates tardíos se descartan."
)
class Drop:
    def weight(self, staleness: int) -> float | None:
        return None


class NextRoundParams(BaseModel):
    weighting: str | dict[str, Any] = Field(
        "constant", description="Plugin de ponderación por antigüedad"
    )


@stalenesses.register(
    "next_round",
    title="Siguiente ronda",
    description="Los updates tardíos entran en la agregación siguiente, ponderados por antigüedad.",
    params=NextRoundParams,
)
class NextRound:
    def __init__(self, weighting: str | dict[str, Any] = "constant") -> None:
        self.weighting = create(stale_weightings, weighting)

    def weight(self, staleness: int) -> float | None:
        return self.weighting.factor(staleness)


def create(
    registry: Registry, spec: str | Mapping[str, Any] | None, default: str = ""
) -> Any:
    """Build a plugin from ``"name"`` or ``{"name": ..., **params}``."""
    if spec is None:
        return registry.create(default)
    if isinstance(spec, str):
        return registry.create(spec)
    params = dict(spec)
    return registry.create(params.pop("name", default), params)


def quorum_needed(quorum: float, participants: int) -> int:
    """An int is a count of children (``quorum: 2``); a float a fraction (``quorum: 1.0``)."""
    if isinstance(quorum, int) and not isinstance(quorum, bool):
        return min(quorum, participants)
    return math.ceil(quorum * participants - 1e-9)


UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_duration(value: float | str | None) -> float | None:
    """Seconds from a number or a string such as ``30s``, ``500ms``, ``2m`` or ``1h``."""
    if value is None or isinstance(value, int | float):
        return None if value is None else float(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*", str(value))
    if match is None:
        raise ValueError(
            f"duration {value!r}: use a number of seconds or 30s, 500ms, 2m, 1h"
        )
    return float(match.group(1)) * UNITS[match.group(2)]
