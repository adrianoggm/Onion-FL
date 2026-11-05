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


class _Observed:
    """What a real coordinator knows of each child: how long its past rounds took.

    The node calls ``observe`` when an answer arrives (seconds since the round
    opened, the update's metrics) and, when a round closes, for every child it
    asked that did not answer (``answered=False``, the seconds the round lasted:
    a lower bound). The true compute profile is never looked at. Children never
    observed come first, by name, so each is tried once before being judged.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, list[float]] = {}
        self.answered: dict[str, list[bool]] = {}
        self.metrics: dict[str, Mapping[str, float]] = {}

    def observe(
        self,
        child: str,
        round: int,
        seconds: float,
        answered: bool,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        self.seconds.setdefault(child, []).append(float(seconds))
        self.answered.setdefault(child, []).append(bool(answered))
        if answered and metrics:
            self.metrics[child] = dict(metrics)

    def estimated(self, child: str) -> float:
        """Mean seconds of its past rounds; one it missed counts as the round's length."""
        return float(np.mean(self.seconds[child]))

    def _split(self, children: Sequence[str]) -> tuple[list[str], list[str]]:
        ordered = sorted(children)
        return (
            [c for c in ordered if c not in self.seconds],
            [c for c in ordered if c in self.seconds],
        )

    def _top(self, children: Sequence[str], k: int, score) -> list[str]:
        """The ``k`` children never observed first, then the best by ``score``."""
        new, seen = self._split(children)
        ranked = new + sorted(seen, key=lambda c: (score(c), c))
        return sorted(ranked[:k])


class FastestKParams(BaseModel):
    k: int = Field(1, ge=1, description="Hijos por ronda")


@participations.register(
    "fastest_k",
    title="Los k más rápidos",
    description=(
        "Cada ronda participan los k hijos con menor tiempo estimado por sus rondas "
        "anteriores; los que aún no han participado van primero."
    ),
    params=FastestKParams,
)
class FastestK(_Observed):
    def __init__(self, k: int = 1) -> None:
        super().__init__()
        self.k = k

    def select(
        self, children: Sequence[str], round: int, rng: np.random.Generator
    ) -> list[str]:
        return self._top(children, self.k, self.estimated)


class DeadlineFeasibleParams(BaseModel):
    threshold: float = Field(
        0.5, ge=0, le=1, description="Probabilidad mínima de cumplir el deadline"
    )
    deadline: float | str | None = Field(
        None,
        description="Deadline que se estima (30s, 2m); sin él, responder antes del cierre",
    )
    min_rounds: int = Field(
        1, ge=1, description="Rondas observadas antes de poder excluir a un hijo"
    )


@participations.register(
    "deadline_feasible",
    title="Cumplen el deadline",
    description=(
        "Participan los hijos cuya fracción de rondas anteriores respondidas a tiempo "
        "supera un umbral; los poco observados siempre participan."
    ),
    params=DeadlineFeasibleParams,
)
class DeadlineFeasible(_Observed):
    def __init__(
        self,
        threshold: float = 0.5,
        deadline: float | str | None = None,
        min_rounds: int = 1,
    ) -> None:
        super().__init__()
        self.threshold, self.min_rounds = threshold, min_rounds
        self.deadline = parse_duration(deadline)

    def probability(self, child: str) -> float:
        """Share of its past rounds answered in time (before the close, and the deadline)."""
        hits = [
            ok and (self.deadline is None or s <= self.deadline)
            for s, ok in zip(self.seconds[child], self.answered[child], strict=True)
        ]
        return sum(hits) / len(hits)

    def select(
        self, children: Sequence[str], round: int, rng: np.random.Generator
    ) -> list[str]:
        ordered = sorted(children)
        if not ordered:
            return []
        young = [c for c in ordered if len(self.seconds.get(c, [])) < self.min_rounds]
        rest = [c for c in ordered if c not in young]
        picked = young + [c for c in rest if self.probability(c) >= self.threshold]
        if not picked:  # nobody is likely: the most likely, so the round still runs
            picked = sorted(rest, key=lambda c: (-self.probability(c), c))[:1]
        return sorted(picked)


class UtilityParams(BaseModel):
    k: int = Field(1, ge=1, description="Hijos por ronda")
    preferred: float | str | None = Field(
        None,
        description="Duración preferida de ronda (30s); sin ella, la mediana estimada",
    )
    alpha: float = Field(2.0, ge=0, description="Exponente de la penalización")
    min_loss: float = Field(
        1e-3, gt=0, description="Pérdida mínima con la que cuenta un hijo"
    )


@participations.register(
    "utility",
    title="Utilidad (Oort)",
    description=(
        "Participan los k hijos de mayor utilidad: muestras × pérdida de su último "
        "entrenamiento, penalizada por (T/t)^alpha si su tiempo estimado t supera T."
    ),
    params=UtilityParams,
    explain="Versión simplificada de Oort (Lai et al., OSDI 2021), sin exploración.",
)
class Utility(_Observed):
    def __init__(
        self,
        k: int = 1,
        preferred: float | str | None = None,
        alpha: float = 2.0,
        min_loss: float = 1e-3,
    ) -> None:
        super().__init__()
        self.k, self.alpha, self.min_loss = k, alpha, min_loss
        self.preferred = parse_duration(preferred)

    def statistical(self, child: str) -> float:
        """Samples times the training loss it last reported (at least ``min_loss``)."""
        metrics = self.metrics.get(child, {})
        loss = max(float(metrics.get("train_loss", 0.0)), self.min_loss)
        return float(metrics.get("train_examples", 0.0)) * loss

    def score(self, child: str, preferred: float) -> float:
        t = self.estimated(child)
        penalty = (preferred / t) ** self.alpha if t > preferred > 0 else 1.0
        return self.statistical(child) * penalty

    def select(
        self, children: Sequence[str], round: int, rng: np.random.Generator
    ) -> list[str]:
        _, seen = self._split(children)
        preferred = self.preferred
        if preferred is None:
            preferred = (
                float(np.median([self.estimated(c) for c in seen])) if seen else 0.0
            )
        return self._top(children, self.k, lambda c: -self.score(c, preferred))


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
    max_staleness: int | None = Field(
        None, ge=1, description="Antigüedad máxima en rondas; más vieja se descarta"
    )


@stalenesses.register(
    "next_round",
    title="Siguiente ronda",
    description="Los updates tardíos entran en la agregación siguiente, ponderados por antigüedad.",
    params=NextRoundParams,
)
class NextRound:
    def __init__(
        self,
        weighting: str | dict[str, Any] = "constant",
        max_staleness: int | None = None,
    ) -> None:
        self.weighting = create(stale_weightings, weighting)
        self.max_staleness = max_staleness

    def weight(self, staleness: int) -> float | None:
        if self.max_staleness is not None and staleness > self.max_staleness:
            return None
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
