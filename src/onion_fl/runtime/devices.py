from __future__ import annotations

"""Device models for simulation: compute time and availability (spec §9.1).

A compute model whose ``duration`` takes an ``rng`` gets the node's compute
stream, ``node_rng(seed, "<node>/compute")``: the same seed gives the same times,
and the node's own stream (what its trainer draws) is not touched.
"""

import bisect
import math

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from onion_fl.core.context import node_rng
from onion_fl.core.registry import Registry

compute_models = Registry("compute")
availability_models = Registry("availability")


# --- compute -----------------------------------------------------------------


class SamplesPerSecondParams(BaseModel):
    samples_per_second: float = Field(
        gt=0, description="Muestras procesadas por segundo"
    )
    overhead_s: float = Field(
        0.0, ge=0, description="Coste fijo de cada tarea con trabajo"
    )


@compute_models.register(
    "samples_per_second",
    title="Muestras por segundo",
    description="Duración = overhead + muestras declaradas / muestras_por_segundo.",
    params=SamplesPerSecondParams,
    explain="Determinista: permite simular dispositivos lentos y rezagados sin depender de esta máquina.",
)
class SamplesPerSecond:
    def __init__(self, samples_per_second: float, overhead_s: float = 0.0) -> None:
        self.samples_per_second = samples_per_second
        self.overhead_s = overhead_s

    def duration(self, samples: float, wall_s: float) -> float:
        if samples <= 0:
            return 0.0
        return self.overhead_s + samples / self.samples_per_second


class MeasuredParams(BaseModel):
    factor: float = Field(1.0, gt=0, description="Multiplicador del tiempo de pared")


@compute_models.register(
    "measured",
    title="Tiempo medido",
    description="Duración = tiempo de pared real del manejador × factor.",
    params=MeasuredParams,
    explain="Más realista, pero no determinista: dos ejecuciones con la misma semilla pueden diferir.",
)
class Measured:
    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def duration(self, samples: float, wall_s: float) -> float:
        return wall_s * self.factor


class LognormalRateParams(BaseModel):
    samples_per_second: float = Field(
        gt=0, description="Muestras por segundo que dan la duración media"
    )
    sigma: float = Field(
        ge=0, description="Dispersión: desviación típica del logaritmo de la tasa"
    )
    overhead_s: float = Field(
        0.0, ge=0, description="Coste fijo de cada tarea con trabajo"
    )


@compute_models.register(
    "lognormal_rate",
    title="Tasa lognormal",
    description=(
        "Cada tarea con trabajo tiene su propia tasa lognormal; de media dura "
        "overhead + muestras / muestras_por_segundo."
    ),
    params=LognormalRateParams,
    explain=(
        "Hace que cumplir el deadline sea una probabilidad y no un sí o no fijo; "
        "la misma semilla da los mismos tiempos."
    ),
)
class LognormalRate:
    """``duration = overhead + samples / samples_per_second * exp(sigma Z - sigma²/2)``
    with Z standard normal: the rate is lognormal and the mean duration is that of
    ``samples_per_second`` at the same rate."""

    def __init__(
        self, samples_per_second: float, sigma: float, overhead_s: float = 0.0
    ) -> None:
        self.samples_per_second = samples_per_second
        self.sigma = sigma
        self.overhead_s = overhead_s

    def duration(
        self, samples: float, wall_s: float, rng: np.random.Generator
    ) -> float:
        if samples <= 0:
            return 0.0
        factor = np.exp(self.sigma * rng.standard_normal() - self.sigma**2 / 2)
        return self.overhead_s + samples / self.samples_per_second * float(factor)


class TraceParams(BaseModel):
    durations: list[float] = Field(
        min_length=1, description="Duraciones medidas de tareas con trabajo (s)"
    )

    @field_validator("durations")
    @classmethod
    def _non_negative(cls, durations: list[float]) -> list[float]:
        if any(d < 0 for d in durations):
            raise ValueError("durations must be >= 0")
        return durations


@compute_models.register(
    "trace",
    title="Tiempos medidos",
    description=(
        "Cada tarea con trabajo dura una de las duraciones medidas, elegida al "
        "azar con la semilla del nodo."
    ),
    params=TraceParams,
    explain="Reproduce la distribución observada en dispositivos reales.",
)
class Trace:
    """Each task with work lasts one of ``durations``, drawn uniformly: the
    empirical distribution of the trace, whatever the samples."""

    def __init__(self, durations: list[float]) -> None:
        self.durations = list(durations)

    def duration(
        self, samples: float, wall_s: float, rng: np.random.Generator
    ) -> float:
        if samples <= 0:
            return 0.0
        return float(self.durations[rng.integers(len(self.durations))])


# --- availability ------------------------------------------------------------


@availability_models.register(
    "always", title="Siempre disponible", description="El nodo nunca se cae."
)
class Always:
    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return True


class BernoulliParams(BaseModel):
    p: float = Field(
        ge=0, le=1, description="Probabilidad de estar desconectado en una ronda"
    )


@availability_models.register(
    "bernoulli",
    title="Bernoulli por ronda",
    description="En cada ronda el nodo está desconectado con probabilidad p.",
    params=BernoulliParams,
    explain="Solo afecta a mensajes de una ronda; hello y control siempre llegan.",
)
class Bernoulli:
    def __init__(self, p: float) -> None:
        self.p = p

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        if round is None:
            return True
        return node_rng(seed, f"{node_id}/availability/{round}").random() >= self.p


class ScheduleParams(BaseModel):
    offline: list[tuple[float, float]] = Field(
        description="Ventanas [inicio, fin) sin conexión"
    )

    @field_validator("offline")
    @classmethod
    def _ordered(cls, windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for start, end in windows:
            if not 0 <= start < end:
                raise ValueError(
                    f"window [{start}, {end}) must satisfy 0 <= start < end"
                )
        return windows


@availability_models.register(
    "schedule",
    title="Calendario",
    description="Desconectado dentro de las ventanas [inicio, fin) indicadas.",
    params=ScheduleParams,
)
class Schedule:
    def __init__(self, offline: list[tuple[float, float]]) -> None:
        self.offline = offline

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return not any(start <= t < end for start, end in self.offline)

    def up_again(self, t: float) -> float:
        """When a node offline at ``t`` is up again; windows may touch."""
        while not self.is_up(t, None, "", 0):
            t = max(end for start, end in self.offline if start <= t < end)
        return t


class CrashAtParams(BaseModel):
    t: float = Field(ge=0, description="Instante virtual de la caída")


@availability_models.register(
    "crash_at",
    title="Caída definitiva",
    description="Disponible hasta el instante t; después, caído para siempre.",
    params=CrashAtParams,
)
class CrashAt:
    def __init__(self, t: float) -> None:
        self.t = t

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        return t < self.t


class WindowsParams(BaseModel):
    active: list[float] = Field(
        min_length=1, description="Inicio de cada ventana con conexión (s)"
    )
    inactive: list[float] = Field(
        min_length=1, description="Fin de cada ventana con conexión (s)"
    )
    period: float | None = Field(
        None, gt=0, description="Cada cuánto se repiten las ventanas; sin él, no"
    )

    @model_validator(mode="after")
    def _ordered(self) -> WindowsParams:
        if len(self.active) != len(self.inactive):
            raise ValueError("active and inactive must have the same length")
        edges = [
            x for pair in zip(self.active, self.inactive, strict=True) for x in pair
        ]
        if edges[0] < 0 or any(b < a for a, b in zip(edges, edges[1:], strict=False)):
            raise ValueError("windows must be ordered: 0 <= active[i] <= inactive[i]")
        if self.period is not None and edges[-1] > self.period:
            raise ValueError(f"the last window ends after the period {self.period}")
        return self


@availability_models.register(
    "windows",
    title="Ventanas de conexión",
    description=(
        "Conectado dentro de las ventanas [inicio, fin]; con periodo, se repiten "
        "cada periodo (la traza de disponibilidad de FedScale)."
    ),
    params=WindowsParams,
    explain="Reproduce cuándo se conectó un dispositivo real; fuera de ellas, nada llega.",
)
class Windows:
    """Up inside ``[active[i], inactive[i]]`` of ``t`` modulo ``period``, as
    FedScale's ``ClientMetadata.is_active``; without a period, the last window
    is the last time it is up."""

    def __init__(
        self, active: list[float], inactive: list[float], period: float | None = None
    ) -> None:
        self.active, self.inactive, self.period = list(active), list(inactive), period
        # a time computed from a cycle start may fall an ulp before its window
        self.eps = 1e-9 * max(1.0, period or 0.0, self.inactive[-1])

    def _cycle(self, t: float) -> tuple[float, float]:
        if self.period is None:
            return 0.0, t
        start = math.floor(t / self.period) * self.period
        return start, t - start

    def is_up(self, t: float, round: int | None, node_id: str, seed: int) -> bool:
        _, tt = self._cycle(t)
        i = bisect.bisect_right(self.active, tt + self.eps) - 1
        return i >= 0 and tt <= self.inactive[i] + self.eps

    def up_again(self, t: float) -> float | None:
        """When a node offline at ``t`` is up again; None if never."""
        if self.is_up(t, None, "", 0):
            return t
        start, tt = self._cycle(t)
        j = bisect.bisect_right(self.active, tt + self.eps)
        if j < len(self.active):
            return start + self.active[j]
        return None if self.period is None else start + self.period + self.active[0]
