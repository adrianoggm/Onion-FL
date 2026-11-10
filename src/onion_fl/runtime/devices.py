from __future__ import annotations

"""Device models for simulation: compute time and availability (spec §9.1).

A compute model whose ``duration`` takes an ``rng`` gets the node's compute
stream, ``node_rng(seed, "<node>/compute")``: the same seed gives the same times,
and the node's own stream (what its trainer draws) is not touched.
"""

import numpy as np
from pydantic import BaseModel, Field, field_validator

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
